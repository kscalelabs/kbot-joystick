"""Defines simple task for training a joystick walking policy for K-Bot."""

import asyncio
import functools
import math
from dataclasses import dataclass
from typing import Collection, Self

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import optax
import xax
from jaxtyping import Array, PRNGKeyArray

# These are in the order of the neural network outputs.
# Joint name, target position, penalty weight.
ZEROS: list[tuple[str, float, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0, 1.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0), 1.0),
    ("dof_right_shoulder_yaw_02", 0.0, 1.0),
    ("dof_right_elbow_02", math.radians(90.0), 1.0),
    ("dof_right_wrist_00", 0.0, 1.0),
    ("dof_left_shoulder_pitch_03", 0.0, 1.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0), 1.0),
    ("dof_left_shoulder_yaw_02", 0.0, 1.0),
    ("dof_left_elbow_02", math.radians(-90.0), 1.0),
    ("dof_left_wrist_00", 0.0, 1.0),
    ("dof_right_hip_pitch_04", math.radians(-20.0), 0.01),
    ("dof_right_hip_roll_03", math.radians(-0.0), 1.0),
    ("dof_right_hip_yaw_03", 0.0, 2.0),
    ("dof_right_knee_04", math.radians(-50.0), 0.01),
    ("dof_right_ankle_02", math.radians(30.0), 1.0),
    ("dof_left_hip_pitch_04", math.radians(20.0), 0.01),
    ("dof_left_hip_roll_03", math.radians(0.0), 2.0),
    ("dof_left_hip_yaw_03", 0.0, 2.0),
    ("dof_left_knee_04", math.radians(50.0), 0.01),
    ("dof_left_ankle_02", math.radians(-30.0), 1.0),
]


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the RNN.",
    )
    depth: int = xax.field(
        value=3,
        help="The depth for the RNN",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )
    use_acc_gyro: bool = xax.field(
        value=True,
        help="Whether to use the IMU acceleration and gyroscope observations.",
    )
    stand_still_threshold: float = xax.field(
        value=0.1,
        help="The mininum command magnitude. Don't like tiny Vx commands like 0.00123",
    )
    # Curriculum parameters.
    num_curriculum_levels: int = xax.field(
        value=100,
        help="The number of curriculum levels to use.",
    )
    increase_threshold: float = xax.field(
        value=5.0,
        help="Increase the curriculum level when the mean trajectory length is above this threshold.",
    )
    decrease_threshold: float = xax.field(
        value=1.0,
        help="Decrease the curriculum level when the mean trajectory length is below this threshold.",
    )
    min_level_steps: int = xax.field(
        value=1,
        help="The minimum number of steps to wait before changing the curriculum level.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )


# TODO put this in xax?
def rotate_quat_by_quat(quat_to_rotate: Array, rotating_quat: Array, inverse: bool = False, eps: float = 1e-6) -> Array:
    """Rotates one quaternion by another quaternion through quaternion multiplication.
    
    This performs the operation: rotating_quat * quat_to_rotate * rotating_quat^(-1) if inverse=False
    or rotating_quat^(-1) * quat_to_rotate * rotating_quat if inverse=True
    
    Args:
        quat_to_rotate: The quaternion being rotated (w,x,y,z), shape (*, 4)
        rotating_quat: The quaternion performing the rotation (w,x,y,z), shape (*, 4)
        inverse: If True, rotate by the inverse of rotating_quat
        eps: Small epsilon value to avoid division by zero in normalization
        
    Returns:
        The rotated quaternion (w,x,y,z), shape (*, 4)
    """
    # Normalize both quaternions
    quat_to_rotate = quat_to_rotate / (jnp.linalg.norm(quat_to_rotate, axis=-1, keepdims=True) + eps)
    rotating_quat = rotating_quat / (jnp.linalg.norm(rotating_quat, axis=-1, keepdims=True) + eps)
    
    # If inverse requested, conjugate the rotating quaternion (negate x,y,z components)
    if inverse:
        rotating_quat = rotating_quat.at[..., 1:].multiply(-1)
    
    # Extract components of both quaternions
    w1, x1, y1, z1 = jnp.split(rotating_quat, 4, axis=-1)        # rotating quaternion 
    w2, x2, y2, z2 = jnp.split(quat_to_rotate, 4, axis=-1)      # quaternion being rotated
    
    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    result = jnp.concatenate([w, x, y, z], axis=-1)
    
    # Normalize result
    return result / (jnp.linalg.norm(result, axis=-1, keepdims=True) + eps)


@attrs.define(frozen=True, kw_only=True)
class ContactForcePenalty(ksim.Reward):
    """Penalises vertical forces above threshold."""

    scale: float = -1.0
    max_contact_force: float = 350.0
    sensor_names: tuple[str, ...]

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        forces = jnp.stack([traj.obs[n] for n in self.sensor_names], axis=-1)
        cost = jnp.clip(jnp.abs(forces[..., 2, :]) - self.max_contact_force, 0)
        return jnp.sum(cost, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetSlipPenalty(ksim.Reward):
    """Penalises COM motion while feet are in contact."""

    scale: float = -1.0
    com_vel_obs_name: str = "center_of_mass_velocity_observation"
    feet_contact_obs_name: str = "feet_contact_observation"

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        vel = jnp.linalg.norm(traj.obs[self.com_vel_obs_name][..., :2], axis=-1, keepdims=True)
        contact = traj.obs[self.feet_contact_obs_name]
        return jnp.sum(vel * contact, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class SingleFootContactReward(ksim.Reward):
    """Reward that encourages a single-foot contact pattern during locomotion.

    The reward returns ``1`` whenever exactly one foot is in contact with the
    ground and the robot is commanded to move (i.e. *not* stand-still).  When
    both feet are in contact or both are in the air the reward is ``0``.

    For stand-still commands 
    the reward is always ``1`` so as not to penalise recovery steps that may be
    required in order to keep balance while standing.
    """

    scale: float = 1.0
    feet_contact_obs_name: str = "feet_contact_observation"
    linear_velocity_cmd_name: str = "linear_velocity_command"
    angular_velocity_cmd_name: str = "angular_velocity_command"
    ctrl_dt: float = 0.02
    window_size: float = 0.2  # seconds

    def _single_contact(self, contact: Array) -> Array:
        """Returns a boolean mask that is ``True`` when exactly one foot is in contact."""
        left_contact = jnp.any(contact[..., :2] > 0.5, axis=-1)
        right_contact = jnp.any(contact[..., 2:] > 0.5, axis=-1)
        return jnp.logical_xor(left_contact, right_contact)

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        """Implements Eq. (14) – single-foot contact with 0.2 s window.

        • reward = 1 if, within the past 0.2 s, there was at least one frame
        where *exactly one* foot touched the ground **and** the command
        magnitude is nonzero
        • While standing (|cmd| ≤ threshold) the reward is fixed to 1.
        """
        contact = traj.obs[self.feet_contact_obs_name]
        left = jnp.any(contact[..., :2] > 0.5, axis=-1)
        right = jnp.any(contact[..., 2:] > 0.5, axis=-1)
        single = jnp.logical_xor(left, right)

        k = int(self.window_size / self.ctrl_dt)

        padded = jnp.concatenate([jnp.zeros_like(single[..., :k]), single], axis=-1)

        window_any = (
            jax.lax.reduce_window(
                padded.astype(jnp.int32),
                0,
                jax.lax.add,
                window_dimensions=(k + 1,),
                window_strides=(1,),
                padding="valid",
            )
            > 0
        )

        reward = window_any.astype(jnp.float32)

        lin_cmd = traj.command[self.linear_velocity_cmd_name]
        ang_cmd = traj.command[self.angular_velocity_cmd_name]
        cmd_mag = jnp.linalg.norm(jnp.concatenate([lin_cmd, ang_cmd], axis=-1), axis=-1)
        reward = jnp.where(cmd_mag <= 0.01, 1.0, reward)

        return reward


@attrs.define(frozen=True, kw_only=True)
class FeetAirtimeReward(ksim.Reward):
    """Encourages reasonable step frequency by rewarding foot airtime.

    Each foot accumulates a *positive* reward proportional to the duration it
    spends in the air (i.e. not in contact).  Whenever the foot touches the
    ground a small fixed penalty (``touchdown_penalty``) is applied.  This leads
    to larger rewards for longer swing phases while discouraging excessively
    rapid stepping.
    """

    scale: float = 1.0
    feet_contact_obs_name: str = "feet_contact_observation"
    ctrl_dt: float = 0.02
    touchdown_penalty: float = 0.4

    def _airtime_sequence(self, contact_bool: Array) -> Array:
        """Returns an array with the airtime (in seconds) for each timestep."""

        def _body(time_since_liftoff: Array, is_contact: Array) -> tuple[Array, Array]:
            new_time = jnp.where(is_contact, 0.0, time_since_liftoff + self.ctrl_dt)
            return new_time, new_time

        _, airtime = jax.lax.scan(_body, 0.0, contact_bool)
        return airtime

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        contact = traj.obs[self.feet_contact_obs_name]
        left_in, right_in = contact[..., 0] > 0.5, contact[..., 1] > 0.5

        # airtime counters
        left_air = self._airtime_sequence(left_in)
        right_air = self._airtime_sequence(right_in)

        # touchdown boolean (0→1 transition)
        def touchdown(c: Array) -> Array:
            prev = jnp.concatenate([jnp.array([False]), c[:-1]])
            return jnp.logical_and(c, jnp.logical_not(prev))

        td_l = jnp.roll(touchdown(left_in), shift=-1).at[-1].set(0)
        td_r = jnp.roll(touchdown(right_in), shift=-1).at[-1].set(0)

        swing_reward = (left_air - self.touchdown_penalty) * td_l.astype(jnp.float32) + (right_air - self.touchdown_penalty) * td_r.astype(jnp.float32)

        # standing mask
        stand = (jnp.linalg.norm(traj.command["linear_velocity_command"], axis=-1) < 1e-3) & (
            jnp.abs(traj.command["angular_velocity_command"][..., 0]) < 1e-3
        )

        return jnp.where(stand, 0.0, swing_reward)


@attrs.define(frozen=True, kw_only=True)
class JointPositionPenalty(ksim.JointDeviationPenalty):
    @classmethod
    def create_from_names(
        cls,
        names: list[str],
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        zeros = {k: v for k, v, _ in ZEROS}
        weights = {k: v for k, _, v in ZEROS}
        joint_targets = [zeros[name] for name in names]
        joint_weights = [weights[name] for name in names]

        return cls.create(
            physics_model=physics_model,
            joint_names=tuple(names),
            joint_targets=tuple(joint_targets),
            joint_weights=tuple(joint_weights),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class BentArmPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_right_shoulder_pitch_03",
                "dof_right_shoulder_roll_03",
                "dof_right_shoulder_yaw_02",
                "dof_right_elbow_02",
                "dof_right_wrist_00",
                "dof_left_shoulder_pitch_03",
                "dof_left_shoulder_roll_03",
                "dof_left_shoulder_yaw_02",
                "dof_left_elbow_02",
                "dof_left_wrist_00",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class StraightLegPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_left_hip_roll_03",
                "dof_left_hip_yaw_03",
                "dof_right_hip_roll_03",
                "dof_right_hip_yaw_03",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)
    linvel_obs_name: str = attrs.field(default="sensor_observation_base_site_linvel")
    command_name: str = attrs.field(default="linear_velocity_command")
    norm: xax.NormType = attrs.field(default="l2")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # need to get lin vel obs from sensor, because xvel is not available in Trajectory.
        if self.linvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.linvel_obs_name} not found; add it as an observation in your task.")

        # Get global frame velocities
        global_vel = trajectory.obs[self.linvel_obs_name]

        # get base quat, only yaw.
        # careful to only rotate in z, disregard rx and ry, bad conflict with roll and pitch.
        base_euler = xax.quat_to_euler(trajectory.xquat[..., 1, :])
        base_euler = base_euler.at[..., 0].set(0.0)
        base_euler = base_euler.at[..., 1].set(0.0)
        base_z_quat = xax.euler_to_quat(base_euler)

        # rotate local frame commands to global frame
        robot_vel_cmd = jnp.zeros_like(global_vel).at[..., :2].set(trajectory.command[self.command_name])
        global_vel_cmd = xax.rotate_vector_by_quat(robot_vel_cmd, base_z_quat, inverse=False)

        # drop vz. vz conflicts with base height reward.
        global_vel_xy_cmd = global_vel_cmd[..., :2]
        global_vel_xy = global_vel[..., :2]

        # now compute error. special trick: different kernels for standing and walking.
        zero_cmd_mask = (
            jnp.linalg.norm(trajectory.command["linear_velocity_command"], axis=-1) < 1e-3
        ) & (
            jnp.abs(trajectory.command["angular_velocity_command"][..., 0]) < 1e-3
        )
        vel_error = jnp.linalg.norm(global_vel_xy - global_vel_xy_cmd, axis=-1)
        error = jnp.where(zero_cmd_mask, vel_error, jnp.square(vel_error))
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the heading using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="angular_velocity_command")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        base_yaw = xax.quat_to_euler(trajectory.xquat[..., 1, :])[:, 2]
        base_yaw_cmd = trajectory.command[self.command_name][..., 2]

        base_yaw_quat = xax.euler_to_quat(jnp.stack([
            jnp.zeros_like(base_yaw_cmd),
            jnp.zeros_like(base_yaw_cmd),
            base_yaw
        ], axis=-1))
        base_yaw_target_quat = xax.euler_to_quat(jnp.stack([
            jnp.zeros_like(base_yaw_cmd),
            jnp.zeros_like(base_yaw_cmd),
            base_yaw_cmd
        ], axis=-1))

        # Compute quaternion error
        quat_error = 1 - jnp.sum(base_yaw_target_quat * base_yaw_quat, axis=-1)**2
        return jnp.exp(-quat_error / self.error_scale)


@attrs.define(frozen=True)
class XYOrientationReward(ksim.Reward):
    """Reward for tracking the xy base orientation using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="xyorientation_command")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        euler_orientation = xax.quat_to_euler(trajectory.xquat[..., 1, :])
        euler_orientation = euler_orientation.at[..., 2].set(0.0)  # ignore yaw
        base_xy_quat = xax.euler_to_quat(euler_orientation)

        commanded_euler = jnp.stack([
            trajectory.command[self.command_name][..., 0],
            trajectory.command[self.command_name][..., 1],
            jnp.zeros_like(trajectory.command[self.command_name][..., 0])
        ], axis=-1)
        base_xy_quat_cmd = xax.euler_to_quat(commanded_euler)

        quat_error = 1 - jnp.sum(base_xy_quat_cmd * base_xy_quat, axis=-1)**2
        return jnp.exp(-quat_error / self.error_scale)


@attrs.define(frozen=True)
class BaseHeightReward(ksim.Reward):
    """Reward for keeping the base height at the commanded height."""

    error_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        current_height = trajectory.xpos[..., 1, 2] # 1st body, because world is 0. 2nd element is z.
        commanded_height = trajectory.command["base_height_command"][..., 0]
        
        height_error = jnp.abs(current_height - commanded_height)
        return jnp.exp(-height_error / self.error_scale)

@attrs.define(frozen=True)
class FeetPositionReward(ksim.Reward):
    """ Reward for keeping the feet next to each other when standing still."""

    error_scale: float = attrs.field(default=0.25)
    stance_width: float = attrs.field(default=0.3)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        feet_pos = trajectory.obs["feet_position_observation"]
        left_foot_pos = feet_pos[..., :3]  
        right_foot_pos = feet_pos[..., 3:]

        # standing?
        zero_cmd_mask = (
            jnp.linalg.norm(trajectory.command["linear_velocity_command"], axis=-1) < 1e-3
        ) & (
            jnp.abs(trajectory.command["angular_velocity_command"][..., 0]) < 1e-3
        )

        # Calculate stance errors
        stance_x_error = jnp.abs(left_foot_pos[..., 0] - right_foot_pos[..., 0])
        stance_y_error = jnp.abs(jnp.abs(left_foot_pos[..., 1] - right_foot_pos[..., 1]) - self.stance_width)
        stance_error = jnp.where(zero_cmd_mask, stance_x_error + stance_y_error, 0.0)**2 # ^2 to make less sensitive to small errors, smooth kernel

        return jnp.exp(-stance_error / self.error_scale)

@attrs.define(frozen=True, kw_only=True)
class FeetContactObservation(ksim.FeetContactObservation):
    """Flattened (4,) contact flags of both feet."""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return super().observe(state, curriculum_level, rng).flatten()


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    foot_left_idx: int
    foot_right_idx: int
    floor_threshold: float = 0.0
    in_robot_frame: bool = True

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        foot_left_site_name: str,
        foot_right_site_name: str,
        floor_threshold: float = 0.0,
        in_robot_frame: bool = True,
    ) -> Self:
        fl = ksim.get_site_data_idx_from_name(physics_model, foot_left_site_name)
        fr = ksim.get_site_data_idx_from_name(physics_model, foot_right_site_name)
        return cls(foot_left_idx=fl, foot_right_idx=fr, floor_threshold=floor_threshold, in_robot_frame=in_robot_frame)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        fl = ksim.get_site_pose(state.physics_state.data, self.foot_left_idx)[0] + jnp.array(
            [0.0, 0.0, self.floor_threshold]
        )
        fr = ksim.get_site_pose(state.physics_state.data, self.foot_right_idx)[0] + jnp.array(
            [0.0, 0.0, self.floor_threshold]
        )
        
        if self.in_robot_frame:
            # Transform foot positions to robot frame
            base_quat = state.physics_state.data.qpos[3:7]  # Base quaternion
            fl = xax.rotate_vector_by_quat(fl, base_quat, inverse=True)
            fr = xax.rotate_vector_by_quat(fr, base_quat, inverse=True)
        
        return jnp.concatenate([fl, fr], axis=-1)


@attrs.define(frozen=True)
class BaseHeightObservation(ksim.Observation):
    """Observation of the base height."""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.xpos[1, 2:]


@attrs.define(kw_only=True)
class AngularVelocityCommandMarker(ksim.vis.Marker):
    """Visualise the 1-D yaw-rate command."""

    command_name: str = attrs.field()
    radius: float = attrs.field(default=0.05)
    size: float = attrs.field(default=0.03)
    arrow_len: float = attrs.field(default=0.25)
    height: float = attrs.field(default=0.75)

    def update(self, trajectory: ksim.Trajectory) -> None:
        w = float(trajectory.command[self.command_name][0])
        self.pos = (0.0, 0.0, self.height)

        if abs(w) < 1e-4:  # zero → grey sphere
            self.geom = mujoco.mjtGeom.mjGEOM_SPHERE
            self.scale = (self.radius, self.radius, self.radius)
            self.rgba = (0.8, 0.8, 0.8, 0.8)
            return

        self.geom = mujoco.mjtGeom.mjGEOM_ARROW
        self.scale = (self.size, self.size, self.arrow_len)
        direction = (0.0, 0.0, 1.0 if w > 0 else -1.0)
        self.orientation = self.quat_from_direction(direction)
        self.rgba = (0.2, 0.2, 1.0, 0.8) if w > 0 else (1.0, 0.5, 0.0, 0.8)

    @classmethod
    def get(cls, command_name: str, *, height: float = 0.75) -> Self:
        return cls(
            command_name=command_name,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,
            scale=(0.0, 0.0, 0.0),
            height=height,
        )


@attrs.define(kw_only=True)
class LinearVelocityCommandMarker(ksim.vis.Marker):
    """Visualise the planar (x,y) velocity command."""

    command_name: str = attrs.field()
    size: float = attrs.field(default=0.03)
    arrow_scale: float = attrs.field(default=1.0)
    height: float = attrs.field(default=0.5)

    def update(self, trajectory: ksim.Trajectory) -> None:
        vx, vy = map(float, trajectory.command[self.command_name])
        speed = (vx * vx + vy * vy) ** 0.5
        self.pos = (0.0, 0.0, self.height)

        if speed < 1e-4:  # zero → grey sphere
            self.geom = mujoco.mjtGeom.mjGEOM_SPHERE
            self.scale = (self.size, self.size, self.size)
            self.rgba = (0.8, 0.8, 0.8, 0.8)
            return

        self.geom = mujoco.mjtGeom.mjGEOM_ARROW
        self.scale = (self.size, self.size, self.arrow_scale * speed)
        self.orientation = self.quat_from_direction((vx, vy, 0.0))
        self.rgba = (0.2, 0.8, 0.2, 0.8)

    @classmethod
    def get(
        cls,
        command_name: str,
        *,
        arrow_scale: float = 0.25,
        height: float = 0.5,
    ) -> Self:
        return cls(
            command_name=command_name,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,
            scale=(0.0, 0.0, 0.0),
            arrow_scale=arrow_scale,
            height=height,
        )


@attrs.define(frozen=True)
class AngularVelocityCommand(ksim.Command):
    """Command to turn the robot."""

    scale: float = attrs.field()
    ctrl_dt: float = attrs.field()
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    min_magnitude: float = attrs.field(default=0.01)

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        # sample ang vel cmd
        rng_a, rng_b = jax.random.split(rng)
        yaw_vel_cmd = jax.random.uniform(rng_b, (1,), minval=-self.scale, maxval=self.scale)
        zero_mask = jnp.logical_or(jax.random.bernoulli(rng_a, self.zero_prob), jnp.abs(yaw_vel_cmd) < self.min_magnitude)
        yaw_vel_cmd = jnp.where(zero_mask, jnp.zeros_like(yaw_vel_cmd), yaw_vel_cmd)

        # init: read og base quat, get rz, thats init heading.
        # then spin back base quat by euler [0 0 -init_rz], to get it to face 1 0 0 0.
        # then every timestep, add rz_cmd*dt to heading, and spin back the base quat. 
        # spun back base quat is obs. goal for model is to keep it at 1 0 0 0.
        # can also use this method for roll and pitch cmd.

        # get init heading rz
        init_quat = physics_data.xquat[1]
        init_euler = xax.quat_to_euler(init_quat)
        init_rz = init_euler[..., 2] + self.ctrl_dt * yaw_vel_cmd # add 1 step of yaw vel cmd to init rz.
        init_rz_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, init_rz[0]]))

        # get heading obs, spin back by init_rz, to get it to face 1 0 0 0.
        # TODO temp heading obs, no noise for now. need solution.
        heading_obs = physics_data.xquat[..., 1, :]
        
        # Rotate heading_obs by inverse of init_rz_quat to spin it back
        heading_obs = rotate_quat_by_quat(heading_obs, init_rz_quat, inverse=True)
        
        # return yaw velocity cmd, heading_obs, for heading carry: rz # TODO temp HACK, dont want rz in obs but need for carry. 
        # Combine into single vector [1], [4], [1] -- > [6]
        return jnp.concatenate([yaw_vel_cmd, heading_obs, init_rz], axis=0)


    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:       
        # Extract previous values
        prev_yaw_vel = prev_command[0]
        yaw_cmd = prev_command[5]  # accumulated yaw, passed by carry for now
        
        # Update accumulated yaw by integrating angular velocity
        yaw_cmd = yaw_cmd + prev_yaw_vel * self.ctrl_dt
        
        # Get current heading observation and rotate it back by yaw_cmd
        heading_obs = physics_data.xquat[1]  # Base quaternion
        yaw_cmd_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, yaw_cmd]))
        heading_obs = rotate_quat_by_quat(heading_obs, yaw_cmd_quat, inverse=True)

        updated_command = jnp.concatenate([jnp.array([prev_yaw_vel]), heading_obs, jnp.array([yaw_cmd])], axis=0)
        
        # sample new commands
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        cmd = jnp.where(switch_mask, new_commands, updated_command)
        return cmd

    def get_markers(self) -> Collection[ksim.vis.Marker]:
        return [AngularVelocityCommandMarker.get(self.command_name)]


@attrs.define(frozen=True)
class LinearVelocityCommand(ksim.Command):
    """Command to move the robot in a straight line.

    By convention, X is forward and Y is left. The switching probability is the
    probability of resampling the command at each step. The zero probability is
    the probability of the command being zero - this can be used to turn off
    any command.
    """

    x_range: tuple[float, float] = attrs.field()
    y_range: tuple[float, float] = attrs.field()
    x_zero_prob: float = attrs.field(default=0.0)
    y_zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=1.0)
    vis_scale: float = attrs.field(default=0.05)
    min_magnitude: float = attrs.field(default=0.01)

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_x, rng_y, rng_zero_x, rng_zero_y = jax.random.split(rng, 4)
        (xmin, xmax), (ymin, ymax) = self.x_range, self.y_range
        x = jax.random.uniform(rng_x, (1,), minval=xmin, maxval=xmax)
        y = jax.random.uniform(rng_y, (1,), minval=ymin, maxval=ymax)
        x_zero_mask = jnp.logical_or(jax.random.bernoulli(rng_zero_x, self.x_zero_prob), jnp.abs(x) < self.min_magnitude)  # don't like small commands
        y_zero_mask = jnp.logical_or(jax.random.bernoulli(rng_zero_y, self.y_zero_prob), jnp.abs(y) < self.min_magnitude)
        return jnp.concatenate(
            [
                jnp.where(x_zero_mask, 0.0, x),
                jnp.where(y_zero_mask, 0.0, y),
            ]
        )

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def get_markers(self) -> Collection[ksim.vis.Marker]:
        return [LinearVelocityCommandMarker.get(self.command_name)]


@attrs.define(frozen=True)
class BaseHeightCommand(ksim.Command):
    """Command to set the base height.
    
    Samples heights uniformly between min_height and max_height with a configurable
    switch probability. Otherwise keeps the previous height.
    """

    min_height: float = attrs.field()
    max_height: float = attrs.field()
    switch_prob: float = attrs.field(default=0.0)

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        height = jax.random.uniform(rng_b, (1,), minval=self.min_height, maxval=self.max_height)
        return height

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_height = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_height, prev_command)


@attrs.define(frozen=True)
class XYOrientationCommand(ksim.Command):
    """Command to set base xy orientation. """

    range: float = attrs.field()
    zero_prob: float = attrs.field(default=0.85)
    switch_prob: float = attrs.field(default=0.0)

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_rx, rng_ry, rng_zero = jax.random.split(rng, 3)
        rx = jax.random.uniform(rng_rx, (1,), minval=-self.range, maxval=self.range)
        ry = jax.random.uniform(rng_ry, (1,), minval=-self.range, maxval=self.range)

        # Apply zero probability to rx and ry independently
        rng_zero_rx, rng_zero_ry = jax.random.split(rng_zero)
        zero_rx = jax.random.bernoulli(rng_zero_rx, self.zero_prob)
        zero_ry = jax.random.bernoulli(rng_zero_ry, self.zero_prob)
        rx = jnp.where(zero_rx, 0.0, rx)
        ry = jnp.where(zero_ry, 0.0, ry)

        return jnp.concatenate([rx, ry])
    
    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_command = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_command, prev_command)


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = self.num_outputs * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(self.num_outputs, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(self.num_outputs, self.num_mixtures)
        logits_nm = out_n[..., slice_len * 2 :].reshape(self.num_outputs, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means.
        mean_nm = mean_nm + jnp.array([v for _, v, _ in ZEROS])[:, None]

        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

        self.num_inputs = num_inputs

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_actor_inputs: int,
        num_actor_outputs: int,
        num_critic_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = Actor(
            actor_key,
            num_inputs=num_actor_inputs,
            num_outputs=num_actor_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
        )


class HumanoidWalkingTask(ksim.PPOTask[HumanoidWalkingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        return (
            optax.adam(self.config.learning_rate)
            if self.config.adam_weight_decay == 0.0
            else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot", name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot"))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        if metadata.actuator_type_to_metadata is None:
            raise ValueError("Actuator metadata is not available")
        return metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.95, scale_upper=1.05),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-2), scale_upper=math.radians(2)),
            ksim.FloorFrictionRandomizer.from_geom_name(
                model=physics_model, floor_geom_name="floor", scale_lower=0.1, scale_upper=2.0
            ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            # ksim.PushEvent(
            #     x_force=1.0,
            #     y_force=1.0,
            #     z_force=0.3,
            #     force_range=(0.2, 0.8),
            #     x_angular_force=0.4,
            #     y_angular_force=0.4,
            #     z_angular_force=0.4,
            #     interval_range=(4.0, 6.0),
            # ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v, _ in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(noise=math.radians(2)),
            ksim.JointVelocityObservation(noise=math.radians(10)),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                noise=math.radians(1),
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=0.5,
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(10),
            ),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_force", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_force", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="base_site_linvel", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="base_site_angvel", noise=0.0),
            FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names=(
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_0",
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_1",
                ),
                foot_right_geom_names=(
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_0",
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_1",
                ),
                floor_geom_names="floor",
            ),
            FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_site_name="left_foot",
                foot_right_site_name="right_foot",
                floor_threshold=0.0,
                in_robot_frame=True,
            ),
            BaseHeightObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            # LinearVelocityCommand(
            #     x_range=(-0.3, 0.8),
            #     y_range=(-0.3, 0.3),
            #     x_zero_prob=0.5,
            #     y_zero_prob=0.5,
            #     switch_prob=self.config.ctrl_dt / 3,  # once per 3 seconds
            #     min_magnitude=self.config.stand_still_threshold,
            # ),
            LinearVelocityCommand(
                x_range=(-0.3, 0.8),
                y_range=(-0.3, 0.3),
                x_zero_prob=1.0,
                y_zero_prob=1.0,
                switch_prob=self.config.ctrl_dt / 3,  # once per 3 seconds
                min_magnitude=self.config.stand_still_threshold,
            ),
            # AngularVelocityCommand(
            #     scale=0.5,
            #     zero_prob=0.9,
            #     switch_prob=self.config.ctrl_dt / 3,  # once per 3 seconds
            #     min_magnitude=self.config.stand_still_threshold,
            #     ctrl_dt=self.config.ctrl_dt,
            # ),
            AngularVelocityCommand(
                scale=0.5,
                zero_prob=1.0,
                switch_prob=self.config.ctrl_dt / 3,  # once per 3 seconds
                min_magnitude=self.config.stand_still_threshold,
                ctrl_dt=self.config.ctrl_dt,
            ),
            BaseHeightCommand(
                min_height=0.7,
                max_height=1.0,
                switch_prob=self.config.ctrl_dt / 3, 
            ),
            XYOrientationCommand(
                range=0.05*math.pi,
                zero_prob=0.85,
                switch_prob=self.config.ctrl_dt / 3,  # once per 3 seconds
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            # cmd
            LinearVelocityTrackingReward(scale=1.0, error_scale=0.1),
            AngularVelocityTrackingReward(scale=1.0, error_scale=0.05),
            XYOrientationReward(scale=1.0, error_scale=0.025),
            BaseHeightReward(scale=1.0, error_scale=0.05),
            # shaping
            # SingleFootContactReward(scale=0.3),
            # FeetAirtimeReward(scale=3.0, ctrl_dt=self.config.ctrl_dt, touchdown_penalty=0.35),
            FeetPositionReward(scale=0.2, error_scale=0.05, stance_width=0.3),
            # sim2real
            ksim.ActionAccelerationPenalty(scale=-0.01, scale_by_curriculum=False),
            BentArmPenalty.create_penalty(physics_model, scale=-0.1), # TODO best to have this in joint space vs pos space. looks like it is.

        
            # ksim.AvoidLimitsPenalty.create(physics_model, scale=-0.01, scale_by_curriculum=True),
            # ksim.JointAccelerationPenalty(scale=-0.01, scale_by_curriculum=False),
            # ksim.JointJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.LinkAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.LinkJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            # StraightLegPenalty.create_penalty(physics_model, scale=-0.2),
            # FeetSlipPenalty(scale=-0.25),
            # ContactForcePenalty( # NOTE this could actually be good but eliminate until needed
            #     scale=-0.03,
            #     sensor_names=("sensor_observation_left_foot_force", "sensor_observation_right_foot_force"),
            # ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.6, unhealthy_z_upper=1.2),
            ksim.NotUprightTermination(max_radians=math.radians(60)),
            ksim.EpisodeLengthTermination(max_length=24),
            # ksim.FarFromOriginTermination(max_dist15.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.EpisodeLengthCurriculum(
            num_levels=self.config.num_curriculum_levels,
            increase_threshold=self.config.increase_threshold,
            decrease_threshold=self.config.decrease_threshold,
            min_level_steps=self.config.min_level_steps,
            min_level=0.1,
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        num_joints = len(ZEROS)

        #  joint pos / vel + proj_grav
        num_actor_obs = num_joints * 2 + 3

        if self.config.use_acc_gyro:
            num_actor_obs += 6

        num_commands = (
            2  # linear velocity command (x, y)
            + 6  # angular velocity command (cmd, heading_obs[4], carry[1])
            + 1  # base height command
            + 2  # base xy orientation command
        )
        num_actor_inputs = num_actor_obs + num_commands

        num_critic_inputs = (
            num_actor_inputs
            + 4  # feet contact
            + 6  # feet position
            + 3
            + 4  # base pos / quat
            + 138
            + 230  # COM inertia / velocity
            + 3
            + 3  # base linear / angular vel
            + num_joints  # actuator force
            + 3
            + 3  # imu_acc/gyro (privileged copies)
            + 1  # base height
        )

        if self.config.use_acc_gyro:
            num_critic_inputs -= 6

        return Model(
            key,
            num_actor_inputs=num_actor_inputs,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=num_critic_inputs,
            min_std=0.01,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd = commands["angular_velocity_command"]
        base_height_cmd = commands["base_height_command"]
        base_xy_orient_cmd = commands["xyorientation_command"]

        obs = [
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n,  # NUM_JOINTS
            proj_grav_3,  # 3
            lin_vel_cmd_2,  # 2
            # ang_vel_cmd,  # 6

            # TODO HACK heading
            ang_vel_cmd[..., :-1],
            jnp.zeros_like(ang_vel_cmd[..., -1:]),

            (base_height_cmd-0.85),  # 1
            base_xy_orient_cmd,  # 2
        ]
        if self.config.use_acc_gyro:
            obs += [
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
            ]

        obs_n = jnp.concatenate(obs, axis=-1)
        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd = commands["angular_velocity_command"]
        base_height_cmd = commands["base_height_command"]
        base_xy_orient_cmd = commands["xyorientation_command"]

        # privileged obs
        feet_contact_4 = observations["feet_contact_observation"]
        feet_position_6 = observations["feet_position_observation"]
        base_position_3 = observations["base_position_observation"]
        base_orientation_4 = observations["base_orientation_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        base_lin_vel_3 = observations["base_linear_velocity_observation"]
        base_ang_vel_3 = observations["base_angular_velocity_observation"]
        actuator_force_n = observations["actuator_force_observation"]
        base_height = observations["base_height_observation"]

        obs_n = jnp.concatenate(
            [   
                # actor obs:
                joint_pos_n,
                joint_vel_n / 10.0, # TODO fix this
                proj_grav_3,
                lin_vel_cmd_2,
                # ang_vel_cmd,

                # TODO HACK heading
                ang_vel_cmd[..., :-1],
                jnp.zeros_like(ang_vel_cmd[..., -1:]),

                (base_height_cmd-0.85),
                base_xy_orient_cmd,
                imu_acc_3, # m/s^2
                imu_gyro_3, # rad/s
                # privileged obs:
                com_inertia_n,
                com_vel_n,
                actuator_force_n / 4.0,
                base_position_3,
                base_orientation_4,
                base_lin_vel_3,
                base_ang_vel_3,
                feet_contact_4,
                feet_position_6,
                base_height,
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def _ppo_scan_fn(
        self,
        actor_critic_carry: tuple[Array, Array],
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
        transition, rng = xs

        actor_carry, critic_carry = actor_critic_carry
        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_carry,
        )

        # Gets the log probabilities of the action.
        log_probs = actor_dist.log_prob(transition.action)
        assert isinstance(log_probs, Array)

        value, next_critic_carry = self.run_critic(
            model=model.critic,
            observations=transition.obs,
            commands=transition.command,
            carry=critic_carry,
        )

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=value.squeeze(-1),
        )

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(rng),
            (next_actor_carry, next_critic_carry),
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        scan_fn = functools.partial(self._ppo_scan_fn, model=model)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=4,
        )
        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=2.0, # temporarily putting this lower to go faster
            global_grad_clip=2.0,
            entropy_coef=0.004,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.003, 0.01),  # Simulate 3-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            render_track_body_id=0,
            render_markers=True,
            render_full_every_n_seconds=0,
            render_length_seconds=10,
            max_values_per_plot=50,
            # Checkpointing parameters.
            save_every_n_seconds=60,
        ),
    )