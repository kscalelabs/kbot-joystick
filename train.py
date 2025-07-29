"""Defines simple task for training a joystick walking policy for K-Bot."""

import asyncio
import functools
import math
from dataclasses import dataclass
from typing import Self

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
from jaxtyping import Array, PRNGKeyArray, PyTree

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
    ("dof_right_hip_pitch_04", math.radians(-20.0), 1.0),
    ("dof_right_hip_roll_03", math.radians(-0.0), 1.0),
    ("dof_right_hip_yaw_03", 0.0, 1.0),
    ("dof_right_knee_04", math.radians(-50.0), 1.0),
    ("dof_right_ankle_02", math.radians(30.0), 1.0),
    ("dof_left_hip_pitch_04", math.radians(20.0), 1.0),
    ("dof_left_hip_roll_03", math.radians(0.0), 1.0),
    ("dof_left_hip_yaw_03", 0.0, 1.0),
    ("dof_left_knee_04", math.radians(50.0), 1.0),
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
        value=2,
        help="The depth for the RNN",
    )
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )
    use_gyro: bool = xax.field(
        value=True,
        help="Whether to use the IMU gyroscope observations.",
    )
    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=5e-4,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )
    use_lr_decay: bool = xax.field(
        value=False,
        help="Whether to use cosine learning rate decay",
    )
    lr_decay_steps: int = xax.field(
        value=19_200_000,  # 19.2mm is about 5k iters with current batching
        help="Number of steps for cosine decay schedule",
    )
    lr_final_multiplier: float = xax.field(
        value=0.01,
        help="Final learning rate will be this * initial learning rate",
    )
    mirror_loss_scale: float = xax.field(
        value=0.01,
        help="Scale for the mirror loss",
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
    w1, x1, y1, z1 = jnp.split(rotating_quat, 4, axis=-1)  # rotating quaternion
    w2, x2, y2, z2 = jnp.split(quat_to_rotate, 4, axis=-1)  # quaternion being rotated

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
        cost = jnp.clip(jnp.abs(forces[:, 2, :]) - self.max_contact_force, 0)
        return jnp.sum(cost, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class SingleFootContactReward(ksim.StatefulReward):
    """Reward having one and only one foot in contact with the ground, while walking.

    Allows for small grace period when both feet are in contact for less jumpy gaits.
    """

    scale: float = 1.0
    ctrl_dt: float = 0.02
    grace_period: float = 0.2  # seconds

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        return jnp.array([0.0])

    def get_reward_stateful(self, traj: ksim.Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        left_contact = jnp.where(traj.obs["sensor_observation_left_foot_touch"] > 0.1, True, False)[:, 0]
        right_contact = jnp.where(traj.obs["sensor_observation_right_foot_touch"] > 0.1, True, False)[:, 0]
        single = jnp.logical_xor(left_contact, right_contact)
        is_zero_cmd = jnp.linalg.norm(traj.command["unified_command"][:, :3], axis=-1) < 1e-3

        def _body(time_since_single_contact: Array, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
            is_single_contact, is_zero = inputs
            new_time = jnp.where(is_single_contact, 0.0, time_since_single_contact + self.ctrl_dt)
            # if zero cmd, then max out time to reset grace period.
            new_time = jnp.where(is_zero, self.grace_period, new_time)
            return new_time, new_time

        carry, time_since_single_contact = jax.lax.scan(_body, reward_carry, (single, is_zero_cmd))
        single_contact_grace = time_since_single_contact < self.grace_period
        reward = jnp.where(is_zero_cmd, 1.0, single_contact_grace[:, 0])
        return reward, carry


@attrs.define(frozen=True, kw_only=True)
class FeetAirtimeReward(ksim.StatefulReward):
    """Encourages reasonable step frequency by rewarding long swing phases and penalizing quick stepping."""

    scale: float = 1.0
    ctrl_dt: float = 0.02
    touchdown_penalty: float = 0.4
    scale_by_curriculum: bool = False

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        # initial left and right airtime
        return jnp.array([0.0, 0.0])

    def _airtime_sequence(self, initial_airtime: Array, contact_bool: Array, done: Array) -> tuple[Array, Array]:
        """Returns an array with the airtime (in seconds) for each timestep."""

        def _body(time_since_liftoff: Array, is_contact: Array) -> tuple[Array, Array]:
            new_time = jnp.where(is_contact, 0.0, time_since_liftoff + self.ctrl_dt)
            return new_time, new_time

        # or with done to reset the airtime counter when the episode is done
        contact_or_done = jnp.logical_or(contact_bool, done)
        carry, airtime = jax.lax.scan(_body, initial_airtime, contact_or_done)
        return carry, airtime

    def get_reward_stateful(self, traj: ksim.Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        left_contact = jnp.where(traj.obs["sensor_observation_left_foot_touch"] > 0.1, True, False)[:, 0]
        right_contact = jnp.where(traj.obs["sensor_observation_right_foot_touch"] > 0.1, True, False)[:, 0]

        # airtime counters
        left_carry, left_air = self._airtime_sequence(reward_carry[0], left_contact, traj.done)
        right_carry, right_air = self._airtime_sequence(reward_carry[1], right_contact, traj.done)

        reward_carry = jnp.array([left_carry, right_carry])

        # touchdown boolean (0→1 transition)
        def touchdown(c: Array) -> Array:
            prev = jnp.concatenate([jnp.array([False]), c[:-1]])
            return jnp.logical_and(c, jnp.logical_not(prev))

        td_l = touchdown(left_contact)
        td_r = touchdown(right_contact)

        left_air_shifted = jnp.roll(left_air, 1)
        right_air_shifted = jnp.roll(right_air, 1)

        left_feet_airtime_reward = (left_air_shifted - self.touchdown_penalty) * td_l.astype(jnp.float32)
        right_feet_airtime_reward = (right_air_shifted - self.touchdown_penalty) * td_r.astype(jnp.float32)

        reward = jnp.minimum(left_feet_airtime_reward + right_feet_airtime_reward, 0.0)

        # standing mask
        is_zero_cmd = jnp.linalg.norm(traj.command["unified_command"][:, :3], axis=-1) < 1e-3
        reward = jnp.where(is_zero_cmd, 0.0, reward)

        return reward, reward_carry


@attrs.define(frozen=True, kw_only=True)
class ArmPositionReward(ksim.Reward):
    """Reward for tracking commanded arm joint positions.

    Compares the current arm joint positions against commanded positions from
    trajectory.command["unified_command"][7:].
    """

    joint_indices: Array = attrs.field(eq=False)
    joint_biases: Array = attrs.field(eq=False)
    error_scale: float = attrs.field(default=0.1)
    norm: xax.NormType = attrs.field(default="l2")

    @classmethod
    def create_reward(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = 0.05,
        error_scale: float = 0.1,
        scale_by_curriculum: bool = False,
    ) -> Self:
        # Define the arm joint names in order
        joint_names = (
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
        )

        # Map joint names to indices
        joint_to_idx = ksim.get_qpos_data_idxs_by_name(physics_model)
        joint_indices = jnp.array([int(joint_to_idx[name][0]) - 7 for name in joint_names])
        joint_biases = jnp.array([bias for (name, bias, _) in ZEROS if name in joint_names])

        return cls(
            joint_indices=joint_indices,
            joint_biases=joint_biases,
            error_scale=error_scale,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        qpos_sel = trajectory.qpos[..., jnp.array(self.joint_indices) + 7]
        target = trajectory.command["unified_command"][..., 7:17] + self.joint_biases
        error = xax.get_norm(qpos_sel - target, self.norm).sum(axis=-1)
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="unified_command")
    norm: xax.NormType = attrs.field(default="l2")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # Get global frame velocities
        global_vel = trajectory.qvel[:, :3]

        # get base quat, only yaw.
        # careful to only rotate in z, disregard rx and ry, bad conflict with roll and pitch.
        base_euler = xax.quat_to_euler(trajectory.xquat[:, 1, :])
        base_euler = base_euler.at[:, :2].set(0.0)
        base_z_quat = xax.euler_to_quat(base_euler)

        # rotate local frame commands to global frame
        robot_vel_cmd = jnp.zeros_like(global_vel).at[:, :2].set(trajectory.command[self.command_name][:, :2])
        global_vel_cmd = xax.rotate_vector_by_quat(robot_vel_cmd, base_z_quat, inverse=False)

        # drop vz. vz conflicts with base height reward.
        global_vel_xy_cmd = global_vel_cmd[:, :2]
        global_vel_xy = global_vel[:, :2]

        # now compute error. special trick: different kernels for standing and walking.
        zero_cmd_mask = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        vel_error = jnp.linalg.norm(global_vel_xy - global_vel_xy_cmd, axis=-1)
        error = jnp.where(zero_cmd_mask, vel_error, jnp.square(vel_error))
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the heading using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="unified_command")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        base_yaw = xax.quat_to_euler(trajectory.xquat[:, 1, :])[:, 2]
        base_yaw_cmd = trajectory.command[self.command_name][:, 3]

        base_yaw_quat = xax.euler_to_quat(
            jnp.stack([jnp.zeros_like(base_yaw_cmd), jnp.zeros_like(base_yaw_cmd), base_yaw], axis=-1)
        )
        base_yaw_target_quat = xax.euler_to_quat(
            jnp.stack([jnp.zeros_like(base_yaw_cmd), jnp.zeros_like(base_yaw_cmd), base_yaw_cmd], axis=-1)
        )

        # Compute quaternion error
        quat_error = 1 - jnp.sum(base_yaw_target_quat * base_yaw_quat, axis=-1) ** 2
        return jnp.exp(-quat_error / self.error_scale)


@attrs.define(frozen=True)
class XYOrientationReward(ksim.Reward):
    """Reward for tracking the xy base orientation using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="unified_command")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        euler_orientation = xax.quat_to_euler(trajectory.xquat[:, 1, :])
        euler_orientation = euler_orientation.at[:, 2].set(0.0)  # ignore yaw
        base_xy_quat = xax.euler_to_quat(euler_orientation)

        commanded_euler = jnp.stack(
            [
                trajectory.command[self.command_name][:, 5],
                trajectory.command[self.command_name][:, 6],
                jnp.zeros_like(trajectory.command[self.command_name][:, 6]),
            ],
            axis=-1,
        )
        base_xy_quat_cmd = xax.euler_to_quat(commanded_euler)

        quat_error = 1 - jnp.sum(base_xy_quat_cmd * base_xy_quat, axis=-1) ** 2
        return jnp.exp(-quat_error / self.error_scale)


@attrs.define(frozen=True)
class TerrainBaseHeightReward(ksim.Reward):
    """Reward for keeping a set distance between the base and the lowest foot.

    Compatible with hfield scenes, where floor height is variable.
    """

    base_idx: int = attrs.field()
    foot_left_idx: int = attrs.field()
    foot_right_idx: int = attrs.field()
    error_scale: float = attrs.field(default=0.25)
    standard_height: float = attrs.field(default=0.9)
    foot_origin_height: float = attrs.field(default=0.0)

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        base_body_name: str,
        foot_left_body_name: str,
        foot_right_body_name: str,
        scale: float,
        error_scale: float,
        standard_height: float,
        foot_origin_height: float,
    ) -> Self:
        base = ksim.get_body_data_idx_from_name(physics_model, base_body_name)
        fl = ksim.get_body_data_idx_from_name(physics_model, foot_left_body_name)
        fr = ksim.get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(
            base_idx=base,
            foot_left_idx=fl,
            foot_right_idx=fr,
            scale=scale,
            error_scale=error_scale,
            standard_height=standard_height,
            foot_origin_height=foot_origin_height,
        )

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        left_foot_z = trajectory.xpos[:, self.foot_left_idx, 2] - self.foot_origin_height
        right_foot_z = trajectory.xpos[:, self.foot_right_idx, 2] - self.foot_origin_height
        lowest_foot_z = jnp.minimum(left_foot_z, right_foot_z)

        base_z = trajectory.xpos[:, self.base_idx, 2]

        current_height = base_z - lowest_foot_z
        commanded_height = trajectory.command["unified_command"][:, 4] + self.standard_height
        height_diff = current_height - commanded_height

        # for walking: only care about minimum height.
        is_zero_cmd = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        height_error = jnp.where(is_zero_cmd, jnp.abs(height_diff), jnp.abs(jnp.minimum(height_diff, 0.0)))
        return jnp.exp(-height_error / self.error_scale)


@attrs.define(frozen=True)
class StandingFeetPositionReward(ksim.Reward):
    """Reward for keeping feet at a set distance from each other when standing still."""

    error_scale: float = attrs.field(default=0.25)
    stance_width: float = attrs.field(default=0.3)
    base_idx: int = attrs.field(default=1)
    foot_left_idx: int = attrs.field(default=0)
    foot_right_idx: int = attrs.field(default=0)

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        base_body_name: str,
        foot_left_body_name: str,
        foot_right_body_name: str,
        scale: float,
        error_scale: float,
        stance_width: float,
    ) -> Self:
        base = ksim.get_body_data_idx_from_name(physics_model, base_body_name)
        fl = ksim.get_body_data_idx_from_name(physics_model, foot_left_body_name)
        fr = ksim.get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(
            base_idx=base,
            foot_left_idx=fl,
            foot_right_idx=fr,
            scale=scale,
            error_scale=error_scale,
            stance_width=stance_width,
        )

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # get global positions
        global_l_foot_pos = trajectory.xpos[:, self.foot_left_idx]
        global_r_foot_pos = trajectory.xpos[:, self.foot_right_idx]
        base_pos = trajectory.xpos[:, self.base_idx]
        base_quat = trajectory.xquat[:, self.base_idx, :]

        # compute feet pos in base frame
        l_foot_pos = xax.rotate_vector_by_quat((global_l_foot_pos - base_pos), base_quat, inverse=True)
        r_foot_pos = xax.rotate_vector_by_quat((global_r_foot_pos - base_pos), base_quat, inverse=True)

        # calculate stance errors
        stance_x_error = jnp.abs(l_foot_pos[:, 0] - r_foot_pos[:, 0])
        stance_y_error = jnp.abs(jnp.abs(l_foot_pos[:, 1] - r_foot_pos[:, 1]) - self.stance_width)
        stance_error = stance_x_error + stance_y_error

        # only apply reward for standing
        zero_cmd_mask = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        error = jnp.where(zero_cmd_mask, stance_error, 0.0)
        reward = jnp.exp(-error / self.error_scale)
        return reward


@attrs.define(frozen=True)
class FeetOrientationReward(ksim.Reward):
    """Reward for keeping feet pitch and roll oriented parallel to the ground."""

    scale: float = attrs.field(default=1.0)
    error_scale: float = attrs.field(default=0.25)
    foot_left_idx: int = attrs.field(default=0)
    foot_right_idx: int = attrs.field(default=0)

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        foot_left_body_name: str,
        foot_right_body_name: str,
        scale: float,
        error_scale: float,
    ) -> Self:
        fl = ksim.get_body_data_idx_from_name(physics_model, foot_left_body_name)
        fr = ksim.get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(foot_left_idx=fl, foot_right_idx=fr, scale=scale, error_scale=error_scale)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # compute error for standing
        straight_foot_euler = jnp.stack(
            [
                jnp.full_like(trajectory.command["unified_command"][:, 3], -jnp.pi / 2),
                jnp.zeros_like(trajectory.command["unified_command"][:, 3]),
                trajectory.command["unified_command"][:, 3] - jnp.pi,  # include yaw
            ],
            axis=-1,
        )
        straight_foot_quat = xax.euler_to_quat(straight_foot_euler)

        left_quat_error = 1 - jnp.sum(straight_foot_quat * trajectory.xquat[:, self.foot_left_idx, :], axis=-1) ** 2
        right_quat_error = 1 - jnp.sum(straight_foot_quat * trajectory.xquat[:, self.foot_right_idx, :], axis=-1) ** 2
        standing_error = left_quat_error + right_quat_error

        # compute error for walking
        left_foot_euler = xax.quat_to_euler(trajectory.xquat[:, self.foot_left_idx, :])
        right_foot_euler = xax.quat_to_euler(trajectory.xquat[:, self.foot_right_idx, :])

        # for walking, mask out yaw
        left_foot_quat = xax.euler_to_quat(left_foot_euler.at[:, 2].set(0.0))
        right_foot_quat = xax.euler_to_quat(right_foot_euler.at[:, 2].set(0.0))

        straight_foot_euler = jnp.stack([-jnp.pi / 2, 0, 0], axis=-1)
        straight_foot_quat = xax.euler_to_quat(straight_foot_euler)

        left_quat_error = 1 - jnp.sum(straight_foot_quat * left_foot_quat, axis=-1) ** 2
        right_quat_error = 1 - jnp.sum(straight_foot_quat * right_foot_quat, axis=-1) ** 2
        walking_error = left_quat_error + right_quat_error

        # choose standing or walking error based on command
        is_zero_cmd = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        total_error = jnp.where(is_zero_cmd, standing_error, walking_error)

        return jnp.exp(-total_error / self.error_scale)


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    base_idx: int
    foot_left_idx: int
    foot_right_idx: int

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        base_body_name: str,
        foot_left_body_name: str,
        foot_right_body_name: str,
    ) -> Self:
        base = ksim.get_body_data_idx_from_name(physics_model, base_body_name)
        fl = ksim.get_body_data_idx_from_name(physics_model, foot_left_body_name)
        fr = ksim.get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(base_idx=base, foot_left_idx=fl, foot_right_idx=fr)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        # get global positions
        base_pos = state.physics_state.data.xpos[self.base_idx]
        left_foot_pos = state.physics_state.data.xpos[self.foot_left_idx]
        right_foot_pos = state.physics_state.data.xpos[self.foot_right_idx]

        base_yaw = xax.quat_to_euler(state.physics_state.data.xquat[self.base_idx, :])[2]
        base_yaw_quat = xax.euler_to_quat(
            jnp.stack([jnp.zeros_like(base_yaw), jnp.zeros_like(base_yaw), base_yaw], axis=-1)
        )

        # transform feet pos to base frame
        relative_left_foot_pos = left_foot_pos - base_pos
        relative_right_foot_pos = right_foot_pos - base_pos
        fl_ndarray = xax.rotate_vector_by_quat(relative_left_foot_pos, base_yaw_quat, inverse=True)
        fr_ndarray = xax.rotate_vector_by_quat(relative_right_foot_pos, base_yaw_quat, inverse=True)

        return jnp.concatenate([fl_ndarray, fr_ndarray], axis=-1)


@attrs.define(frozen=True)
class BaseHeightObservation(ksim.Observation):  # TODO not terrain compatible
    """Observation of the base height."""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.xpos[1, 2:]


class QposObservation(ksim.Observation):
    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.qpos[7:]


class QvelObservation(ksim.Observation):
    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.qvel[6:]


@attrs.define(frozen=True, kw_only=True)
class ImuOrientationObservation(ksim.StatefulObservation):
    """Observes the IMU orientation, back spun in yaw heading, as commanded.

    This provides an approximation of reading the IMU orientation from
    the IMU on the physical robot, backspun by commanded heading. The `framequat_name` should be the name of
    the framequat sensor attached to the IMU.

    Example: if yaw cmd = 3.14, and IMU reading is [0, 0, 0, 1], then back spun IMU heading obs is [1, 0, 0, 0]

    The policy learns to keep the IMU heading obs around [1, 0, 0, 0].
    """

    framequat_idx_range: tuple[int, int | None] = attrs.field()
    lag_range: tuple[float, float] = attrs.field(
        default=(0.01, 0.1),
        validator=attrs.validators.deep_iterable(
            attrs.validators.and_(
                attrs.validators.ge(0.0),
                attrs.validators.lt(1.0),
            ),
        ),
    )
    bias_euler: tuple[float, float, float] = attrs.field(
        default=(0.0, 0.0, 0.0),
        validator=attrs.validators.deep_iterable(
            attrs.validators.and_(
                attrs.validators.ge(0.0),
                attrs.validators.le(math.pi),
            ),
        ),
    )

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        noise: float = 0.0,
        framequat_name: str,
        lag_range: tuple[float, float] = (0.01, 0.1),
        bias_euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Self:
        """Create an IMU orientation observation from a physics model.

        Args:
            physics_model: MuJoCo physics model
            framequat_name: The name of the framequat sensor
            lag_range: The range of EMA factors to use, to approximate the
                variation in the amount of smoothing of the Kalman filter.
            noise: The observation noise
            bias_euler: The bias in euler angles, in roll, pitch, yaw.
        """
        sensor_name_to_idx_range = ksim.get_sensor_data_idxs_by_name(physics_model)
        if framequat_name not in sensor_name_to_idx_range:
            options = "\n".join(sorted(sensor_name_to_idx_range.keys()))
            raise ValueError(f"{framequat_name} not found in model. Available:\n{options}")

        return cls(
            framequat_idx_range=sensor_name_to_idx_range[framequat_name],
            lag_range=lag_range,
            bias_euler=bias_euler,
            noise=noise,
        )

    def initial_carry(self, physics_state: ksim.PhysicsState, rng: PRNGKeyArray) -> tuple[Array, Array, Array]:
        lrng, brng = jax.random.split(rng, 2)
        minval, maxval = self.lag_range
        lag = jax.random.uniform(lrng, (1,), minval=minval, maxval=maxval)

        bias_range = jnp.array(self.bias_euler)
        bias = jax.random.uniform(brng, (3,), minval=-bias_range, maxval=bias_range)
        bias_quat = xax.euler_to_quat(bias)

        return jnp.zeros((4,)), lag, bias_quat

    def observe_stateful(
        self,
        state: ksim.ObservationInput,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[Array, tuple[Array, Array]]:
        x, lag, bias = state.obs_carry

        framequat_start, framequat_end = self.framequat_idx_range
        framequat_data = state.physics_state.data.sensordata[framequat_start:framequat_end].ravel()

        # apply bias noise
        framequat_data = rotate_quat_by_quat(framequat_data, bias)

        # get heading cmd
        heading_yaw_cmd = state.commands["unified_command"][3]

        # spin back
        heading_yaw_cmd_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, heading_yaw_cmd]))
        backspun_framequat = rotate_quat_by_quat(framequat_data, heading_yaw_cmd_quat, inverse=True)
        # ensure positive quat hemisphere
        backspun_framequat = jnp.where(backspun_framequat[..., 0] < 0, -backspun_framequat, backspun_framequat)

        # Get current Kalman filter state
        x = x * lag + backspun_framequat * (1 - lag)

        return x, (x, lag, bias)


@attrs.define(frozen=True)
class UnifiedCommand(ksim.Command):
    """Unifiying all commands into one to allow for covariance control."""

    vx_range: tuple[float, float] = attrs.field()
    vy_range: tuple[float, float] = attrs.field()
    wz_range: tuple[float, float] = attrs.field()
    bh_range: tuple[float, float] = attrs.field()
    bh_standing_range: tuple[float, float] = attrs.field()
    rx_range: tuple[float, float] = attrs.field()
    ry_range: tuple[float, float] = attrs.field()
    arms_range: tuple[list[float], list[float]] = attrs.field()
    ctrl_dt: float = attrs.field()
    switch_prob: float = attrs.field()

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_a, rng_b, rng_c, rng_d, rng_e, rng_f, rng_g, rng_h, rng_i = jax.random.split(rng, 9)

        # cmd  = [vx, vy, wz, bh, rx, ry]
        vx = jax.random.uniform(rng_b, (1,), minval=self.vx_range[0], maxval=self.vx_range[1])
        vy = jax.random.uniform(rng_c, (1,), minval=self.vy_range[0], maxval=self.vy_range[1])
        wz = jax.random.uniform(rng_d, (1,), minval=self.wz_range[0], maxval=self.wz_range[1])
        bh = jax.random.uniform(rng_e, (1,), minval=self.bh_range[0], maxval=self.bh_range[1])
        bhs = jax.random.uniform(rng_f, (1,), minval=self.bh_standing_range[0], maxval=self.bh_standing_range[1])
        rx = jax.random.uniform(rng_g, (1,), minval=self.rx_range[0], maxval=self.rx_range[1])
        ry = jax.random.uniform(rng_h, (1,), minval=self.ry_range[0], maxval=self.ry_range[1])

        # 50% chance to sample from wide uniform distribution
        arms_uni = jax.random.uniform(
            rng_i, (10,), minval=jnp.array(self.arms_range[0]), maxval=jnp.array(self.arms_range[1])
        )
        # 50% chance to mask out 9/10 arm commands and have only 1 active
        active_idx = jax.random.randint(rng_i, (), minval=0, maxval=10)
        mask = jnp.arange(10) == active_idx
        arms_gau = jax.random.normal(rng_i, (10,)) * 0.5 * mask
        arms_gau = jnp.clip(arms_gau, min=jnp.array(self.arms_range[0]), max=jnp.array(self.arms_range[1]))
        arms = jnp.where(jax.random.bernoulli(rng_i), arms_uni, arms_gau)

        _ = jnp.zeros_like(vx)
        __ = jnp.zeros_like(arms)

        # Create each mode's command vector
        forward_cmd = jnp.concatenate([vx, _, _, bh, _, _, __])
        sideways_cmd = jnp.concatenate([_, vy, _, bh, _, _, __])
        rotate_cmd = jnp.concatenate([_, _, wz, bh, _, _, __])
        omni_cmd = jnp.concatenate([vx, vy, wz, bh, _, _, arms])
        stand_bend_cmd = jnp.concatenate([_, _, _, bhs, rx, ry, arms])
        stand_cmd = jnp.concatenate([_, _, _, _, _, _, __])

        # randomly select a mode
        mode = jax.random.randint(rng_a, (), minval=0, maxval=6)  # 0 1 2 3 4s 5s -- 2/6 standing
        cmd = jax.lax.switch(
            mode,
            [
                lambda: forward_cmd,
                lambda: sideways_cmd,
                lambda: rotate_cmd,
                lambda: omni_cmd,
                lambda: stand_bend_cmd,
                lambda: stand_cmd,
            ],
        )

        # get initial heading
        init_euler = xax.quat_to_euler(physics_data.xquat[1])
        init_heading = init_euler[2] + self.ctrl_dt * cmd[2]  # add 1 step of yaw vel cmd to initial heading.
        cmd = jnp.concatenate([cmd[:3], jnp.array([init_heading]), cmd[3:]])
        assert cmd.shape == (17,)

        return cmd

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        def update_heading(prev_command: Array) -> Array:
            """Update the heading by integrating the angular velocity."""
            wz_cmd, heading = prev_command[2], prev_command[3]
            heading = heading + wz_cmd * self.ctrl_dt
            prev_command = prev_command.at[3].set(heading)
            return prev_command

        continued_command = update_heading(prev_command)

        # def update_arms(prev_command: Array) -> Array:
        #     """Move arm commands by x rad/s."""
        #     arms = prev_command[7:17]
        #     arm_mask = jnp.where(arms != 0.0, 1.0, 0.0)
        #     arms = arms + arm_mask * self.ctrl_dt * 0.5
        #     arms = arms.clip(self.arms_range[:, 0], self.arms_range[:, 1])
        #     prev_command = prev_command.at[7:17].set(arms)
        #     return prev_command

        # continued_command = update_arms(continued_command)

        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_command = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_command, continued_command)


@attrs.define(frozen=True, kw_only=True)
class TerrainBadZTermination(ksim.Termination):
    """Terminates the episode if the robot base is too low. Compatible with terrain."""

    base_idx: int = attrs.field()
    foot_left_idx: int = attrs.field()
    foot_right_idx: int = attrs.field()
    unhealthy_z_lower: float = attrs.field()
    unhealthy_z_upper: float = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        base_body_name: str,
        foot_left_body_name: str,
        foot_right_body_name: str,
        unhealthy_z_lower: float,
        unhealthy_z_upper: float,
    ) -> Self:
        base = ksim.get_body_data_idx_from_name(physics_model, base_body_name)
        fl = ksim.get_body_data_idx_from_name(physics_model, foot_left_body_name)
        fr = ksim.get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(
            base_idx=base,
            foot_left_idx=fl,
            foot_right_idx=fr,
            unhealthy_z_lower=unhealthy_z_lower,
            unhealthy_z_upper=unhealthy_z_upper,
        )

    def __call__(self, state: ksim.PhysicsData, curriculum_level: Array) -> Array:
        base_z = state.xpos[self.base_idx, 2]
        left_foot_z = state.xpos[self.foot_left_idx, 2]
        right_foot_z = state.xpos[self.foot_right_idx, 2]
        lowest_foot_z = jnp.minimum(left_foot_z, right_foot_z)
        height = base_z - lowest_foot_z
        return jnp.where((height < self.unhealthy_z_lower) | (height > self.unhealthy_z_upper), -1, 0)


@attrs.define(frozen=True, kw_only=True)
class PlaneXYPositionReset(ksim.Reset):
    """Resets the robot's XY position"""

    x_range: float = attrs.field(default=1.0)
    y_range: float = attrs.field(default=1.0)

    def __call__(self, data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> ksim.PhysicsData:
        keyx, keyy = jax.random.split(rng)
        new_x = jax.random.uniform(keyx, (1,), minval=-self.x_range, maxval=self.x_range)
        new_y = jax.random.uniform(keyy, (1,), minval=-self.y_range, maxval=self.y_range)

        qpos_j = data.qpos
        if type(qpos_j) != jnp.ndarray:
            return data
        qpos_j = qpos_j.at[0:1].set(new_x)
        qpos_j = qpos_j.at[1:2].set(new_y)
        data = ksim.update_data_field(data, "qpos", qpos_j)
        return data


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
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

        # Project to output - mean and std for each action
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 2,  # mean and std for each output
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
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

        # Split into means and stds
        mean_n = out_n[..., : self.num_outputs]
        std_n = out_n[..., self.num_outputs :]

        # Softplus and clip to ensure positive standard deviations
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means
        arm_cmd_bias = jnp.concatenate([obs_n[..., -13:-3], jnp.zeros(shape=(10,))], axis=-1)
        mean_n = mean_n + jnp.array([v for _, v, _ in ZEROS]) + arm_cmd_bias

        # Create diagonal gaussian distribution
        dist_n = distrax.MultivariateNormalDiag(loc=mean_n, scale_diag=std_n)

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
        if not self.config.use_lr_decay:
            # Use constant learning rate
            if self.config.adam_weight_decay == 0.0:
                return optax.adam(self.config.learning_rate)
            else:
                return optax.adamw(learning_rate=self.config.learning_rate, weight_decay=self.config.adam_weight_decay)

        # Use cosine decay
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=self.config.learning_rate,
            decay_steps=self.config.lr_decay_steps,
            alpha=self.config.lr_final_multiplier,
        )

        if self.config.adam_weight_decay == 0.0:
            return optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(cosine_schedule))
        else:
            return optax.chain(optax.adamw(learning_rate=cosine_schedule, weight_decay=self.config.adam_weight_decay))

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-headless", name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot-headless"))
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
            ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.75, scale_upper=1.25),
            ksim.JointDampingRandomizer(scale_lower=0.5, scale_upper=2.5),
            ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-3), scale_upper=math.radians(3)),
            ksim.FloorFrictionRandomizer.from_geom_name(
                model=physics_model, floor_geom_name="floor", scale_lower=0.5, scale_upper=1.5
            ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.LinearPushEvent(
                linvel=1.0,  # BUG: this is not used in ksim actually
                vel_range=(0.3, 0.8),
                interval_range=(3.0, 6.0),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v, _ in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(scale=2.0),
            ksim.RandomBaseVelocityXYReset(scale=0.2),
            ksim.RandomHeadingReset(),
            PlaneXYPositionReset(x_range=2.0, y_range=2.0),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(noise=math.radians(2)),
            QposObservation(),  # noise free joint position for critic
            ksim.JointVelocityObservation(noise=math.radians(10)),
            QvelObservation(),  # noise free joint velocity for critic
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
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(10),
            ),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_touch", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_touch", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="base_site_linvel", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="base_site_angvel", noise=0.0),
            FeetPositionObservation.create(
                physics_model=physics_model,
                base_body_name="base",
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
            ),
            BaseHeightObservation(),
            ImuOrientationObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                bias_euler=(0.05, 0.05, 0.0),  # roll, pitch, yaw
                noise=math.radians(1),
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        arm_joint_names = [name for name, _, _ in ZEROS[:10]]
        joint_limits = ksim.get_position_limits(physics_model)
        arm_joint_limits = tuple(zip(*[joint_limits[name] for name in arm_joint_names]))
        return [
            UnifiedCommand(
                vx_range=(-0.5, 1.5),  # m/s
                vy_range=(-0.5, 0.5),  # m/s
                wz_range=(-0.7, 0.7),  # rad/s
                bh_range=(-0.10, 0.0),  # m
                bh_standing_range=(-0.25, 0.0),  # m
                rx_range=(-0.3, 0.3),  # rad
                ry_range=(-0.3, 0.3),  # rad
                arms_range=arm_joint_limits,  # rad
                ctrl_dt=self.config.ctrl_dt,
                switch_prob=self.config.ctrl_dt / 5,  # once per x seconds
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            # cmd
            LinearVelocityTrackingReward(scale=0.3, error_scale=0.1),
            AngularVelocityTrackingReward(scale=0.1, error_scale=0.005),
            XYOrientationReward(scale=0.1, error_scale=0.01),
            TerrainBaseHeightReward.create(
                physics_model=physics_model,
                base_body_name="base",
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
                scale=0.05,
                error_scale=0.05,
                standard_height=0.98,
                foot_origin_height=0.06,
            ),
            ArmPositionReward.create_reward(physics_model, scale=0.2, error_scale=0.05),
            # shaping
            SingleFootContactReward(scale=0.1, ctrl_dt=self.config.ctrl_dt, grace_period=0.15),
            FeetAirtimeReward(scale=1.0, ctrl_dt=self.config.ctrl_dt, touchdown_penalty=0.4),
            FeetOrientationReward.create(
                physics_model=physics_model,
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
                scale=0.02,
                error_scale=0.02,
            ),
            # StandingFeetPositionReward.create(
            #     physics_model=physics_model,
            #     base_body_name="base",
            #     foot_left_body_name="KB_D_501L_L_LEG_FOOT",
            #     foot_right_body_name="KB_D_501R_R_LEG_FOOT",
            #     scale=0.02,
            #     error_scale=0.05,
            #     stance_width=0.30,
            # ),
            # sim2real
            ksim.ActionVelocityPenalty(scale=-0.05),
            ksim.JointVelocityPenalty(scale=-0.05),
            ksim.JointAccelerationPenalty(scale=-0.05),
            ksim.CtrlPenalty(scale=-0.00001),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            TerrainBadZTermination.create(
                physics_model=physics_model,
                base_body_name="base",
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
                unhealthy_z_lower=0.6,
                unhealthy_z_upper=1.2,
            ),
            ksim.NotUprightTermination(max_radians=math.radians(45)),
            ksim.EpisodeLengthTermination(max_length_sec=24),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.LinearCurriculum(
            step_size=1,
            step_every_n_epochs=1,
            min_level=1.0,  # disable curriculum
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        num_joints = len(ZEROS)

        num_commands = (
            2  # linear velocity command (vx, vy)
            + 1  # angular velocity command (wz)
            + 1  # base height command (bh)
            + 2  # base xy orientation command (rx, ry)
            + 10  # arm commands (10)
            + 1  # zero command ohe
        )

        num_actor_inputs = (
            num_joints * 2  # joint pos and vel
            + 4  # imu quat
            + num_commands
            + (3 if self.config.use_gyro else 0)  # imu_gyro
        )

        num_critic_inputs = (
            num_joints * 2  # joint pos and vel
            + 4  # imu quat
            + num_commands
            + 3  # imu gyro
            + 2  # feet touch
            + 6  # feet position
            + 3  # base pos
            + 4  # base quat
            + 138  # COM inertia
            + 230  # COM velocity
            + 3  # base linear vel
            + 3  # base angular vel
            + num_joints  # actuator force
            + 1  # base height
        )

        return Model(
            key,
            num_actor_inputs=num_actor_inputs,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=num_critic_inputs,
            min_std=0.01,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
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
        imu_quat_4 = observations["imu_orientation_observation"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        zero_cmd = (jnp.linalg.norm(commands["unified_command"][..., :3], axis=-1) < 1e-3)[..., None]
        lin_vel_cmd = commands["unified_command"][..., :2]
        ang_vel_cmd = commands["unified_command"][..., 2:3]
        base_height_cmd = commands["unified_command"][..., 4:5]
        base_roll_pitch_cmd = commands["unified_command"][..., 5:7]
        arms_cmd = commands["unified_command"][..., 7:17]

        obs = [
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n,  # NUM_JOINTS
            imu_quat_4,  # 4
            zero_cmd,  # 1
            lin_vel_cmd,  # 2
            ang_vel_cmd,  # 1
            base_height_cmd,  # 1
            base_roll_pitch_cmd,  # 2
            arms_cmd,  # 10
        ]
        if self.config.use_gyro:
            obs += [
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
        qpos_n = observations["qpos_observation"]
        qvel_n = observations["qvel_observation"]
        imu_quat_4 = observations["imu_orientation_observation"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        zero_cmd = (jnp.linalg.norm(commands["unified_command"][..., :3], axis=-1) < 1e-3)[..., None]
        lin_vel_cmd = commands["unified_command"][..., :2]
        ang_vel_cmd = commands["unified_command"][..., 2:3]
        base_height_cmd = commands["unified_command"][..., 3:4]
        base_roll_pitch_cmd = commands["unified_command"][..., 4:6]
        arms_cmd = commands["unified_command"][..., 7:17]

        # privileged obs
        left_touch = observations["sensor_observation_left_foot_touch"]
        right_touch = observations["sensor_observation_right_foot_touch"]
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
                qpos_n,
                qvel_n / 10.0,
                imu_quat_4,
                zero_cmd,
                lin_vel_cmd,
                ang_vel_cmd,
                base_height_cmd,
                base_roll_pitch_cmd,
                arms_cmd,
                imu_gyro_3,
                # privileged obs:
                left_touch,
                right_touch,
                feet_position_6,
                base_position_3,
                base_orientation_4,
                com_inertia_n,
                com_vel_n,
                base_lin_vel_3,
                base_ang_vel_3,
                actuator_force_n / 4.0,
                base_height,
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def _ppo_scan_fn(
        self,
        actor_critic_carry: tuple[Array, Array, Array],
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[tuple[Array, Array, Array], ksim.PPOVariables]:
        transition, rng = xs

        actor_carry, critic_carry, actor_mirror_carry = actor_critic_carry
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

        # compute mirror loss
        mirrored_actor_dist, next_actor_mirror_carry = self.run_actor(
            model=model.actor,
            observations=self.mirror_obs(transition.obs),
            commands=self.mirror_cmd(transition.command),
            carry=actor_mirror_carry,
        )
        unmirrored_actor_dist = self.unmirror_action(mirrored_actor_dist.mean())
        mse_loss = jnp.mean((actor_dist.mean() - unmirrored_actor_dist) ** 2)
        mirror_loss = jnp.mean(mse_loss) * self.config.mirror_loss_scale

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=jnp.expand_dims(log_probs, axis=0),
            values=value.squeeze(-1),
            entropy=jnp.expand_dims(actor_dist.entropy(), axis=0),
            action_std=actor_dist.stddev(),
            aux_losses={"mirror_loss": mirror_loss},
        )

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            (
                jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
                jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
                jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            ),
            (next_actor_carry, next_critic_carry, next_actor_mirror_carry),
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

        # add a third carry for the mirror actor
        model_carry = model_carry + (jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=4,
        )
        return ppo_variables, next_model_carry[:-1]

    def get_initial_model_carry(self, model: Model, rng: PRNGKeyArray) -> tuple[Array, Array]:
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

    def mirror_joints(self, j: Array) -> Array:
        assert j.shape[0] == 20, "Joints must be 20-dimensional"
        j_m = jnp.zeros_like(j)

        # Arms (first 10 joints)
        # Mirror right arm (0-4) to left arm positions (5-9)
        j_m = j_m.at[0:5].set(j[5:10])
        # Mirror left arm (5-9) to right arm positions (0-4)
        j_m = j_m.at[5:10].set(j[0:5])

        # Legs (next 10 joints)
        # Mirror right leg (10-14) to left leg positions (15-19)
        j_m = j_m.at[10:15].set(j[15:20])
        # Mirror left leg (15-19) to right leg positions (10-14)
        j_m = j_m.at[15:20].set(j[10:15])

        # Negate roll and yaw angles while preserving pitch
        # For arms: pitch=0,4,5,9; roll=1,6; yaw=2,7
        j_m = j_m.at[1].multiply(-1)  # right shoulder roll
        j_m = j_m.at[2].multiply(-1)  # right shoulder yaw
        j_m = j_m.at[6].multiply(-1)  # left shoulder roll
        j_m = j_m.at[7].multiply(-1)  # left shoulder yaw

        # For legs: pitch=10,13,15,18; roll=11,16; yaw=12,17
        j_m = j_m.at[11].multiply(-1)  # right hip roll
        j_m = j_m.at[12].multiply(-1)  # right hip yaw
        j_m = j_m.at[16].multiply(-1)  # left hip roll
        j_m = j_m.at[17].multiply(-1)  # left hip yaw

        return j_m

    def mirror_obs(self, obs: xax.FrozenDict[str, Array]) -> xax.FrozenDict[str, Array]:
        # only mirror obs the actor takes as input
        joint_pos_n_m = self.mirror_joints(obs["joint_position_observation"])
        joint_vel_n_m = self.mirror_joints(obs["joint_velocity_observation"])
        imu_quat_4_m = jnp.concatenate(
            [
                obs["imu_orientation_observation"][..., :1],
                -obs["imu_orientation_observation"][..., 1:2],
                obs["imu_orientation_observation"][..., 2:3],
                -obs["imu_orientation_observation"][..., 3:],
            ],
            axis=-1,
        )
        imu_gyro_3_m = jnp.concatenate(
            [
                -obs["sensor_observation_imu_gyro"][..., 0:1],
                obs["sensor_observation_imu_gyro"][..., 1:2],
                -obs["sensor_observation_imu_gyro"][..., 2:3],
            ],
            axis=-1,
        )

        obs_m = {
            "joint_position_observation": joint_pos_n_m,
            "joint_velocity_observation": joint_vel_n_m,
            "imu_orientation_observation": imu_quat_4_m,
            "sensor_observation_imu_gyro": imu_gyro_3_m,
        }
        return obs_m

    def mirror_cmd(self, cmd: xax.FrozenDict[str, Array]) -> xax.FrozenDict[str, Array]:
        cmd_u = cmd["unified_command"]
        cmd_u_m = jnp.concatenate(
            [
                cmd_u[..., :1],  # x
                -cmd_u[..., 1:2],  # y
                -cmd_u[..., 2:3],  # z
                -cmd_u[..., 3:4],  # heading carry
                cmd_u[..., 4:5],  # base height
                -cmd_u[..., 5:6],  # base roll
                cmd_u[..., 6:7],  # base pitch
                self.mirror_joints(
                    jnp.concatenate(
                        [
                            cmd_u[..., 7:17],
                            jnp.zeros(shape=(10,)),
                        ]
                    )
                )[..., :10],  # arms
            ],
            axis=-1,
        )
        cmd = {"unified_command": cmd_u_m}
        return cmd

    def unmirror_action(self, action: ksim.Action) -> ksim.Action:
        action_m = self.mirror_joints(action)
        return action_m


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=4096,
            batch_size=512,
            num_passes=3,
            epochs_per_log_step=1,
            rollout_length_seconds=2.0,
            global_grad_clip=2.0,
            entropy_coef=0.002,
            learning_rate=5e-4,
            gamma=0.9,
            lam=0.94,
            mirror_loss_scale=0.01,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            # sim2real parameters.
            action_latency_range=(0.003, 0.01),  # Simulate 3-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            render_track_body_id=0,
            render_full_every_n_seconds=0,
            render_length_seconds=10,
            max_values_per_plot=50,
            # Checkpointing parameters.
            save_every_n_seconds=60,
            valid_every_n_steps=100,
            valid_every_n_seconds=None,
        ),
    )
