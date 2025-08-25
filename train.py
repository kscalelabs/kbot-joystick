"""Defines simple task for training a joystick walking policy for K-Bot."""

import asyncio
import functools
import math
from dataclasses import dataclass
from typing import Self, TypedDict

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
# Joint name, neutral position
JOINT_BIASES: list[tuple[str, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0)),
    ("dof_right_shoulder_yaw_02", 0.0),
    ("dof_right_elbow_02", math.radians(90.0)),
    ("dof_right_wrist_00", 0.0),
    ("dof_left_shoulder_pitch_03", 0.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0)),
    ("dof_left_shoulder_yaw_02", 0.0),
    ("dof_left_elbow_02", math.radians(-90.0)),
    ("dof_left_wrist_00", 0.0),
    ("dof_right_hip_pitch_04", math.radians(-20.0)),
    ("dof_right_hip_roll_03", math.radians(-0.0)),
    ("dof_right_hip_yaw_03", 0.0),
    ("dof_right_knee_04", math.radians(-50.0)),
    ("dof_right_ankle_02", math.radians(30.0)),
    ("dof_left_hip_pitch_04", math.radians(20.0)),
    ("dof_left_hip_roll_03", math.radians(0.0)),
    ("dof_left_hip_yaw_03", 0.0),
    ("dof_left_knee_04", math.radians(50.0)),
    ("dof_left_ankle_02", math.radians(-30.0)),
]

JOINT_LIMITS = {
    "dof_right_shoulder_pitch_03": (-3.1415929794311523, 1.3962630033493042),
    "dof_right_shoulder_roll_03": (-1.6580630540847778, 0.34906598925590515),
    "dof_right_shoulder_yaw_02": (-1.6580630540847778, 1.6580630540847778),
    "dof_right_elbow_02": (0.0, 2.478368043899536),
    "dof_right_wrist_00": (-1.7453290224075317, 1.7453290224075317),
    "dof_left_shoulder_pitch_03": (-1.3962630033493042, 3.1415929794311523),
    "dof_left_shoulder_roll_03": (-0.34906598925590515, 1.6580630540847778),
    "dof_left_shoulder_yaw_02": (-1.6580630540847778, 1.6580630540847778),
    "dof_left_elbow_02": (-2.478368043899536, 0.0),
    "dof_left_wrist_00": (-1.7453290224075317, 1.7453290224075317),
    "dof_right_hip_pitch_04": (-2.2165679931640625, 1.0471980571746826),
    "dof_right_hip_roll_03": (-2.268928050994873, 0.20943999290466309),
    "dof_right_hip_yaw_03": (-1.570796012878418, 1.570796012878418),
    "dof_right_knee_04": (-2.7052600383758545, 0.0),
    "dof_right_ankle_02": (-0.22689299285411835, 1.2566369771957397),
    "dof_left_hip_pitch_04": (-1.0471980571746826, 2.2165679931640625),
    "dof_left_hip_roll_03": (-0.20943999290466309, 2.268928050994873),
    "dof_left_hip_yaw_03": (-1.570796012878418, 1.570796012878418),
    "dof_left_knee_04": (0.0, 2.7052600383758545),
    "dof_left_ankle_02": (-1.2566369771957397, 0.22689299285411835),
}


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
    cutoff_frequency: float = xax.field(
        value=4.0,
        help="The cutoff frequency for the low-pass filter.",
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
    actor_mirror_loss_scale: float = xax.field(
        value=1.0,
        help="Scale for the actor mirror loss",
    )
    critic_mirror_loss_scale: float = xax.field(
        value=0.01,
        help="Scale for the critic mirror loss",
    )


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
        left_contact = jnp.where(traj.obs["left_foot_touch"] > 0.1, True, False)[:, 0]
        right_contact = jnp.where(traj.obs["right_foot_touch"] > 0.1, True, False)[:, 0]
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
class NoContactPenalty(ksim.Reward):
    """Penalty for having no contact with the ground when walking."""

    scale: float = -0.1

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        left_contact = jnp.where(traj.obs["left_foot_touch"] > 0.1, True, False)[:, 0]
        right_contact = jnp.where(traj.obs["right_foot_touch"] > 0.1, True, False)[:, 0]
        is_zero_cmd = jnp.linalg.norm(traj.command["unified_command"][:, :3], axis=-1) < 1e-3
        return jnp.where(is_zero_cmd, 0.0, jnp.where(jnp.logical_or(left_contact, right_contact), 0.0, 1.0))


@attrs.define(frozen=True, kw_only=True)
class FeetAirtimeReward(ksim.StatefulReward):
    """Encourages reasonable step frequency by rewarding long swing phases and penalizing quick stepping."""

    scale: float = 1.0
    ctrl_dt: float = 0.02
    touchdown_penalty: float = 0.4
    scale_by_curriculum: bool = False

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        airtime_carry = jnp.array([0.0, 0.0])
        contact_carry = jnp.array([False, False])
        return airtime_carry, contact_carry

    def _compute_airtime(self, initial_airtime: Array, contact_bool: Array, done: Array) -> tuple[Array, Array]:
        """Returns an array with the airtime (in seconds) for each timestep."""

        def _body(time_since_liftoff: Array, is_contact: Array) -> tuple[Array, Array]:
            new_time = jnp.where(is_contact, 0.0, time_since_liftoff + self.ctrl_dt)
            return new_time, new_time

        contact_or_done = jnp.logical_or(contact_bool, done[:, None])
        carry, airtime = jax.lax.scan(_body, initial_airtime, contact_or_done)
        return carry, airtime

    def _compute_first_contact(self, contact_carry: Array, contact_bool: Array) -> Array:
        """Returns a boolean array indicating if a timestep is the first contact after flight."""
        prev_contact = jnp.concatenate([contact_carry[None, :], contact_bool[:-1]], axis=0)
        first_contact = jnp.logical_and(contact_bool, jnp.logical_not(prev_contact))
        return first_contact

    def get_reward_stateful(self, traj: ksim.Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        airtime_carry, contact_carry = reward_carry

        contact_l = jnp.where(traj.obs["left_foot_touch"] > 0.1, True, False)
        contact_r = jnp.where(traj.obs["right_foot_touch"] > 0.1, True, False)
        contact = jnp.stack([contact_l[:, 0], contact_r[:, 0]], axis=-1)

        new_airtime_carry, airtime = self._compute_airtime(airtime_carry, contact, traj.done)
        first_contact = self._compute_first_contact(contact_carry, contact) * ~traj.done[:, None]
        # shift airtime by 1 to match touchdowns with previous step airtimes
        airtime = jnp.concatenate([airtime_carry[None, :], airtime], axis=0)[:-1, :]
        reward = jnp.sum((airtime - self.touchdown_penalty) * first_contact.astype(jnp.float32), axis=-1)

        is_zero_cmd = jnp.linalg.norm(traj.command["unified_command"][:, :3], axis=-1) < 1e-3
        reward = jnp.where(is_zero_cmd, 0.0, reward)
        reward_carry = (new_airtime_carry, contact[-1, :])
        return reward, reward_carry


@attrs.define(frozen=True, kw_only=True)
class ArmPositionReward(ksim.Reward):
    """Reward for tracking commanded arm joint positions.

    Compares the current arm joint positions against commanded positions from
    trajectory.command["unified_command"][6:].
    """

    joint_indices: Array = attrs.field(eq=False)
    joint_biases: Array = attrs.field(eq=False)
    error_scale: float = attrs.field(default=0.1)

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
        joint_biases = jnp.array([bias for (name, bias) in JOINT_BIASES if name in joint_names])

        return cls(
            joint_indices=joint_indices,
            joint_biases=joint_biases,
            error_scale=error_scale,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        qpos_sel = trajectory.qpos[..., jnp.array(self.joint_indices) + 7]
        target = trajectory.command["unified_command"][..., 6:16] + self.joint_biases
        error = xax.get_norm(qpos_sel - target, norm="l2").sum(axis=-1)
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # get base quat, yaw only
        base_euler = xax.quat_to_euler(trajectory.xquat[:, 1, :])
        base_euler = base_euler.at[:, :2].set(0.0)
        base_z_quat = xax.euler_to_quat(base_euler)

        # rotate local frame commands to global frame
        robot_vel_cmd = jnp.pad(trajectory.command["unified_command"][:, :2], ((0, 0), (0, 1)))
        global_vel_cmd = xax.rotate_vector_by_quat(robot_vel_cmd, base_z_quat, inverse=False)

        # drop vz. vz conflicts with base height reward.
        global_vel_xy_cmd = global_vel_cmd[:, :2]
        global_vel_xy = trajectory.qvel[:, :2]

        # compute error. steep kernel for standing, smooth for walking.
        zero_cmd_mask = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        vel_error = jnp.linalg.norm(global_vel_xy - global_vel_xy_cmd, axis=-1)
        error = jnp.where(zero_cmd_mask, vel_error, jnp.square(vel_error))
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityReward(ksim.Reward):
    """Reward for tracking the angular velocity."""

    error_scale: float = attrs.field(default=0.25)

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        base_ang_vel = traj.qvel[:, 5]
        base_ang_vel_cmd = traj.command["unified_command"][:, 2]

        ang_vel_error = jnp.abs(base_ang_vel - base_ang_vel_cmd)
        return jnp.exp(-ang_vel_error / self.error_scale)


@attrs.define(frozen=True)
class XYOrientationReward(ksim.Reward):
    """Reward for tracking the xy base orientation using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.03)
    error_scale_zero_cmd: float = attrs.field(default=0.003)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        euler_orientation = xax.quat_to_euler(trajectory.xquat[:, 1, :])
        euler_orientation = euler_orientation.at[:, 2].set(0.0)  # ignore yaw
        base_xy_quat = xax.euler_to_quat(euler_orientation)

        commanded_euler = jnp.stack(
            [
                trajectory.command["unified_command"][:, 4],
                trajectory.command["unified_command"][:, 5],
                jnp.zeros_like(trajectory.command["unified_command"][:, 5]),
            ],
            axis=-1,
        )
        base_xy_quat_cmd = xax.euler_to_quat(commanded_euler)
        quat_error = 1 - jnp.sum(base_xy_quat_cmd * base_xy_quat, axis=-1) ** 2

        is_zero_cmd = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        error_scale = jnp.where(is_zero_cmd, self.error_scale_zero_cmd, self.error_scale)
        return jnp.exp(-quat_error / error_scale)


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
        commanded_height = trajectory.command["unified_command"][:, 3] + self.standard_height
        height_diff = current_height - commanded_height

        # for walking: only care about minimum height.
        is_zero_cmd = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        height_error = jnp.where(is_zero_cmd, jnp.abs(height_diff), jnp.abs(jnp.minimum(height_diff, 0.0)))
        return jnp.exp(-height_error / self.error_scale)


@attrs.define(frozen=True)
class FeetOrientationReward(ksim.Reward):
    """Reward for keeping feet oriented parallel to the ground.

    For linear walking and standing, this reward considers roll, pitch, and yaw angles
    of the feet. For angular walking (turning/rotating), it only considers roll and pitch,
    allowing the feet to yaw freely.
    """

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
        base_yaw = xax.quat_to_euler(trajectory.xquat[:, 1, :])[:, 2]
        straight_foot_euler = jnp.stack(
            [
                jnp.full_like(base_yaw, -jnp.pi / 2),
                jnp.zeros_like(base_yaw),
                base_yaw - jnp.pi,
            ],
            axis=-1,
        )

        # compute rpy error
        straight_foot_quat = xax.euler_to_quat(straight_foot_euler)
        left_quat_error = 1 - jnp.sum(straight_foot_quat * trajectory.xquat[:, self.foot_left_idx, :], axis=-1) ** 2
        right_quat_error = 1 - jnp.sum(straight_foot_quat * trajectory.xquat[:, self.foot_right_idx, :], axis=-1) ** 2
        rpy_error = left_quat_error + right_quat_error

        # compute rp error
        feet_euler = xax.quat_to_euler(trajectory.xquat[:, [self.foot_left_idx, self.foot_right_idx], :])
        feet_quat = xax.euler_to_quat(feet_euler.at[:, :, 2].set(0.0))

        straight_foot_quat = xax.euler_to_quat(straight_foot_euler.at[:, 2].set(0.0))
        rp_error = (1 - jnp.sum(straight_foot_quat[:, None, :] * feet_quat, axis=-1) ** 2).sum(axis=1)

        # choose rp error or rpy error based on command
        is_rotating = jnp.linalg.norm(trajectory.command["unified_command"][:, 2], axis=-1) > 1e-3
        error = jnp.where(is_rotating, rp_error, rpy_error)
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class BaseAccelerationReward(ksim.Reward):
    """Reward for minimizing base acceleration calculated from qvel."""

    error_scale: float = attrs.field(default=1.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        base_vel = trajectory.qvel[:, :6]  # (T, 6)
        base_vel_padded = jnp.pad(base_vel, ((1, 0), (0, 0)), mode="edge")  # (T+1, 6)
        done_padded = jnp.pad(trajectory.done, ((1, 0),), mode="edge")  # (T+1,)
        vel_diff = base_vel_padded[1:] - base_vel_padded[:-1]  # (T, 6)
        acc = jnp.where(done_padded[:-1, None], 0.0, vel_diff)  # (T, 6)
        error = jnp.abs(acc).sum(axis=-1)  # (T,)
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class JointVelocityPenalty(ksim.Reward):
    """Penalty for how fast the joint angular velocities are changing."""

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        qpos = trajectory.qpos[..., 7:]
        qpos_zp = jnp.pad(qpos, ((1, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((1, 0),), mode="edge")[..., :-1, None]
        qvel = jnp.where(done, 0.0, qpos_zp[..., 1:, :] - qpos_zp[..., :-1, :])
        return xax.get_norm(qvel, "l2").mean(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class JointAccelerationPenalty(ksim.Reward):
    """Penalty for high joint accelerations."""

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        qpos = trajectory.qpos[..., 7:]
        qpos_zp = jnp.pad(qpos, ((2, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((2, 0),), mode="edge")[..., :-1, None]
        qvel = jnp.where(done, 0.0, qpos_zp[..., 1:, :] - qpos_zp[..., :-1, :])
        qacc = jnp.where(done[..., 1:, :], 0.0, qvel[..., 1:, :] - qvel[..., :-1, :])
        penalty = xax.get_norm(qacc, "l2").mean(axis=-1)
        return penalty


@attrs.define(frozen=True, kw_only=True)
class CtrlPenalty(ksim.Reward):
    """Penalty for large torque commands."""

    scales: tuple[float, ...] | None = attrs.field(default=None)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        ctrl = trajectory.ctrl
        if self.scales is not None:
            ctrl = ctrl / jnp.array(self.scales)
        return xax.get_norm(ctrl, "l2").mean(axis=-1)

    @classmethod
    def create(cls, model: ksim.PhysicsModel, scale: float = -1.0, scale_by_curriculum: bool = False) -> Self:
        ctrl_min = model.actuator_ctrlrange[..., 0]
        ctrl_max = model.actuator_ctrlrange[..., 1]
        ctrl_range = (ctrl_max - ctrl_min) / 2.0
        ctrl_range_list = ctrl_range.flatten().tolist()
        return cls(
            scales=tuple(ctrl_range_list),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


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


@attrs.define(frozen=True)
class UnifiedCommand(ksim.Command):
    """Unifiying all commands into one to allow for covariance control."""

    vx_range: tuple[float, float] = attrs.field()
    vy_range: tuple[float, float] = attrs.field()
    wz_range: tuple[float, float] = attrs.field()
    bh_range: tuple[float, float] = attrs.field()
    rx_range: tuple[float, float] = attrs.field()
    ry_range: tuple[float, float] = attrs.field()
    arms_range: tuple[tuple[float, ...], ...] = attrs.field()
    ctrl_dt: float = attrs.field()
    switch_prob: float = attrs.field()

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_a, rng_b, rng_c, rng_d, rng_e, rng_f, rng_g, rng_h, rng_i = jax.random.split(rng, 9)

        # cmd  = [vx, vy, wz, bh, rx, ry]
        vx = jax.random.uniform(rng_b, (1,), minval=self.vx_range[0], maxval=self.vx_range[1])
        vy = jax.random.uniform(rng_c, (1,), minval=self.vy_range[0], maxval=self.vy_range[1])
        wz = jax.random.uniform(rng_d, (1,), minval=self.wz_range[0], maxval=self.wz_range[1])
        bh = jax.random.uniform(rng_e, (1,), minval=self.bh_range[0], maxval=self.bh_range[1])
        rx = jax.random.uniform(rng_f, (1,), minval=self.rx_range[0], maxval=self.rx_range[1])
        ry = jax.random.uniform(rng_g, (1,), minval=self.ry_range[0], maxval=self.ry_range[1])

        # 50% chance to sample from wide uniform distribution
        arms_uni = jax.random.uniform(
            rng_h, (10,), minval=jnp.array(self.arms_range[0]), maxval=jnp.array(self.arms_range[1])
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
        forward_cmd = jnp.concatenate([vx, _, _, _, _, _, __])
        sideways_cmd = jnp.concatenate([_, vy, _, _, _, _, __])
        rotate_cmd = jnp.concatenate([_, _, wz, _, _, _, __])
        omni_cmd = jnp.concatenate([vx, vy, wz, _, _, _, arms])
        stand_bend_cmd = jnp.concatenate([_, _, _, bh, rx, ry, arms])
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

        assert cmd.shape == (16,)
        return cmd

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        # def update_arms(prev_command: Array) -> Array:
        #     """Move arm commands by x rad/s."""
        #     arms = prev_command[6:16]
        #     arm_mask = jnp.where(arms != 0.0, 1.0, 0.0)
        #     arms = arms + arm_mask * self.ctrl_dt * 0.5
        #     arms = arms.clip(self.arms_range[:, 0], self.arms_range[:, 1])
        #     prev_command = prev_command.at[6:16].set(arms)
        #     return prev_command

        # continued_command = update_arms(continued_command)

        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_command = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_command, prev_command)


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
    """Resets the robot's XY position."""

    x_range: float = attrs.field(default=1.0)
    y_range: float = attrs.field(default=1.0)

    def __call__(self, data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> ksim.PhysicsData:
        keyx, keyy = jax.random.split(rng)
        new_x = jax.random.uniform(keyx, (1,), minval=-self.x_range, maxval=self.x_range)
        new_y = jax.random.uniform(keyy, (1,), minval=-self.y_range, maxval=self.y_range)

        qpos_j = data.qpos
        if not isinstance(qpos_j, jnp.ndarray):
            return data
        qpos_j = qpos_j.at[0:1].set(new_x)
        qpos_j = qpos_j.at[1:2].set(new_y)
        data = ksim.update_data_field(data, "qpos", qpos_j)
        return data


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.LSTMCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.field()
    num_outputs: int = eqx.field()
    clip_positions: ksim.TanhPositions = eqx.field()
    min_std: float = eqx.field()
    max_std: float = eqx.field()
    var_scale: float = eqx.field()
    cutoff_frequency: float = eqx.field()
    ctrl_dt: float = eqx.field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        physics_model: ksim.PhysicsModel,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
        cutoff_frequency: float,
        ctrl_dt: float,
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
                eqx.nn.LSTMCell(
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
        self.clip_positions = ksim.TanhPositions.from_physics_model(physics_model)
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.cutoff_frequency = cutoff_frequency
        self.ctrl_dt = ctrl_dt

    def forward(
        self, obs_n: Array, carry: Array | tuple[tuple[Array, ...], ...], lpf_params: ksim.LowPassFilterParams
    ) -> tuple[distrax.Distribution, tuple[tuple[Array, ...], ...], ksim.LowPassFilterParams]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            h, c = rnn(x_n, carry[i])
            out_carries.append((h, c))
            x_n = h
        out_n = self.output_proj(h)

        # Split into means and stds
        mean_n = out_n[..., : self.num_outputs]
        std_n = out_n[..., self.num_outputs :]

        # Softplus and clip to ensure positive standard deviations
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means
        arm_cmd_bias = jnp.concatenate([obs_n[..., -10:], jnp.zeros(shape=(10,))], axis=-1)
        mean_n = mean_n + jnp.array([v for _, v in JOINT_BIASES]) + arm_cmd_bias

        # Clip the target positions to the minimum and maximum ranges.
        # mean_n = self.clip_positions.clip(mean_n) # TODO disable for now - its bad

        # Apply low-pass filter
        mean_n, lpf_params = ksim.lowpass_one_pole(mean_n, self.ctrl_dt, self.cutoff_frequency, lpf_params)

        # Create diagonal gaussian distribution
        dist_n = distrax.MultivariateNormalDiag(loc=mean_n, scale_diag=std_n)

        return dist_n, tuple(out_carries), lpf_params


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.LSTMCell, ...]
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
                eqx.nn.LSTMCell(
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

    def forward(
        self, obs_n: Array, carry: tuple[tuple[Array, ...], ...]
    ) -> tuple[Array, tuple[tuple[Array, Array], ...]]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            h, c = rnn(x_n, carry[i])
            out_carries.append((h, c))
            x_n = h
        out_n = self.output_proj(h)

        return out_n, tuple(out_carries)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        physics_model: ksim.PhysicsModel,
        num_actor_inputs: int,
        num_actor_outputs: int,
        num_critic_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
        cutoff_frequency: float,
        ctrl_dt: float,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = Actor(
            actor_key,
            physics_model=physics_model,
            num_inputs=num_actor_inputs,
            num_outputs=num_actor_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            depth=depth,
            cutoff_frequency=cutoff_frequency,
            ctrl_dt=ctrl_dt,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
        )


class Carry(TypedDict):
    actor: tuple[tuple[Array, Array], ...]  # for every layer in lstm: (h, c)
    actor_mirror: tuple[tuple[Array, Array], ...]
    critic: tuple[tuple[Array, Array], ...]
    critic_mirror: tuple[tuple[Array, Array], ...]
    lpf_params: ksim.LowPassFilterParams


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

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.PhysicsRandomizer]:
        return {
            "static_friction": ksim.StaticFrictionRandomizer(),
            "armature": ksim.ArmatureRandomizer(),
            "mass_multiplication": ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.75, scale_upper=1.25),
            "joint_damping": ksim.JointDampingRandomizer(scale_lower=0.5, scale_upper=2.5),
            "joint_zero_position": ksim.JointZeroPositionRandomizer(
                scale_lower=math.radians(-3), scale_upper=math.radians(3)
            ),
            "floor_friction": ksim.FloorFrictionRandomizer.from_geom_name(
                model=physics_model, floor_geom_name="floor", scale_lower=0.5, scale_upper=1.5
            ),
        }

    def get_events(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Event]:
        return {
            "linear push": ksim.LinearPushEvent(
                linvel=1.0,  # BUG: this is not used in ksim actually
                vel_range=(0.3, 0.8),
                interval_range=(3.0, 6.0),
            ),
        }

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in JOINT_BIASES}, scale=0.1),
            ksim.RandomJointVelocityReset(scale=2.0),
            ksim.RandomBaseVelocityXYReset(scale=0.2),
            ksim.RandomHeadingReset(),
            PlaneXYPositionReset(x_range=2.0, y_range=2.0),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Observation]:
        return {
            "joint_position": ksim.JointPositionObservation(noise=ksim.AdditiveUniformNoise(mag=math.radians(2))),
            "joint_velocity": ksim.JointVelocityObservation(
                noise=ksim.MultiplicativeUniformNoise(mag=math.radians(10))  # TODO confirm noise levels
            ),
            "actuator_force": ksim.ActuatorForceObservation(),
            "center_of_mass_inertia": ksim.CenterOfMassInertiaObservation(),
            "center_of_mass_velocity": ksim.CenterOfMassVelocityObservation(),
            "base_position": ksim.BasePositionObservation(),
            "base_orientation": ksim.BaseOrientationObservation(),
            "base_linear_velocity": ksim.BaseLinearVelocityObservation(),
            "base_angular_velocity": ksim.BaseAngularVelocityObservation(),
            "base_linear_acceleration": ksim.BaseLinearAccelerationObservation(),
            "base_angular_acceleration": ksim.BaseAngularAccelerationObservation(),
            "actuator_acceleration": ksim.ActuatorAccelerationObservation(),
            "imu_gyro": ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=ksim.AdditiveGaussianNoise(std=math.radians(10)),  # TODO confirm noise level
            ),
            "left_foot_touch": ksim.SensorObservation.create(
                physics_model=physics_model, sensor_name="left_foot_touch"
            ),
            "right_foot_touch": ksim.SensorObservation.create(
                physics_model=physics_model, sensor_name="right_foot_touch"
            ),
            "base_site_linvel": ksim.SensorObservation.create(
                physics_model=physics_model, sensor_name="base_site_linvel"
            ),
            "base_site_angvel": ksim.SensorObservation.create(
                physics_model=physics_model, sensor_name="base_site_angvel"
            ),
            "feet_position": FeetPositionObservation.create(
                physics_model=physics_model,
                base_body_name="base",
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
            ),
            "base_height": BaseHeightObservation(),
            "imu_projected_gravity": ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                noise=ksim.AdditiveGaussianNoise(std=math.radians(1)),  # TODO confirm noise level
                min_lag=0.0,
                max_lag=0.1,
                bias=math.radians(2),
            ),
            "projected_gravity": ksim.ProjectedGravityObservation.create(  # TODO confirm this is even needed
                physics_model=physics_model,
                framequat_name="base_site_quat",
            ),
        }

    def get_commands(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Command]:
        arm_joint_names = [name for name, _ in JOINT_BIASES[:10]]
        joint_limits = ksim.get_position_limits(physics_model)
        arm_joint_limits = tuple(zip(*[joint_limits[name] for name in arm_joint_names]))
        return {
            "unified_command": UnifiedCommand(
                vx_range=(-0.5, 1.5),  # m/s
                vy_range=(-0.5, 0.5),  # m/s
                wz_range=(-1.0, 1.0),  # rad/s
                bh_range=(-0.25, 0.05),  # m
                rx_range=(-0.3, 0.3),  # rad
                ry_range=(-0.3, 0.3),  # rad
                arms_range=arm_joint_limits,  # rad
                ctrl_dt=self.config.ctrl_dt,
                switch_prob=self.config.ctrl_dt / 5,  # once per x seconds
            ),
        }

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Reward]:
        return {
            # cmd
            "linvel": LinearVelocityTrackingReward(scale=0.2, error_scale=0.2),
            "angvel": AngularVelocityReward(scale=0.1, error_scale=0.2),
            "roll_pitch": XYOrientationReward(scale=0.1, error_scale=0.03, error_scale_zero_cmd=0.01),
            "base_height": TerrainBaseHeightReward.create(
                physics_model=physics_model,
                base_body_name="base",
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
                scale=0.2,
                error_scale=0.02,
                standard_height=0.99,
                foot_origin_height=0.06,
            ),
            "arm_pos": ArmPositionReward.create_reward(physics_model, scale=0.2, error_scale=0.1),
            # shaping
            "single_contact": SingleFootContactReward(scale=0.1, ctrl_dt=self.config.ctrl_dt, grace_period=2.0),
            "no_contact_p": NoContactPenalty(scale=-0.1),
            "feet_airtime": FeetAirtimeReward(scale=1.5, ctrl_dt=self.config.ctrl_dt, touchdown_penalty=0.4),
            "feet_orient": FeetOrientationReward.create(
                physics_model=physics_model,
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
                scale=0.1,
                error_scale=0.02,
            ),
            # sim2real
            "base_accel": BaseAccelerationReward(scale=0.1, error_scale=5.0),
            "action_vel": ksim.ActionVelocityPenalty(scale=-0.05),
            "joint_vel": JointVelocityPenalty(scale=-0.05),  # TODO
            "joint_accel": JointAccelerationPenalty(scale=-0.05),  # TODO
            "ctrl": CtrlPenalty(scale=-0.00001),  # TODO maybe ksim new version?
        }

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Termination]:
        return {
            "bad_z": TerrainBadZTermination.create(
                physics_model=physics_model,
                base_body_name="base",
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
                unhealthy_z_lower=0.6,
                unhealthy_z_upper=1.2,
            ),
            "not_upright": ksim.NotUprightTermination(max_radians=math.radians(45)),
            "episode_length": ksim.EpisodeLengthTermination(max_length_sec=24),
        }

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.LinearCurriculum(
            step_size=1,
            step_every_n_epochs=1,
            min_level=1.0,  # disable curriculum
        )

    def get_model(self, params: ksim.InitParams) -> Model:
        num_joints = len(JOINT_BIASES)

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
            + 5 # projected gravity
            + 3  # imu_gyro
            + num_commands
        )

        num_critic_inputs = (
            num_joints * 2  # joint pos and vel
            + 5  # projected gravity
            + 3  # imu gyro
            + num_commands
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
            params.key,
            physics_model=params.physics_model,
            num_actor_inputs=num_actor_inputs,
            num_actor_outputs=len(JOINT_BIASES),
            num_critic_inputs=num_critic_inputs,
            min_std=0.01,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            cutoff_frequency=self.config.cutoff_frequency,
            ctrl_dt=self.config.ctrl_dt,
        )

    def normalize_joint_pos(self, joint_pos: Array) -> Array:
        # Get joint biases and limits for normalization
        joint_names = [name for name, _ in JOINT_BIASES]
        joint_biases = jnp.array([bias for _, bias in JOINT_BIASES])
        joint_min = jnp.array([JOINT_LIMITS[name][0] for name in joint_names])
        joint_max = jnp.array([JOINT_LIMITS[name][1] for name in joint_names])

        # Calculate maximum range from bias for each joint
        range_negative = joint_biases - joint_min
        range_positive = joint_max - joint_biases
        max_range = jnp.maximum(range_negative, range_positive)

        # Normalize joint positions relative to bias, scaled by max range
        return (joint_pos - joint_biases) / max_range

    def normalize_joint_vel(self, joint_vel: Array) -> Array:
        return joint_vel / 10.0

    def encode_projected_gravity(self, projected_gravity: Array) -> Array:
        roll = jnp.arctan2(projected_gravity[1], -projected_gravity[2])
        pitch = jnp.arctan2(-projected_gravity[0], jnp.sqrt(projected_gravity[1] ** 2 + projected_gravity[2] ** 2))
        projected_gravity_unit = projected_gravity / jnp.linalg.norm(projected_gravity, axis=-1, keepdims=True)
        return jnp.concatenate(
            [
                roll[..., None],
                pitch[..., None],
                projected_gravity_unit,
            ],
            axis=-1,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: tuple[tuple[Array, ...], ...],
        lpf_params: ksim.LowPassFilterParams,
    ) -> tuple[distrax.Distribution, tuple[tuple[Array, ...], ...], ksim.LowPassFilterParams]:
        joint_pos_n = observations["noisy_joint_position"]
        joint_vel_n = observations["noisy_joint_velocity"]
        projected_gravity_3 = observations["noisy_imu_projected_gravity"]
        imu_gyro_3 = observations["noisy_imu_gyro"]
        cmd = commands["unified_command"]
        zero_cmd = (jnp.linalg.norm(cmd[..., :3], axis=-1) < 1e-3)[..., None]

        obs = [
            self.normalize_joint_pos(joint_pos_n),  # NUM_JOINTS
            self.normalize_joint_vel(joint_vel_n),  # NUM_JOINTS
            self.encode_projected_gravity(projected_gravity_3),  # 5
            imu_gyro_3,  # 3
            zero_cmd,  # 1
            cmd,  # 16
        ]

        obs_n = jnp.concatenate(obs, axis=-1)
        action, carry, lpf_params = model.forward(obs_n, carry, lpf_params)

        return action, carry, lpf_params

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: tuple[tuple[Array, ...], ...],
    ) -> tuple[Array, tuple[tuple[Array, ...], ...]]:
        qpos_n = observations["joint_position"]
        qvel_n = observations["joint_velocity"]
        projected_gravity_3 = observations["projected_gravity"]
        imu_gyro_3 = observations["imu_gyro"]
        cmd = commands["unified_command"]
        zero_cmd = (jnp.linalg.norm(cmd[..., :3], axis=-1) < 1e-3)[..., None]

        # privileged obs
        left_touch = observations["left_foot_touch"]
        right_touch = observations["right_foot_touch"]
        feet_position_6 = observations["feet_position"]
        base_position_3 = observations["base_position"]
        base_orientation_4 = observations["base_orientation"]
        com_inertia_n = observations["center_of_mass_inertia"]
        com_vel_n = observations["center_of_mass_velocity"]
        base_lin_vel_3 = observations["base_linear_velocity"]
        base_ang_vel_3 = observations["base_angular_velocity"]
        actuator_force_n = observations["actuator_force"]
        base_height = observations["base_height"]

        obs_n = jnp.concatenate(
            [
                # actor obs:
                self.normalize_joint_pos(qpos_n),
                self.normalize_joint_vel(qvel_n),
                self.encode_projected_gravity(projected_gravity_3),
                imu_gyro_3,
                zero_cmd,
                cmd,
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
        carry: Carry,
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[Carry, ksim.PPOVariables]:
        transition, rng = xs

        actor_c = carry["actor"]
        critic_c = carry["critic"]
        actor_c_m = carry["actor_mirror"]
        critic_c_m = carry["critic_mirror"]

        actor_dist, next_actor_c, next_lpf_params = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_c,
            lpf_params=carry["lpf_params"],
        )

        # Gets the log probabilities of the action.
        log_probs = actor_dist.log_prob(transition.action)
        assert isinstance(log_probs, Array)

        value, next_critic_c = self.run_critic(
            model=model.critic,
            observations=transition.obs,
            commands=transition.command,
            carry=critic_c,
        )

        # compute mirror losses
        mirrored_actor_dist, next_actor_c_m, _ = self.run_actor(
            model=model.actor,
            observations=self.mirror_obs(transition.obs),
            commands=self.mirror_cmd(transition.command),
            carry=actor_c_m,
            lpf_params=carry["lpf_params"],
        )
        double_mirrored_actor_dist = self.mirror_joints(mirrored_actor_dist.mean())
        action_mirror_loss = (
            jnp.mean((actor_dist.mean() - double_mirrored_actor_dist) ** 2) * self.config.actor_mirror_loss_scale
        )

        mirrored_value, next_critic_c_m = self.run_critic(
            model=model.critic,
            observations=self.mirror_obs(transition.obs),
            commands=self.mirror_cmd(transition.command),
            carry=critic_c_m,
        )
        value_mirror_loss = jnp.mean((value - mirrored_value) ** 2) * self.config.critic_mirror_loss_scale

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=jnp.expand_dims(log_probs, axis=0),
            values=value.squeeze(-1),
            entropy=jnp.expand_dims(actor_dist.entropy(), axis=0),
            action_std=actor_dist.stddev(),
            aux_losses={
                "action_mirror_loss": action_mirror_loss,
                "value_mirror_loss": value_mirror_loss,
            },
        )

        carry = {
            "actor": next_actor_c,
            "critic": next_critic_c,
            "actor_mirror": next_actor_c_m,
            "critic_mirror": next_critic_c_m,
            "lpf_params": next_lpf_params,
        }
        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(model, rng),
            carry,
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: Carry,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, Carry]:
        scan_fn = functools.partial(self._ppo_scan_fn, model=model)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=ksim.JitLevel.RL_CORE,
        )
        return ppo_variables, next_model_carry

    def get_initial_model_carry(
        self,
        model: Model,
        rng: PRNGKeyArray,
    ) -> Carry:
        return Carry(
            **{
                **{
                    name: tuple(
                        (jnp.zeros(shape=(self.config.hidden_size)), jnp.zeros(shape=(self.config.hidden_size)))
                        for _ in range(self.config.depth)
                    )
                    for name in ["actor", "actor_mirror", "critic", "critic_mirror"]
                },
                "lpf_params": ksim.LowPassFilterParams.initialize(len(JOINT_BIASES)),
            }
        )

    def sample_action(
        self,
        model: Model,
        model_carry: Carry,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
        curriculum_level: Array,
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        action_dist_j, actor_carry, next_lpf_params = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=model_carry["actor"],
            lpf_params=model_carry["lpf_params"],
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(
            action=action_j,
            carry={
                **model_carry,
                "actor": actor_carry,
                "lpf_params": next_lpf_params,
            },
        )

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

        # Negate every joint angle
        j_m = -j_m

        return j_m

    def mirror_obs(self, obs: xax.FrozenDict[str, Array]) -> xax.FrozenDict[str, Array]:
        # actor obs
        noisy_joint_pos_m = self.mirror_joints(obs["noisy_joint_position"])
        noisy_joint_vel_m = self.mirror_joints(obs["noisy_joint_velocity"])
        noisy_imu_gyro_m = jnp.concatenate(
            [
                -obs["noisy_imu_gyro"][..., 0:1],
                obs["noisy_imu_gyro"][..., 1:2],
                -obs["noisy_imu_gyro"][..., 2:3],
            ],
            axis=-1,
        )
        noisy_imu_projected_gravity_m = jnp.concatenate(
            [
                obs["noisy_imu_projected_gravity"][..., 0:1],
                -obs["noisy_imu_projected_gravity"][..., 1:2],
                obs["noisy_imu_projected_gravity"][..., 2:3],
            ],
            axis=-1,
        )

        # critic obs
        joint_pos_m = self.mirror_joints(obs["joint_position"])
        joint_vel_m = self.mirror_joints(obs["joint_velocity"])
        imu_gyro_m = jnp.concatenate(
            [
                -obs["imu_gyro"][..., 0:1],
                obs["imu_gyro"][..., 1:2],
                -obs["imu_gyro"][..., 2:3],
            ],
            axis=-1,
        )
        projected_gravity_m = jnp.concatenate(
            [
                obs["projected_gravity"][..., 0:1],
                -obs["projected_gravity"][..., 1:2],
                obs["projected_gravity"][..., 2:3],
            ],
            axis=-1,
        )

        left_touch_m = obs["right_foot_touch"]  # flip left and right
        right_touch_m = obs["left_foot_touch"]  # flip left and right
        feet_position_m = jnp.concatenate(
            [
                obs["feet_position"][..., 3:4],  # left x = right x
                -obs["feet_position"][..., 4:5],  # left y = -right y
                obs["feet_position"][..., 5:6],  # left z = right z
                obs["feet_position"][..., 0:1],  # right x = left x
                -obs["feet_position"][..., 1:2],  # right y = -left y
                obs["feet_position"][..., 2:3],  # right z = left z
            ],
            axis=-1,
        )
        base_position_m = obs["base_position"]  # no mirror because global frame
        base_orientation_m = jnp.concatenate(
            [
                obs["base_orientation"][..., 0:1],
                -obs["base_orientation"][..., 1:2],
                -obs["base_orientation"][..., 2:3],
                obs["base_orientation"][..., 3:4],
            ],
            axis=-1,
        )

        # NOTE very unsure about COM mirror correctness
        com_inertia_m = obs["center_of_mass_inertia"].reshape(-1, 10)
        # Fields (MuJoCo cinert): [m, m*cx, m*cy, m*cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        com_inertia_m = jnp.concatenate(
            [
                com_inertia_m[..., 0:1],  # m
                com_inertia_m[..., 1:2],  # m*cx
                -com_inertia_m[..., 2:3],  # m*cy -> flip
                com_inertia_m[..., 3:4],  # m*cz
                com_inertia_m[..., 4:5],  # Ixx
                com_inertia_m[..., 5:6],  # Iyy
                com_inertia_m[..., 6:7],  # Izz
                -com_inertia_m[..., 7:8],  # Ixy -> flip
                com_inertia_m[..., 8:9],  # Ixz
                -com_inertia_m[..., 9:10],  # Iyz -> flip
            ],
            axis=-1,
        ).reshape(obs["center_of_mass_inertia"].shape)

        # mirror COM velocity across sagittal plane
        # Assume per-body layout [vx, vy, vz, wx, wy, wz]
        cvel = obs["center_of_mass_velocity"].reshape(-1, 6)
        lin = cvel[..., 0:3]
        ang = cvel[..., 3:6]
        lin_m = jnp.concatenate(
            [
                lin[..., 0:1],  # vx
                -lin[..., 1:2],  # vy -> flip
                lin[..., 2:3],  # vz
            ],
            axis=-1,
        )
        ang_m = jnp.concatenate(
            [
                -ang[..., 0:1],  # wx -> flip (axial with det=-1)
                ang[..., 1:2],  # wy
                -ang[..., 2:3],  # wz -> flip (axial with det=-1)
            ],
            axis=-1,
        )
        cvel_m = jnp.concatenate([lin_m, ang_m], axis=-1).reshape(obs["center_of_mass_velocity"].shape)

        base_lin_vel_m = jnp.concatenate(
            [
                obs["base_linear_velocity"][..., 0:1],
                -obs["base_linear_velocity"][..., 1:2],
                obs["base_linear_velocity"][..., 2:3],
            ],
            axis=-1,
        )
        base_ang_vel_m = jnp.concatenate(
            [
                -obs["base_angular_velocity"][..., 0:1],
                obs["base_angular_velocity"][..., 1:2],
                -obs["base_angular_velocity"][..., 2:3],
            ],
            axis=-1,
        )

        actuator_force_m = self.mirror_joints(obs["actuator_force"])
        base_height_m = obs["base_height"]

        return xax.FrozenDict(
            {
                "noisy_joint_position": noisy_joint_pos_m,
                "noisy_joint_velocity": noisy_joint_vel_m,
                "noisy_imu_gyro": noisy_imu_gyro_m,
                "noisy_imu_projected_gravity": noisy_imu_projected_gravity_m,
                "joint_position": joint_pos_m,
                "joint_velocity": joint_vel_m,
                "imu_gyro": imu_gyro_m,
                "projected_gravity": projected_gravity_m,
                "left_foot_touch": left_touch_m,
                "right_foot_touch": right_touch_m,
                "feet_position": feet_position_m,
                "base_position": base_position_m,
                "base_orientation": base_orientation_m,
                "center_of_mass_inertia": com_inertia_m,
                "center_of_mass_velocity": cvel_m,
                "base_linear_velocity": base_lin_vel_m,
                "base_angular_velocity": base_ang_vel_m,
                "actuator_force": actuator_force_m,
                "base_height": base_height_m,
            }
        )

    def mirror_cmd(self, cmd: xax.FrozenDict[str, Array]) -> xax.FrozenDict[str, Array]:
        cmd_u = cmd["unified_command"]
        cmd_u_m = jnp.concatenate(
            [
                cmd_u[..., :1],  # x
                -cmd_u[..., 1:2],  # y
                -cmd_u[..., 2:3],  # wz
                cmd_u[..., 3:4],  # base height
                -cmd_u[..., 4:5],  # base roll
                cmd_u[..., 5:6],  # base pitch
                self.mirror_joints(
                    jnp.concatenate(
                        [
                            cmd_u[..., 6:16],
                            jnp.zeros(shape=(10,)),
                        ]
                    )
                )[..., :10],  # arms
            ],
            axis=-1,
        )
        return xax.FrozenDict({"unified_command": cmd_u_m})


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=4096,
            batch_size=512,
            num_passes=3,
            # epochs_per_log_step=1,
            rollout_length_seconds=2.0,
            # global_grad_clip=2.0,
            entropy_coef=0.002,
            learning_rate=5e-4,
            gamma=0.9,
            lam=0.94,
            actor_mirror_loss_scale=1.0,
            critic_mirror_loss_scale=0.01,
            hidden_size=256,
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
            # render_full_every_n_seconds=0,
            render_length_seconds=10,
            max_values_per_plot=50,
            # Checkpointing parameters.
            save_every_n_seconds=60,
            valid_every_n_steps=100,
            valid_every_n_seconds=None,
        ),
    )
