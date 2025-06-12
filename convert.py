"""Converts a checkpoint to a deployable model."""

import argparse
import math
from pathlib import Path
from typing import Callable, Literal, cast

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
import xax
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata
from mujoco_animator import MjAnim

from train import HumanoidWalkingTask, Model, rotate_quat_by_quat

NUM_COMMANDS_MODEL = 6 + 3 # 6 user commands + 3 arm commands

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
        w_part = rotating_quat[..., :1]  # w component
        xyz_part = -rotating_quat[..., 1:]  # negate x,y,z components
        rotating_quat = jnp.concatenate([w_part, xyz_part], axis=-1)

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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: HumanoidWalkingTask = HumanoidWalkingTask.load_task(ckpt_path)
    model: Model = task.load_ckpt(ckpt_path, part="model")[0]

    mujoco_model = task.get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]

    # Load arm animations
    arm_animations = Path(__file__).parent / "animations"
    dt = task.config.ctrl_dt
    interp = "linear"
    interp = cast(Literal["linear", "cubic"], interp)

    # Load all arm animations
    home_anim = MjAnim.load(arm_animations / "home.mjanim").to_numpy(dt, interp=interp)
    home_to_wave_anim = MjAnim.load(arm_animations / "home_to_wave.mjanim").to_numpy(dt, interp=interp)
    wave_anim = MjAnim.load(arm_animations / "wave.mjanim").to_numpy(dt, interp=interp)
    wave_to_home_anim = MjAnim.load(arm_animations / "wave_to_home.mjanim").to_numpy(dt, interp=interp)
    punch_anim = MjAnim.load(arm_animations / "punch.mjanim").to_numpy(dt, interp=interp)

    # Animation constants
    lengths = jnp.array(
        [
            home_anim.shape[0],
            home_to_wave_anim.shape[0],
            wave_anim.shape[0],
            wave_to_home_anim.shape[0],
            punch_anim.shape[0],
        ],
        dtype=jnp.int32,
    )

    starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(lengths)[:-1]])

    anims = jnp.concatenate(
        [
            home_anim[..., 7:17], # take arm joints only
            home_to_wave_anim[..., 7:17],
            wave_anim[..., 7:17],
            wave_to_home_anim[..., 7:17],
            punch_anim[..., 7:17],
        ],
        axis=0,
    )

    # Transition table: [current_cmd_id, desired_user_cmd] -> next_cmd_id
    # User commands: 0=home, 1=wave, 2=punch
    transition_table = jnp.array(
        [
            # desired: home wave punch
            [0, 1, 4],  # cur home (0)
            [3, 2, 3],  # cur home_to_wave (1) - must go through wave_to_home first
            [3, 2, 3],  # cur wave (2) - must go through wave_to_home first
            [0, 1, 4],  # cur wave_to_home (3) - can go anywhere from home
            [0, 1, 4],  # cur punch (4) - can go anywhere from home
        ],
        dtype=jnp.int32,
    )

    # Carry layout: [rnn_state(D*H), cmd_id(1), cmd_step(1)]
    rnn_size = task.config.depth * task.config.hidden_size
    cmd_state = 2
    carry_size = rnn_size + cmd_state

    def split_carry(flat_carry: Array) -> tuple[Array, Array, Array]:
        rnn = flat_carry[:rnn_size].reshape(task.config.depth, task.config.hidden_size)
        cmd_id = flat_carry[rnn_size : rnn_size + 1].round().astype(jnp.int32)
        cmd_step = flat_carry[rnn_size + 1 :].round().astype(jnp.int32)
        return rnn, cmd_id, cmd_step

    def merge_carry(rnn: Array, cmd_id: Array, cmd_step: Array) -> Array:
        return jnp.concatenate([rnn.reshape(-1), cmd_id.astype(jnp.float32), cmd_step.astype(jnp.float32)], axis=-1)

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=NUM_COMMANDS_MODEL,
        carry_size=(carry_size,),
    )

    @jax.jit
    def init_fn() -> Array:
        rnn = jnp.zeros((task.config.depth, task.config.hidden_size))
        cmd_id = jnp.zeros(1, dtype=jnp.int32)  # Start with home
        cmd_step = jnp.zeros(1, dtype=jnp.int32)
        return merge_carry(rnn, cmd_id, cmd_step)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        quaternion: Array,  # IMU quat
        initial_heading: Array,
        command: Array,
        gyroscope: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        # Unpack carry
        rnn, cmd_id, cmd_step = split_carry(carry)
        cmd_id = cmd_id.squeeze()
        cmd_step = cmd_step.squeeze()

        # Determine desired command from user input
        desired = jnp.argmax(command[..., 6:]).astype(jnp.int32)

        # Check if current animation is finished
        current_length = lengths[cmd_id]
        is_finished = cmd_step + 1 >= current_length

        # Update command state
        next_cmd_id = jnp.where(is_finished, transition_table[cmd_id, desired], cmd_id)
        next_cmd_step = jnp.where(is_finished, jnp.zeros_like(cmd_step), cmd_step + 1)

        # Get current animation frame
        anim_idx = starts[next_cmd_id] + next_cmd_step
        cmd_val = anims[anim_idx]  # Joint angles for current frame

        # Back-spin IMU quaternion by the (absolute) yaw command
        heading_yaw_cmd = command[..., 2]
        heading_yaw_cmd_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, heading_yaw_cmd]))

        initial_heading_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, initial_heading.squeeze()]))
        relative_quaternion = rotate_quat_by_quat(quaternion, initial_heading_quat, inverse=True)

        backspun_quat = rotate_quat_by_quat(relative_quaternion, heading_yaw_cmd_quat, inverse=True)
        positive_backspun_quat = jnp.where(backspun_quat[..., 0] < 0, -backspun_quat, backspun_quat)

        obs = jnp.concatenate(
            [
                joint_angles,
                joint_angular_velocities,
                positive_backspun_quat,
                command[..., :6],
                cmd_val,  # Current animation frame joint angles
                gyroscope,
            ],
            axis=-1,
        )

        dist, next_rnn = model.actor.forward(obs, rnn)
        action = dist.mode()

        # Pack updated carry
        next_carry = merge_carry(next_rnn, next_cmd_id[None], next_cmd_step[None])

        return action, next_carry

    init_onnx = export_fn(
        model=init_fn,
        metadata=metadata,
    )

    step_onnx = export_fn(
        model=step_fn,
        metadata=metadata,
    )

    kinfer_model = pack(
        init_fn=init_onnx,
        step_fn=step_onnx,
        metadata=metadata,
    )

    # Saves the resulting model.
    (output_path := Path(args.output_path)).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_model)


if __name__ == "__main__":
    main()
