"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import HumanoidWalkingTask, Model

NUM_COMMANDS_MODEL = 16


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

    # Loads the Mujoco model and gets the joint names.
    mujoco_model = task.get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Removes the root joint.

    # Constant values.
    carry_shape = (task.config.depth + 1, task.config.hidden_size)  # +1 to hack in a tensor for heading carry
    num_commands = NUM_COMMANDS_MODEL

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=num_commands,
        carry_size=carry_shape,
    )

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        projected_gravity: Array,
        gyroscope: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        heading_carry = carry[0]
        model_carry = carry[1:]

        cmd_zero = (jnp.linalg.norm(command[..., :3], axis=-1) < 1e-3)[..., None]
        cmd_vel = command[..., :2]
        cmd_yaw_rate = command[..., 2:3]
        cmd_body_height = command[..., 3:4]
        cmd_body_orientation = command[..., 4:6]
        cmd_arms = command[..., 6:]


        obs = jnp.concatenate(
            [
                joint_angles,
                joint_angular_velocities,
                projected_gravity,
                cmd_zero,
                cmd_vel,
                cmd_yaw_rate,
                cmd_body_height,
                cmd_body_orientation,
                cmd_arms,
                gyroscope,
            ],
            axis=-1,
        )
        dist, model_carry = model.actor.forward(obs, model_carry)
        carry = jnp.concatenate([heading_carry[None, :], model_carry], axis=0)
        return dist.mode(), carry

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
