"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata
import jax.numpy as jnp

from train import HumanoidWalkingTask, Model

NUM_COMMANDS_MODEL = 6  # Updated to match unified command structure

def euler_to_quat(euler_3: Array) -> Array:
    """Converts roll, pitch, yaw angles to a quaternion (w, x, y, z).

    Args:
        euler_3: The roll, pitch, yaw angles, shape (*, 3).

    Returns:
        The quaternion with shape (*, 4).
    """
    # Extract roll, pitch, yaw from input
    roll, pitch, yaw = jnp.split(euler_3, 3, axis=-1)

    # Calculate trigonometric functions for each angle
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    # Calculate quaternion components using the conversion formula
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Combine into quaternion [w, x, y, z]
    quat = jnp.concatenate([w, x, y, z], axis=-1)

    # Normalize the quaternion
    quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)

    return quat

def rotate_quat(q1: Array, q2: Array) -> Array:
    """Rotate quaternion q1 by quaternion q2.
    
    Args:
        q1: quaternion to be rotated [w, x, y, z]
        q2: quaternion to rotate by [w, x, y, z]
    
    Returns:
        rotated quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])

def quat_to_euler(quat_4: Array, eps: float = 1e-6) -> Array:
    """Normalizes and converts a quaternion (w, x, y, z) to roll, pitch, yaw.

    Args:
        quat_4: The quaternion to convert, shape (*, 4).
        eps: A small epsilon value to avoid division by zero.

    Returns:
        The roll, pitch, yaw angles with shape (*, 3).
    """
    quat_4 = quat_4 / (jnp.linalg.norm(quat_4, axis=-1, keepdims=True) + eps)
    w, x, y, z = jnp.split(quat_4, 4, axis=-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)

    # Handle edge cases where |sinp| >= 1
    pitch = jnp.where(
        jnp.abs(sinp) >= 1.0,
        jnp.sign(sinp) * jnp.pi / 2.0,  # Use 90 degrees if out of range
        jnp.arcsin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.concatenate([roll, pitch, yaw], axis=-1)

def make_export_model(model: Model) -> Callable:
    def model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
        dist, carry = model.actor.forward(obs, carry)
        return dist.mode(), carry

    def batched_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
        return jax.vmap(model_fn)(obs, carry)

    return batched_model_fn


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
    carry_shape = (task.config.depth, task.config.hidden_size)  # (3, 128) hiddens
    # num_joints = len(joint_names)  # 20, joints outputs
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
        projected_gravity: Array,
        command: Array,
        quaternion: Array,
        initial_heading: Array,
        gyroscope: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        # Match the exact observation structure from training run_actor method
        # cmd format: [vx, vy, yaw, base_height, rx, ry]
        cmd_velocity = command[:3]  # vx, vy, yaw
        cmd_base_height = jnp.zeros_like(command[3:4])  # disabled base height (set to 0)
        cmd_orientation = command[4:]  # rx, ry

        # convert initial heading to quaternion
        initial_heading_quaternion = euler_to_quat(jnp.array([0.0, 0.0, -initial_heading.squeeze()]))

        # rotate quaternion by initial heading quaternion
        base_yaw_quaternion = rotate_quat(quaternion, initial_heading_quaternion)

        obs = jnp.concatenate(
            [
                joint_angles,                    # NUM_JOINTS (20)
                joint_angular_velocities,        # NUM_JOINTS (20)
                projected_gravity,               # 3
                cmd_velocity,                    # 3 (vx, vy, yaw)
                cmd_base_height,                 # 1 (disabled, zeros)
                cmd_orientation,                 # 2 (rx, ry)
                base_yaw_quaternion,             # 4 (w, x, y, z)
                gyroscope,                       # 3
            ],
            axis=-1,
        )
        dist, carry = model.actor.forward(obs, carry)
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
