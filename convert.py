"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import HumanoidWalkingTask, Model

NUM_COMMANDS_MODEL = 16


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
    carry_shape = (task.config.depth, 2, task.config.hidden_size)  # TODO carry broken for gru
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
        dist, carry = model.actor.forward(obs, carry)
        # Convert carry tuple of tuples into a single Array
        carry_array = jnp.stack([jnp.stack(c) for c in carry])
        return dist.mode(), carry_array

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
