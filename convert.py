"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import HumanoidWalkingTask, Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument(
        "--num_commands",
        type=int,
        required=True,
        choices=[3, 16],
        help="Number of commands (3 or 16)",
    )
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: HumanoidWalkingTask = HumanoidWalkingTask.load_task(ckpt_path)
    mujoco_model = task.get_mujoco_model()
    init_params = ksim.InitParams(key=jax.random.PRNGKey(0), physics_model=mujoco_model)
    model = cast(Model, task.load_ckpt(ckpt_path, init_params=init_params, part="model")[0])

    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Removes the root joint.

    # Constant values.
    carry_shape = (task.config.depth, 2, task.config.hidden_size)
    num_commands = args.num_commands

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
        # pad command to 16 regardless of num_commands
        cmd = jnp.pad(command, (0, 16 - command.shape[-1]), mode="constant", constant_values=0)[..., :16]

        cmd_zero = (jnp.linalg.norm(cmd[..., :3], axis=-1) < 1e-3)[..., None]

        obs = jnp.concatenate(
            [
                joint_angles,
                joint_angular_velocities,
                projected_gravity,
                gyroscope,
                cmd_zero,
                cmd,
            ],
            axis=-1,
        )
        dist, carry = model.actor.forward(obs, carry)
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
