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

from train import HumanoidWalkingTask, Model

GAIT_FREQ = jnp.array([1.2])
CTRL_DT = 0.02


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
    carry_shape = (task.config.depth, task.config.hidden_size)
    num_joints = len(joint_names)
    num_commands = 7  # joystick command

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        projected_gravity: Array,
        accelerometer: Array,
        gyroscope: Array,
        time: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        steps = time / CTRL_DT
        phase_dt = 2 * jnp.pi * GAIT_FREQ * CTRL_DT
        start_phase = jnp.array([0, jnp.pi])
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        # Stand still case
        joystick_cmd = command
        is_stand_still_command = joystick_cmd[..., 0] == 1.0
        phase = jnp.where(
            is_stand_still_command,
            jnp.array([jnp.pi / 2, jnp.pi]),
            phase,
        )
        timestep_phase_4 = jnp.array([jnp.cos(phase), jnp.sin(phase)]).flatten()

        # Check if the "stand still" command (index 0 of one-hot encoded vector) is active.
        obs = jnp.concatenate(
            [
                timestep_phase_4,
                joint_angles,
                joint_angular_velocities,
                projected_gravity,
                command,
                GAIT_FREQ,
                accelerometer,
                gyroscope,
            ],
            axis=-1,
        )
        dist, carry = model.actor.forward(obs, carry)
        return dist.mode(), carry

    init_onnx = export_fn(
        model=init_fn,
        num_joints=num_joints,
        num_commands=num_commands,
        carry_shape=carry_shape,
    )

    step_onnx = export_fn(
        model=step_fn,
        num_joints=num_joints,
        num_commands=num_commands,
        carry_shape=carry_shape,
    )

    kinfer_model = pack(
        init_fn=init_onnx,
        step_fn=step_onnx,
        joint_names=joint_names,
        carry_shape=carry_shape,
        num_commands=num_commands,
    )

    # Saves the resulting model.
    (output_path := Path(args.output_path)).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_model)


if __name__ == "__main__":
    main()
