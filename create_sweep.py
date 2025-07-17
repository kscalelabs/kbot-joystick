"""Script to create a list of command line args for hyperparameter sweep."""

import itertools
import json
from pathlib import Path
import random

MAX_STEPS = 10000

# Define the hyperparameters to sweep over
SWEEP_PARAMS = {
    "learning_rate": [1e-4, 3e-4, 5e-4, 6e-4, 7e-4], # 5e-4
    "gamma": [0.90, 0.93, 0.95, 0.96], # 0.9
    "lam": [0.92, 0.93, 0.94, 0.95], # 0.94
}

def get_all_commands():
    """Generate all possible combinations of command line arguments."""
    # Get all combinations of sweep parameters
    keys = SWEEP_PARAMS.keys()
    values = SWEEP_PARAMS.values()
    commands = []
    
    for items in itertools.product(*values):
        # Create dict of current parameter combination
        param_dict = dict(zip(keys, items))

        # Add exp_dir
        exp_dir = f"sweep_runs/logs/sweep__" + "_".join(f"{k}{v}" for k, v in param_dict.items())
        param_dict["exp_dir"] = exp_dir

        # Add max_steps
        param_dict["max_steps"] = MAX_STEPS
        param_dict["disable_multiprocessing"] = True
        param_dict["render_full_every_n_seconds"] = 9999999999

        # Convert to command line args
        cmd_args = " ".join([f"{k}={v}" for k, v in param_dict.items()])
        cmd = f"python -m train {cmd_args}"
        commands.append(cmd)

    # randomize order so results will be useful even after early stopping
    random.shuffle(commands)
    
    return commands

def main():
    """Create the sweep command file."""
    commands = get_all_commands()
    print(f"Generated {len(commands)} commands")
    
    # Create sweep directory
    sweep_dir = Path("sweep_runs")
    sweep_dir.mkdir(exist_ok=True)
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Save commands to file, one per line
    command_file = sweep_dir / "pending_commands.txt"
    with open(command_file, "w") as f:
        for cmd in commands:
            f.write(f"{cmd}\n")
    
    print(f"Saved commands to {command_file}")
    print("To start the sweep, run:")
    print("  python run_worker.py 0  # For GPU 0")
    print("  python run_worker.py 1  # For GPU 1")
    print("  etc...")

if __name__ == "__main__":
    main() 