"""Worker script that processes commands one at a time from a shared command file."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

def get_next_command(command_file: Path) -> str | None:
    """Get next command from file with file locking to prevent race conditions."""
    if not command_file.exists():
        return None
        
    try:
        with open(command_file, "r+") as f:
            # Get exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            try:
                # Read all lines
                lines = f.readlines()
                if not lines:  # Empty file
                    return None
                    
                # Get first command and remove it
                next_command = lines[0].strip()
                
                # Write remaining lines back
                f.seek(0)
                f.writelines(lines[1:])
                f.truncate()
                
                return next_command
                
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
    except IOError as e:
        print(f"Error reading command file: {e}")
        return None

def run_command(command: str, gpu_id: int):
    """Run a single command."""
    try:
        print(f"Starting command on GPU {gpu_id}:")
        print(command)
        
        # Run the command
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            env=os.environ.copy(),  # Use current environment with GPU settings
        )
        
        print(f"Command completed successfully on GPU {gpu_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed on GPU {gpu_id} with error: {e}")
        # Save failed command to separate file
        failed_file = Path("sweep_runs") / "failed_commands.txt"
        with open(failed_file, "a") as f:
            f.write(f"{command}\n")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu_id", type=int, help="GPU ID to use")
    args = parser.parse_args()
    
    command_file = Path("sweep_runs") / "pending_commands.txt"
    
    while True:
        command = get_next_command(command_file)
        if command is None:
            print(f"No more commands to process on GPU {args.gpu_id}")
            break
            
        run_command(command, args.gpu_id)
        
        # Small delay between runs
        time.sleep(1)

if __name__ == "__main__":
    main() 