#!/bin/bash

# Default values
S3_BUCKET=${S3_BUCKET:-""}  # Use environment variable if set

# Help message
show_help() {
    echo "Usage: $0 COMMAND [args]"
    echo
    echo "Commands:"
    echo "  install              Install required dependencies and setup conda environment"
    echo "  launch NUM_GPUS SWEEP_ID    Launch sweep with specified number of GPUs and sweep ID"
    echo "  kill                 Kill all sweep-related processes"
    echo "  rm                   Remove all sweep logs and data"
    echo "  status              Show status of running processes"
    echo "  sync                Enable S3 sync (requires S3_BUCKET env var)"
    echo "  help                Show this help message"
    echo
    echo "Examples:"
    echo "  $0 install          # First time setup"
    echo "  $0 launch 4 1       # Launch sweep with 4 GPUs and sweep ID 1"
    echo "  $0 kill            # Kill all processes"
}

# Install dependencies and setup environment
install() {
    echo "Installing dependencies and setting up environment..."
    
    # Download Miniconda installer if not exists
    if [ ! -f miniconda.sh ]; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    fi

    # Install Miniconda if not already installed
    if [ ! -d "$HOME/miniconda" ]; then
        bash miniconda.sh -b -p $HOME/miniconda
    fi

    # Initialize conda for bash
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"

    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q "ksim"; then
        conda create --name ksim --file conda-spec-file.txt python=3.11 -y
    fi
    
    conda activate ksim
    pip install -r requirements.txt

    # Install system dependencies
    sudo apt update
    sudo apt install -y xvfb libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev libegl1-mesa-dev awscli

    echo "Environment setup complete!"
    echo "To activate the environment manually, run: conda activate ksim"
}

# Launch sweep
launch() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Error: Both NUM_GPUS and SWEEP_ID are required"
        echo "Usage: $0 launch NUM_GPUS SWEEP_ID"
        echo "Example: $0 launch 4 1"
        exit 1
    fi

    NUM_GPUS=$1
    SWEEP_ID=$2
    
    echo "Launching sweep $SWEEP_ID with $NUM_GPUS GPUs..."
    
    # Activate conda environment
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda activate ksim

    # Create logs directory
    mkdir -p logs
    mkdir -p sweep_runs/logs

    # Generate sweep configs
    export CUDA_VISIBLE_DEVICES=""
    export XLA_PYTHON_CLIENT_PREALLOCATE="false"
    python create_sweep.py

    # Launch workers
    for i in $(seq 0 $(($NUM_GPUS-1))); do
        echo "Launching worker for GPU $i"
        # Start Xvfb
        nohup Xvfb :10$i -ac > logs/xvfb_$i.log 2>&1 &
        
        # Launch worker with clean environment
        nohup env -i \
            HOME=$HOME \
            PATH=$PATH \
            PYTHONPATH=$PYTHONPATH \
            CUDA_VISIBLE_DEVICES=$i \
            DISPLAY=:10$i.0 \
            MUJOCO_GL=egl \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
            python run_worker.py $i > logs/worker_$i.log 2>&1 &
        
        sleep 10
    done

    # Start S3 sync if bucket is configured
    if [ -n "$S3_BUCKET" ]; then
        enable_sync "$SWEEP_ID"
    fi

    echo "All workers launched. Check logs/ directory for output."
    echo "Use '$0 status' to check running processes."
}

# Kill all sweep processes
kill_sweep() {
    echo "Killing all sweep-related processes..."
    
    # Kill Python processes using GPUs
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9
    
    # Kill worker processes
    pkill -9 -f run_worker.py
    
    # Kill Xvfb processes
    pkill -9 Xvfb
    
    # Kill S3 sync processes
    pkill -9 -f "aws s3 sync"
    
    echo "All sweep processes killed."
}

# Remove sweep data
remove_data() {
    echo "Removing sweep data..."
    rm -rf sweep_runs/*
    rm -rf logs/*
    echo "Sweep data removed."
}

# Show status
status() {
    echo "=== Worker Processes ==="
    ps aux | grep -E "run_worker.py|[p]ython -m train" | grep -v grep
    
    echo -e "\n=== GPU Usage ==="
    nvidia-smi
    
    echo -e "\n=== Log Files ==="
    ls -l logs/
}

# Enable S3 sync
enable_sync() {
    if [ -z "$S3_BUCKET" ]; then
        echo "Error: S3_BUCKET environment variable not set"
        echo "Usage: S3_BUCKET=your-bucket $0 sync"
        exit 1
    fi

    local sweep_id=${1:-$SWEEP_ID}
    exp_dir="sweep_$sweep_id"
    nohup bash -c '
        while true; do
            sleep 600
            aws s3 sync "sweep_runs/" "s3://$S3_BUCKET/kbot-joystick/$exp_dir" \
                --delete \
                --only-show-errors \
                --exclude "*/.nfs*" \
                --exclude "*/tmp/*"
            echo "Synced data to S3 at $(date)"
        done' > logs/aws_sync.log 2>&1 &
    
    echo "S3 sync enabled. Check logs/aws_sync.log for details."
}

# Main command router
case "$1" in
    "install")
        install
        ;;
    "launch")
        launch "$2" "$3"
        ;;
    "kill")
        kill_sweep
        ;;
    "rm")
        remove_data
        ;;
    "status")
        status
        ;;
    "sync")
        enable_sync
        ;;
    "help"|"--help"|"-h"|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage information."
        exit 1
        ;;
esac 