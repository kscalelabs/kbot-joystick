#!/bin/bash

# Check if number of GPUs argument is provided
if [ $# -ne 2 ]; then
    echo "Error: Number of GPUs and sweep ID arguments are required"
    echo "Usage: $0 NUM_GPUS SWEEP_ID"
    echo "Example: $0 8 1"
    exit 1
fi

NUM_GPUS=$1
SWEEP_ID=$2

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate ksim

# Prevent JAX from grabbing all GPUs in the config generation step
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

# First generate all configs
python create_sweep.py

# Set up S3 sync in background
exp_dir="sweep_$SWEEP_ID"
if [ -z "$S3_BUCKET" ]; then
    echo "Warning: S3_BUCKET environment variable not set, skipping S3 sync"
else
    (while true; do
        sleep 600
        aws s3 sync "xxx" "s3://$S3_BUCKET/kbot-joystick/$exp_dir" \
            --delete \
            --only-show-errors \
            --exclude "*/.nfs*" \
            --exclude "*/tmp/*"
        echo "Synced data to S3 at $(date)"
    done) &
fi


# Launch workers in parallel, each with its own GPU environment
for i in $(seq 0 $(($NUM_GPUS-1)))
do
    echo "Launching worker for GPU $i"
    # Start Xvfb with display number 10$i
    nohup Xvfb :10$i -ac > logs/xvfb_$i.log 2>&1 &
    
    # Set up clean environment for each worker
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

echo "All workers launched in background. Check logs/ directory for output."
echo "Use 'ps aux | grep run_worker.py' to check running processes."
