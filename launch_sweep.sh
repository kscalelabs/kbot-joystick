#!/bin/bash

# Check if number of GPUs argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Number of GPUs argument is required"
    echo "Usage: $0 NUM_GPUS"
    echo "Example: $0 8"
    exit 1
fi

NUM_GPUS=$1

# Activate conda environment
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate ksim

# Prevent JAX from grabbing all GPUs in the config generation step
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

# First generate all configs
python create_sweep.py

# Launch workers in parallel, each with its own GPU environment
for i in $(seq 0 $(($NUM_GPUS-1)))
do
    echo "Launching worker for GPU $i"
    # Start Xvfb with display number 10$i
    Xvfb :10$i -ac &
    # Set up clean environment for each worker
    env -i \
        HOME=$HOME \
        PATH=$PATH \
        PYTHONPATH=$PYTHONPATH \
        CUDA_VISIBLE_DEVICES=$i \
        DISPLAY=:10$i.0 \
        MUJOCO_GL=egl \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
        python run_worker.py $i &
done

# Wait for all workers to complete
wait
echo "All workers completed"
