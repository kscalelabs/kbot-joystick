#!/bin/bash

# Prevent JAX from grabbing all GPUs in the config generation step
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

# First generate all configs
python create_sweep.py

# Launch workers in parallel, each with its own GPU environment
for i in {0..7}
do
    echo "Launching worker for GPU $i"
    # Set up clean environment for each worker
    env -i \
        HOME=$HOME \
        PATH=$PATH \
        PYTHONPATH=$PYTHONPATH \
        CUDA_VISIBLE_DEVICES=$i \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
        python run_worker.py $i &
done

# Wait for all workers to complete
wait
echo "All workers completed" 