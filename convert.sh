#!/bin/bash

for arg in "$@"; do
  python -m convert "humanoid_walking_task/run_${arg}/checkpoints/ckpt.bin" "humanoid_walking_task/${arg}.kinfer"
done
