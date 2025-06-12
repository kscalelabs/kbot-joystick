#!/bin/bash

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Install Miniconda
bash miniconda.sh -b -p $HOME/miniconda

# Initialize conda for bash
eval "$($HOME/miniconda/bin/conda shell.bash hook)"

conda create --name ksim --file conda-spec-file.txt python=3.11
conda activate ksim
pip install -r requirements.txt


echo "Environment setup complete! To activate the environment, run:"
echo "conda activate ksim"
