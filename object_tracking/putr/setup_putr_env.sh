#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "INFO: Starting Conda environment setup for PuTR..."

# 1. Set Conda package cache directory
export CONDA_PKGS_DIRS=/home/jupyter/conda_cache
echo "INFO: CONDA_PKGS_DIRS set to /home/jupyter/conda_cache"

# 2. Create the Conda environment
# Using --prefix to specify the location. Added -y for non-interactive mode.
echo "INFO: Creating Conda environment 'PuTR' at /home/jupyter/PuTR with Python 3.10..."
conda create --prefix /home/jupyter/PuTR python=3.10 -y
echo "INFO: Conda environment 'PuTR' created."

# 3. Activate the environment
# To make 'conda activate' work in a script, we often need to source conda.sh.
# Replace '~/miniconda3' with your actual Conda installation path if different (e.g., ~/anaconda3).
CONDA_BASE_PATH=$(conda info --base)
if [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ]; then
    echo "INFO: Sourcing conda.sh from ${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
    source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found. Please ensure Conda is initialized correctly for shell scripts."
    echo "You might need to run 'conda init bash' and restart your shell, or adjust the path to conda.sh."
    exit 1
fi

echo "INFO: Activating Conda environment '/home/jupyter/PuTR'..."
conda activate /home/jupyter/PuTR
echo "INFO: Conda environment activated. Current environment:"
conda info | grep "active environment"

# 4. Install PyTorch, torchvision, torchaudio, and specific CUDA version
# The PyTorch version must be greater than 2.0 (this command should handle it if available for cuda 11.8)
echo "INFO: Installing PyTorch, torchvision, torchaudio with CUDA 11.8..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
echo "INFO: PyTorch packages installed."

# 5. Install other Conda packages
echo "INFO: Installing matplotlib, pyyaml, scipy, tqdm, tensorboard..."
conda install matplotlib pyyaml scipy tqdm tensorboard -y
echo "INFO: Additional Conda packages installed."

# 6. Install pip packages
# This will use the pip from the activated 'PuTR' environment.
echo "INFO: Installing opencv-python and lap using pip..."
pip install opencv-python lap supervision
echo "INFO: Pip packages installed."

# 7. Rest
# Create putr folder and clone repository
echo "Creating putr directory..."
mkdir -p /home/jupyter/putr # Using absolute path for clarity, adjust if needed
cd /home/jupyter/putr
echo "Navigated to $(pwd)"
echo "Cloning PuTR repository..."
git clone https://github.com/chongweiliu/PuTR.git
cd PuTR
echo "Navigated to $(pwd)"

echo "Script finished successfully. The PuTR environment is active in this shell if you source the script."
echo "To use the environment in a new shell, run: conda activate /home/jupyter/PuTR"
echo "You are currently in the PuTR repository directory: $(pwd)"
