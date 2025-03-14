#!/bin/bash
set -e

# Get the parent parent directory of the current script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"

# Check if running in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: No active conda environment detected"
    echo "Please activate a conda environment before running this script"
    exit 1
fi
echo "Using conda environment: $CONDA_DEFAULT_ENV"

# Check if NVIDIA GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not found or driver not installed"
    exit 1
fi

# Check if the third_party directory exists, if yes, then exit
if [ -d "$PROJECT_ROOT/third_party" ]; then
    echo "Error: third_party directory already exists"
    echo "Please remove the third_party directory before running this script"
    exit 1
fi


# ---- Clone IsaacLab ----
echo "Installing IsaacLab..."
# CLONING REPOSITORIES INTO PROJECT_ROOT/third_party
echo "Cloning repositories into $PROJECT_ROOT/third_party..."
mkdir $PROJECT_ROOT/third_party
git clone -b v1.2.0 git@github.com:isaac-sim/IsaacLab.git $PROJECT_ROOT/third_party/IsaacLab
pushd $PROJECT_ROOT/third_party/IsaacLab
sed -i "s/rsl-rl/rsl-rl-lib/g" source/extensions/omni.isaac.lab_tasks/setup.py


# ---- Install IsaacSim and necessary dependencies ----
echo "Installing IsaacSim..."
pip install isaacsim==4.1.0.0 isaacsim-extscache-physics==4.1.0.0 \
    isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 \
    git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git \
    --extra-index-url https://pypi.nvidia.com


echo "Installing IsaacLab ..."
yes Yes | ./isaaclab.sh --install
popd


# ---- Install robotic.surgery.assets and robotic.surgery.tasks ----
echo "Installing robotic.surgery.assets and robotic.surgery.tasks..."
pip install -e $PROJECT_ROOT/workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.assets
pip install -e $PROJECT_ROOT/workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks

echo "Dependencies installed successfully!"
