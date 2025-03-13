#!/bin/bash
set -e

# Get the parent parent directory of the current script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"

# Get conda env name from the first argument, if not provided, use "isaac-sim"
CONDA_ENV_NAME=${1:-robotic_surgery}

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


# ---- IsaacLab: Create a new conda env and install IsaacLab ----
echo "Installing IsaacLab..."
# CLONING REPOSITORIES INTO PROJECT_ROOT/third_party
echo "Cloning repositories into $PROJECT_ROOT/third_party..."
mkdir $PROJECT_ROOT/third_party
git clone -b v1.2.0 git@github.com:isaac-sim/IsaacLab.git $PROJECT_ROOT/third_party/IsaacLab
pushd $PROJECT_ROOT/third_party/IsaacLab
sed -i "s/rsl-rl/rsl-rl-lib/g" source/extensions/omni.isaac.lab_tasks/setup.py
./isaaclab.sh --conda $CONDA_ENV_NAME
eval "$(conda shell.bash hook)"  # required to activate conda env
conda activate $CONDA_ENV_NAME

# ---- Install IsaacSim and necessary dependencies ----
echo "Installing IsaacSim..."
pip install isaacsim==4.1.0.0 isaacsim-extscache-physics==4.1.0.0 \
    isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 \
    git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git \
    --extra-index-url https://pypi.nvidia.com

echo "Installing IsaacLab ..."
yes Yes | ./isaaclab.sh --install
popd



