# Scripts for all robotic surgery simulations

## System Requirements

- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
    - GPUs without RT Cores (A100, H100) are not supported.
- NVIDIA Driver Version >= 555
- 50GB of disk space

## Environment Setup

The robotic surgery workflow is built on the following dependencies:
- [IsaacSim 4.1.0.0](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/index.html
- [IsaacLab 1.2.0](https://isaac-sim.github.io/IsaacLab/v1.2.0/source/setup/installation/index.html)


### 1. Prepare Isaac Lab Repository and Create a Conda Environment

Clone [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab):

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v1.2.0
sed -i 's/rsl-rl/rsl-rl-lib/g' source/extensions/omni.isaac.lab_tasks/setup.py
# define the IsaacLab_PATH environment variable
export IsaacLab_PATH=`pwd`
```

Create a new conda environment via Isaac Lab script:

```bash
${IsaacLab_PATH}/isaaclab.sh --conda robotic_surgery
# activate conda environment
conda activate robotic_surgery
```

### 2. Install Isaac Sim and Isaac Lab

```bash
pip install isaacsim==4.1.0.0 isaacsim-extscache-physics==4.1.0.0 isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 --extra-index-url https://pypi.nvidia.com
${IsaacLab_PATH}/isaaclab.sh --install
```

### 3. Install Robotic Surgery

```bash
# move to a directory outside of IsaacLab installation directory
cd ..
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows
./workflows/robotic_surgery/scripts/simulation/robotic_surgery.sh
export PYTHONPATH="$(pwd)/workflows/robotic_surgery/scripts"
```

### Download the I4H assets

```sh
i4h-asset-retrieve
```

## Run the scripts

Navigate to these sub-directories and run the scripts.

- [Simulation](./simulation)
