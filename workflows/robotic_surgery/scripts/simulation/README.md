# Isaac for Healthcare - Robotic Surgery


## Overview

Robotic Surgery is a physics-based surgical robot simulation framework with photorealistic rendering in NVIDIA Omniverse. Robotic Surgery leverages GPU parallelization to train reinforcement learning and imitation learning algorithms to facilitate study of robot learning to augment human surgical skills.


## Setup

Robotic Surgery is built upon NVIDIA [Isaac Sim 4.1.0](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/index.html) and [Isaac Lab 1.2.0](https://github.com/isaac-sim/IsaacLab).

For detailed instructions on how to install these dependencies, please refer to the [Isaac Sim installation guide](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/index.html) and [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v1.2.0/source/setup/installation/index.html).

We recommend using `conda` to create a new environment and install all the dependencies:

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

### 2. Install Isaac Sim

```bash
pip install isaacsim==4.1.0.0 isaacsim-extscache-physics==4.1.0.0 isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 --extra-index-url https://pypi.nvidia.com
```

### 3. Install Isaac Lab

```bash
${IsaacLab_PATH}/isaaclab.sh --install
```

### 4. Install Robotic Surgery

```bash
# move to a directory outside of IsaacLab installation directory
cd ..
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows
./workflows/robotic_surgery/scripts/simulation/robotic_surgery.sh
export PYTHONPATH=`pwd`
```

Please note that once you are in the virtual environment, you do not need to use `${IsaacLab_PATH}/isaaclab.sh -p` to run python scripts. You can use the default python executable in your environment by running `python` or `python3`. However, for the rest of the documentation, we will assume that you are using `${IsaacLab_PATH}/isaaclab.sh -p` to run python scripts.
