![Isaac for Healthcare - Robotic Surgery](media/teaser.png)

---

# Isaac for Healthcare - Robotic Surgery

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.2.0-silver)](https://isaac-sim.github.io/IsaacLab/v1.2.0/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)


## Overview

ORBIT-Surgical is a physics-based surgical robot simulation framework with photorealistic rendering in NVIDIA Omniverse. ORBIT-Surgical leverages GPU parallelization to train reinforcement learning and imitation learning algorithms to facilitate study of robot learning to augment human surgical skills. ORBIT-Surgical is built upon [NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) to leverage the latest simulation capabilities for photo-realistic scenes and fast and accurate simulation.


## Setup

### Basic Setup

ORBIT-Surgical is built upon NVIDIA [Isaac Sim 4.1.0](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/index.html) and [Isaac Lab 1.2.0](https://github.com/isaac-sim/IsaacLab).

For detailed instructions on how to install these dependencies, please refer to the [Isaac Sim installation guide](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/index.html) and [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v1.2.0/source/setup/installation/index.html).

We recommend using `conda` to create a new environment and install all the dependencies:

#### 1. Prepare Isaac Lab Repository and Create a Conda Environment

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
${IsaacLab_PATH}/isaaclab.sh --conda orbitsurgical
```

#### 2. Install Isaac Sim

```bash
# Activate conda environment
conda activate orbitsurgical

pip install isaacsim==4.1.0.0 isaacsim-extscache-physics==4.1.0.0 isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 --extra-index-url https://pypi.nvidia.com
```

#### 3. Install ORBIT-Surgical Extensions

```bash
# move to a directory outside of IsaacLab installation directory
cd ..
git clone https://github.com/orbit-surgical/orbit-surgical
cd orbit-surgical
pip install toml
# Install all ORBIT-Surgical extensions
./orbitsurgical.sh
```

#### 4. Install Isaac Lab Extensions

```bash
${IsaacLab_PATH}/isaaclab.sh --install
```

Please note that once you are in the virtual environment, you do not need to use `${IsaacLab_PATH}/isaaclab.sh -p` to run python scripts. You can use the default python executable in your environment by running `python` or `python3`. However, for the rest of the documentation, we will assume that you are using `${IsaacLab_PATH}/isaaclab.sh -p` to run python scripts.

## Acknowledgement

### License

This source code is ported from the [ORBIT-Surgical](https://orbit-surgical.github.io/) framework released under [BSD-3-Clause License](https://github.com/orbit-surgical/orbit-surgical/blob/main/LICENCE).

NVIDIA Isaac Sim is available freely under [individual license](https://www.nvidia.com/en-us/omniverse/download/). For more information about its license terms, please check [here](https://docs.omniverse.nvidia.com/install-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html).

Isaac Lab is released under [BSD-3-Clause License](https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE).

### Citing

The development of Isaac for Healthcare - Robotic Surgery was initiated from the [ORBIT-Surgical](https://orbit-surgical.github.io/) framework. We would appreciate if you would cite [this paper](https://arxiv.org/abs/2404.16027) in academic publications as well:

```text
@article{yu2024orbit,
  title={ORBIT-Surgical: An Open-Simulation Framework for Learning Surgical Augmented Dexterity},
  author={Yu, Qinxi and Moghani, Masoud and Dharmarajan, Karthik and Schorp, Vincent and Panitch, William Chung-Ho and Liu, Jingzhou and Hari, Kush and Huang, Huang and Mittal, Mayank and Goldberg, Ken and others},
  journal={arXiv preprint arXiv:2404.16027},
  year={2024}
}
```
