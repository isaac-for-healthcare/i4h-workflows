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

ORBIT-Surgical is built upon NVIDIA [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) and [Isaac Lab](https://github.com/isaac-sim/IsaacLab). For detailed instructions on how to install these dependencies, please refer to the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v1.2.0/source/setup/installation/index.html).

Please follow the Isaac Sim [documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) to install the Isaac Sim 4.1.0 release.

Clone [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab):

```bash
git clone https://github.com/isaac-sim/IsaacLab.git

cd IsaacLab

# Checkout v1.2.0
git checkout v1.2.0
```

Clone [ORBIT-Surgical repository](https://github.com/orbit-surgical/orbit-surgical) to a directory **outside** the Isaac Lab installation directory:

```bash
git clone https://github.com/orbit-surgical/orbit-surgical

cd <repo root>/workflows/robotic_surgery/scripts/simulation
```

Define the following environment variable to specify the path to your Isaac Lab installation directory:

```bash
# Set the IsaacLab_PATH environment variable to point to your Isaac Lab installation directory
export IsaacLab_PATH=<your_IsaacLab_PATH>
```

Set up a symbolic link between the installed Isaac Sim root folder and `_isaac_sim` in the Isaac Lab directory.

```bash
# create a symbolic link
ln -s ~/.local/share/ov/pkg/isaac-sim-4.1.0 ${IsaacLab_PATH}/_isaac_sim
```

Although using a virtual environment is optional, we recommend using `conda` (detailed instructions [here](https://isaac-sim.github.io/IsaacLab/v1.2.0/source/setup/installation/binaries_installation.html#setting-up-the-conda-environment-optional)).

From within the ORBIT-Surgical directory, create and activate your `conda` environment, followed by installing Isaac Lab and ORBIT-Surgical:

```bash
# Create conda environment
${IsaacLab_PATH}/isaaclab.sh --conda orbitsurgical

# Activate conda environment
conda activate orbitsurgical

# Install all ORBIT-Surgical extensions
./orbitsurgical.sh

# Install all isaac lab extensions
${IsaacLab_PATH}/isaaclab.sh --install
```

Once you are in the virtual environment, you do not need to use `${IsaacLab_PATH}/isaaclab.sh -p` to run python scripts. You can use the default python executable in your environment by running `python` or `python3`. However, for the rest of the documentation, we will assume that you are using `${IsaacLab_PATH}/isaaclab.sh -p` to run python scripts.


## Workflows

We adopt all robot learning workflows from Isaac Lab. We use the OpenAI Gym registry to register these environments. For each environment, we provide a default configuration file that defines the scene, observations, rewards and action spaces.

The list of environments available registered with OpenAI Gym can be found by running:

```bash
${IsaacLab_PATH}/isaaclab.sh -p workflows/robotic_surgery/scripts/simulation/scripts/environments/list_envs.py
```

### Basic agents

These include basic agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

Zero-action agent on `Isaac-Reach-PSM-v0`:

```bash
${IsaacLab_PATH}/isaaclab.sh -p workflows/robotic_surgery/scripts/simulation/scripts/environments/zero_agent.py --task Isaac-Reach-PSM-v0 --num_envs 32
```

Random-action agent on `Isaac-Reach-PSM-v0`:

```bash
${IsaacLab_PATH}/isaaclab.sh -p workflows/robotic_surgery/scripts/simulation/scripts/environments/random_agent.py --task Isaac-Reach-PSM-v0 --num_envs 32
```

### Teleoperation

We provide interfaces for providing commands in SE(2) and SE(3) space for robot control. In case of SE(2) teleoperation, the returned command is the linear x-y velocity and yaw rate, while in SE(3), the returned command is a 6-D vector representing the change in pose.

To play inverse kinematics (IK) control with a keyboard device:

```bash
${IsaacLab_PATH}/isaaclab.sh -p workflows/robotic_surgery/scripts/simulation/scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Needle-PSM-IK-Rel-v0 --num_envs 1 --teleop_device keyboard
```

The script prints the teleoperation events configured. For keyboard, these are as follows:

```
Keyboard Controller for SE(3): Se3Keyboard
    Reset all commands: L
    Toggle gripper (open/close): K
    Move arm along x-axis: W/S
    Move arm along y-axis: A/D
    Move arm along z-axis: Q/E
    Rotate arm along x-axis: Z/X
    Rotate arm along y-axis: T/G
    Rotate arm along z-axis: C/V
```


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
