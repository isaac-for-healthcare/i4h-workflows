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


## Robotic Surgery Tasks

We provide examples on hand-crafted state machines for the robotic surgery environments, demonstrating the execution of surgical subtasks.

- **dVRK-PSM Reach**: da Vinci Research Kit (dVRK) Patient Side Manipulator (PSM) to reach a desired pose:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_psm_sm.py
```

- **Dual-arm dVRK-PSM Reach**: dual-arm dVRK-PSM to reach a desired pose:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_dual_psm_sm.py
```

- **STAR Reach**: STAR arm to reach a desired pose:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_star_sm.py
```

- **Suture Needle Lift**: lift a suture needle to a desired pose:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/lift_needle_sm.py
```

- **Peg Block Lift**: lift a peg block to a desired pose:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/lift_block_sm.py
```

## Reinforcement Learning

Train an agent with [RSL-RL](https://github.com/leggedrobotics/rsl_rl):

- **dVRK-PSM Reach (`Isaac-Reach-PSM-v0`)**:

```bash
# run script for training
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-PSM-v0 --headless
# run script for playing with 50 environments
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-PSM-Play-v0
```

- **Suture Needle Lift (`Isaac-Lift-Needle-PSM-v0`)**:

```bash
# run script for training
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Needle-PSM-v0 --headless
# run script for playing with 50 environments
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Lift-Needle-PSM-Play-v0
```

### TensorBoard: TensorFlow's visualization toolkit

Monitor the training progress stored in the `logs` directory on [Tensorboard](https://www.tensorflow.org/tensorboard):

```bash
# execute from the root directory of the repository
python -m tensorboard.main --logdir=logs
```
