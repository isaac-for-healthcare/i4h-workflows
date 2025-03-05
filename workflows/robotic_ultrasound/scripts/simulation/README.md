# IsaacLab simulation for robotic ultrasound

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Environments](#environments)
- [Apps](#apps)
  - [PI Zero Policy Training](#pi-zero-policy-training)
  - [PI Zero Policy Evaluation](#pi-zero-policy-evaluation)

# Requirements

1. Follow [instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to install IsaacLab. [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is suggested for virtual environment setup. TLDR;

```sh
# Create a new conda environment
conda create -n robotic_ultrasound python=3.10 -y
# Activate the environment
conda activate robotic_ultrasound
# Upgrade pip
pip install --upgrade pip
# Install IsaacSim
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
# Install Isaac Lab outside of the repo
cd <some workspace>
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
# Checkout v1.4.1
git checkout v1.4.1
cd IsaacLab
# Patch (temporary workaround for Isaacsim 4.2 + IsaacLab 1.4.1)
sed -i 's/rsl-rl/rsl-rl-lib/g' source/extensions/omni.isaac.lab_tasks/setup.py
# Install dependencies
sudo apt install cmake build-essential
# Install Isaac Lab
./isaaclab.sh --install
# Test installation
python source/standalone/tutorials/00_sim/create_empty.py
```

# Installation

2. Follow the [Installation](#installation) instructions below to install the extension.

From the root directory, run:

```sh
# Activate the environment
conda activate robotic_ultrasound
# Ensure toml is installed
pip install toml
# Install the environments
cd <repo root>/workflows/robotic_ultrasound/scripts/simulation
python -m pip install -e exts/robotic_us_ext
```

# Environments

Isaac-Lab scripts usually receive a `--task <task_name>` argument. This argument is used to select the environment/task to be run.
These tasks are detected by the `gym.register` function in the [exts/robotic_us_ext/\__init__.py](exts/robotic_us_ext/robotic_us_ext/tasks/ultrasound/approach/config/franka/__init__.py).

These registered tasks are then further defined in the [environment configuration file](exts/robotic_us_ext/robotic_us_ext/tasks/ultrasound/approach/config/franka/franka_manager_rl_env_cfg.py).

Available tasks:
- `Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0`: FrankaUsRs with relative actions
- `Isaac-Reach-Torso-FrankaUsRs-IK-RL-Abs-v0` : FrankaUsRs with absolute actions

Currently there are these robot configurations that can be used in various tasks


| Robot name                                 | task             | applications          |
|----------                                  |---------         |----------             |
| FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG      | \*-FrankaUsRs-*  | Reach, Teleop         |

# Apps

## PI Zero Policy Training
[FIXME]

## PI Zero Policy Evaluation
Set up `openpi` referring to [PI0 runner](../policy_runner/README.md).

4. Now that move to the [scripts](../) folder and specify python path:
```sh
export PYTHONPATH=`pwd`
```

5. Return to this folder and run the following command:
```sh
python policies/state_machine/pi0_policy/eval.py \
    --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
    --enable_camera \
    --ckpt_path <path to ckpt>/pi0_aortic_scan_v0.3/19000 \
    --repo_id hf/chiron_aortic
```
