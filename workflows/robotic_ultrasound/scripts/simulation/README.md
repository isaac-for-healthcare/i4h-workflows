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
conda create -n robotic_ultrasound python=3.10
# Activate the environment
conda activate robotic_ultrasound
# Install PyTorch on CUDA 12
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# Upgrade pip
pip install --upgrade pip
# Install IsaacSim
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
# Verify installation
isaacsim
# Install Isaac Lab outside of the repo
cd ..
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
# Checkout v1.4.1
git checkout v1.4.1
# Install dependencies
sudo apt install cmake build-essential
# Install Isaac Lab
cd IsaacLab
./isaaclab.sh --install
# Test installation
python source/standalone/tutorials/00_sim/create_empty.py
```

# Requirements

2. Follow the [Installation](#installation) instructions below to install the extension.

From the root directory, run:

```sh
# Activate the environment
conda activate robotic_ultrasound
# Ensure toml is installed
pip install toml
# Install the environments
python -m pip install -e exts/robotic_us_ext
```

# Environments

Every Isaac-Lab scripts receives a `--task <task_name>` argument. This argument is used to select the environment/task to be run.
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

### Setup

1. [FIXME: change to `asset-catalog`] Download the v0.3 model weights from [GDrive](https://drive.google.com/drive/folders/1sL4GAETSMbxxcefsTsOkX7wXkTsbDqhW?usp=sharing)

2. Please check the [I4H asset catalog](https://github.com/isaac-for-healthcare/i4h-asset-catalog) for assets downloading, put the USD assets as "./assets/Collected_phantom"

3. [FIXME] Follow the internal GitLab pi0 repo setup instructions: [here](https://gitlab-master.nvidia.com/nigeln/openpi_zero#installation)

4. [FIXME] Use the same [pi0 repo](https://gitlab-master.nvidia.com/nigeln/openpi_zero#3-spinning-up-a-policy-server-and-running-inference) to serve the model over a websocket:
```sh
uv run scripts/serve_policy.py \
   policy:checkpoint \
   --policy.config=pi0_chiron_aortic \
   --policy.dir=<your_path>/19000 # Ensure the ckpt dir passed contains the ./params folder

```
4. Use your conda environment to install their client package:
```sh
cd <path_to_openpi_repo>/openpi/packages/openpi-client
pip install -e .
```
5. Now that you can use their client helper scripts, return to this folder and run the following command:
```sh
export PYTHONPATH=`pwd`
python policies/state_machine/pi0_policy/eval.py \
        --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
        --enable_camera
```
(Optional) We can also use RTI to publish the joint states to the physical robot:
```sh
python policies/state_machine/pi0_policy/eval.py \
        --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
        --enable_camera \
        --send_joints \
        --host 0.0.0.0 \
        --port 8000 \
        --domain_id 17 \
        --rti_license_file /media/m3/repos/robotic_ultrasound_internal/rti_license.dat
```