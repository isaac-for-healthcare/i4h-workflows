# Policy training for robotic ultrasounc

## Table of Contents
- [Setup](#setup)
- [Apps](#apps)
  - [PI Zero Policy Evaluation](#pi-zero-policy-evaluation)

# Setup

1. Follow [simulation setup](../scripts/simulation/README.md) to set up all the requirements.

# Apps

## PI Zero Policy Evaluation

### Setup

1. [FIXME: change to `asset-catalog`] Download the v0.3 model weights from [GDrive](https://drive.google.com/drive/folders/1sL4GAETSMbxxcefsTsOkX7wXkTsbDqhW?usp=sharing)

2. [FIXME] Follow the internal GitLab pi0 repo setup instructions: [here](https://gitlab-master.nvidia.com/nigeln/openpi_zero#installation)

3. [FIXME] Use the same [pi0 repo](https://gitlab-master.nvidia.com/nigeln/openpi_zero#3-spinning-up-a-policy-server-and-running-inference) to serve the model over a websocket:
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
python scripts/state_machine/pi0_policy/pi0_eval.py \
        --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
        --enable_camera
```
(Optional) We can also use RTI to publish the joint states to the physical robot:
```sh
python scripts/state_machine/pi0_policy/pi0_eval.py \
        --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
        --enable_camera \
        --send_joints \
        --host 0.0.0.0 \
        --port 8000 \
        --domain_id 17 \
        --rti_license_file /media/m3/repos/robotic_ultrasound_internal/rti_license.dat
```
