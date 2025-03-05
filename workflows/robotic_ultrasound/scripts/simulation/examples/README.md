# Examples for Isaac simulation

## Simulation for robotic ultrasound based on DDS communication
This example should work together with the `pi0 policy runner` via DDS communication,
so please ensure to launch the `run_policy.py` with `height=224`, `width=224`,
and the same `domain id` as this example in another terminal.

When `run_policy` is launched and idle waiting for the data,
move to the [scripts](../) folder and specify python path:
```sh
export PYTHONPATH=`pwd`
```
Then back to this folder and execute:
```sh
python sim_with_dds.py \
    --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
    --enable_camera \
    --domain_id <domain id> \
    --rti_license_file <path to>/rti_license.dat
```
