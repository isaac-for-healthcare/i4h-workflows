# Examples for Isaac simulation

## Simulation for robotic ultrasound based on DDS communication
This example should work together with the `pi0 policy runner` via DDS communication.
```sh
python examples/sim_with_dds.py \
    --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
    --enable_camera \
    --domain_id <domain id> \
    --rti_license_file <path to>/rti_license.dat
```