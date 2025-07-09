# State Machine Environments

We provide examples on hand-crafted state machines for the robotic surgery environments, demonstrating the execution of surgical subtasks.

## Running the State Machine Environments

### dVRK-PSM Reach

Drive the da Vinci Research Kit (dVRK) Patient Side Manipulator (PSM) to reach a desired pose:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_psm_sm.py
```

### Dual-arm dVRK-PSM Reach

Drive the dual-arm dVRK-PSM to reach a desired pose:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_dual_psm_sm.py
```

### STAR Reach

Drive the STAR arm to reach a desired pose:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_star_sm.py
```

### Suture Needle Lift

Lift a suture needle to a desired pose:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/lift_needle_sm.py
```

### Organs Suture Needle Lift

Lift a suture needle from an organ to a desired pose in the operating room:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/lift_needle_organs_sm.py
```

### Peg Block Lift

Lift a peg block to a desired pose:
```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/lift_block_sm.py
```

## Documentation Links

- [IsaacLab Task Setting](../../../exts/robotic.surgery.tasks/docs/README.md)
- [Assets](../../../utils/assets.py)