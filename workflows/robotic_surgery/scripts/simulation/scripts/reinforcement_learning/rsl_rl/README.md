# Reinforcement Learning

## Training and Playing

Train an agent with [RSL-RL](https://github.com/leggedrobotics/rsl_rl):

- **dVRK-PSM Reach (`Isaac-Reach-PSM-v0`)**:

```bash
# run script for training
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-PSM-v0 --headless
# run script for playing with 50 environments
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-PSM-Play-v0
```

- **Suture Needle Lift (`Isaac-Lift-Needle-PSM-IK-Rel-v0`)**:

```bash
# run script for training
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Needle-PSM-IK-Rel-v0 --headless
# run script for playing with 50 environments
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Lift-Needle-PSM-IK-Rel-Play-v0
```

### TensorBoard: TensorFlow's visualization toolkit

Monitor the training progress stored in the `logs` directory on [Tensorboard](https://www.tensorflow.org/tensorboard):

```bash
# execute from the root directory of the repository
python -m tensorboard.main --logdir=logs
```

## Documentation Links

- [IsaacLab Task Setting](../../../exts/robotic.surgery.tasks/docs/README.md)
- [Assets](../../../utils/assets.py)
