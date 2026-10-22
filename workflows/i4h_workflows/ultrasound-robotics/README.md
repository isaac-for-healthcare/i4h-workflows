# Ultrasound Robotics Workflows

Workflows for robot-assisted ultrasound probe positioning and scanning.

## Workflows

| Workflow | Demonstration | Supported modes ([guide](../../i4h_workflow_modes/README.md)) |
| --- | --- | --- |
| `ultrasound_liver_scan` | Move an ultrasound probe across an abdominal phantom. | IsaacLab-Arena environment `PandaPhantom` (`panda_phantom`): PI0 policy, rule-based sweep, keyboard teleop, replay ([commands](#liver-scan)) |
| `ultrasound_probe_reach` | Align an ultrasound probe with a randomized target. | IsaacLab-Arena environment `UltrasoundProbeReach` (`ultrasound_probe_reach`); evaluate a trained RSL-RL checkpoint with Arena's `rsl_rl` policy (see below) |

## Demonstrations

Open the preview to view the animated demonstration.

| `ultrasound_liver_scan` |
| :---: |
| [![Robot performing an ultrasound liver scan](../../../docs/workflows/images/ultrasound_liver_scan.webp)](../../../docs/workflows/images/ultrasound_liver_scan.gif) |

Note: Complete the [project setup](../../../README.md#setup-from-the-command-line) before you begin.

## Liver Scan

`ultrasound_liver_scan` runs as the IsaacLab-Arena environment `PandaPhantom` (`panda_phantom`). Each episode starts with the probe pointing down above a randomly placed phantom and succeeds once the probe has made contact and scanned across it; it times out after 250 steps. From the repository root:

```bash
# Sweep the phantom with the rule-based controller for 3 episodes. Add --viz kit to watch.
arena/.venv/bin/python -m isaaclab_arena.evaluation.policy_runner \
  --policy_type i4h_arena.policies.scan.ProbeSweepPolicy --num_episodes 3 --enable_cameras \
  --external_environment_class_path i4h_arena.environments:PandaPhantom panda_phantom
```

The PI0 checkpoint, `nvidia/Liver_Scan_Pi0_Cosmos_Rel`, runs in its own openpi environment; `i4h_tasks.openpi_pi0.serve` serves it on openpi's websocket protocol, and Arena's openpi client drives the environment:

```bash
# Start the policy server (terminal 1). It downloads the checkpoint on first use and compiles
# it before it listens; pass --checkpoint for a locally trained one.
tasks/openpi_pi0/.venv/bin/python -m i4h_tasks.openpi_pi0.serve ultrasound_liver_scan --port 8000

# Evaluate it for 5 episodes (terminal 2).
arena/.venv/bin/python -m isaaclab_arena.evaluation.policy_runner \
  --policy_type i4h_arena.policies.pi0.LiverScanPi0Policy --remote_port 8000 \
  --num_episodes 5 --enable_cameras \
  --external_environment_class_path i4h_arena.environments:PandaPhantom panda_phantom
```

Add `--recorder_dataset_export_dir_path <dir> --recorder_dataset_filename <name>` after the environment name to record the episodes.

Drive the probe with the keyboard (`W/S/A/D/Q/E` move, `Z/X/T/G/C/V` rotate, `R` resets) and record successful demonstrations with Isaac Lab's tool:

```bash
arena/.venv/bin/python third_party/IsaacLab-e3f7d9f/scripts/tools/record_demos.py \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --external_environment_class_path i4h_arena.environments:PandaPhantom --task panda_phantom \
  --teleop_device keyboard --enable_cameras --viz kit \
  --dataset_file runs/ultrasound_liver_scan/teleop.hdf5 --num_demos 1
```

Replay a recording with `scripts/tools/replay_demos.py`, the same `--external_callback`, `--external_environment_class_path` and `--task`, and `--dataset_file <recording> --validate_success_rate`, without `--enable_cameras` (see [Hospital Automation Robotics](../hospital-automation-robotics/README.md#g1-loco-manipulation)).

[`scripts/e2e/run.sh`](../../../scripts/e2e/run.sh) runs the whole data-to-policy loop on this environment: record PI0 episodes, augment, grade them with a vision model against the scene's `annotate` criterion, replay, convert to LeRobot, finetune PI0, and validate the finetuned checkpoint.

## RL Training

`ultrasound_probe_reach` supports the maintained RSL-RL PPO training profile. A trained checkpoint is not included; train one, then evaluate it as a policy in the environment.

### Train with an AI Agent

```text
Train ultrasound_probe_reach with RL.
Evaluate 20 episodes and validate the exported policy.
```

The agent dry-runs, trains, evaluates, exports, and validates the policy.

### Train from the Command Line

Run the complete training lifecycle:

```bash
# Inspect the maintained RSL-RL PPO profile.
./train.sh rl show ultrasound_probe_reach

# Dry-run the training configuration.
./train.sh rl ultrasound_probe_reach \
  --num-envs 128 \
  --epochs 400 \
  --dry-run

# Train the policy from scratch.
./train.sh rl ultrasound_probe_reach \
  --num-envs 128 \
  --epochs 400

# Copy the timestamped run directory printed by the training command.
TRAIN_RUN="$PWD/runs/ultrasound_probe_reach/YYYYMMDD_HHMMSS"

# Evaluate the trained checkpoint.
./train.sh rl ultrasound_probe_reach \
  --eval \
  --checkpoint "$TRAIN_RUN/model_final.pt" \
  --episodes 20

# Evaluate the checkpoint as a policy in the environment with IsaacLab-Arena's runner.
# Arena's rsl_rl policy reads params/agent.yaml beside the checkpoint, which the
# timestamped Isaac Lab log directory inside the run holds.
arena/.venv/bin/python -m isaaclab_arena.evaluation.policy_runner \
  --policy_type rsl_rl --checkpoint_path "$(ls -d "$TRAIN_RUN"/20*/ | tail -1)model_399.pt" \
  --num_episodes 20 \
  --external_environment_class_path i4h_arena.environments:UltrasoundProbeReach ultrasound_probe_reach \
  --rl_observations

# Optionally export the policy to TorchScript and ONNX for deployment.
./train.sh rl export ultrasound_probe_reach \
  --checkpoint "$TRAIN_RUN/model_final.pt" \
  --output-dir "$TRAIN_RUN/exported"
```

Add `--viz kit` to watch the policy. `--rl_observations` selects the flat observations the policy was trained on.

RSL-RL training and evaluation are a separate lifecycle, not workflow run modes.
