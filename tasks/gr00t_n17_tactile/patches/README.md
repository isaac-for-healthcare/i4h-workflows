# GR00T N1.7 tactile patch series

This stack runs Isaac-GR00T N1.7 at the revision `tasks/gr00t_n17` already pins
(`4b1dca9d`, see `third_party/setup.sh`) plus one patch that adds the tactile
path: `third_party/gr00t_n17_tactile.patch`. `third_party/setup.sh
tasks/gr00t_n17_tactile` creates a second checkout of that revision,
`third_party/Isaac-GR00T-1.7-tactile`, and applies the patch there through
`apply_patch_once`, the mechanism already used for
`gr00t_action_head_future_tokens.patch` on N1.5, so the vanilla stack is never
touched. The series was developed on the N1.7 release branch four days before
the pinned revision; the two differ only in README and demo metadata, so it
applies as is.

With `use_tactile` off, the patched model, processor and loader behave exactly
as the vanilla ones. That is what the existing N1.7 tests keep checking.

## What the patch changes

24 files, about 4.1k lines added, 37 removed. By concern:

| Concern | Files |
| --- | --- |
| Model: per-fingertip tactile features, zero-initialised tactile cross-attention in the 16 cross-attention blocks of the action head | `gr00t/model/modules/tactile/{sat_encoder,spatial_anchor,fk,sharpa_encoder,trex_force_encoder,checkpoint_io}.py`, `gr00t/model/modules/dit.py`, `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`, `gr00t/model/gr00t_n1d7/setup.py`, `gr00t/configs/model/gr00t_n1d7_tactile.py` |
| Frozen third-party encoders | Imported by module name from their public checkouts (`sharpawave_deform_encoder`, `tactile_vqvae`). Neither is installed as a package (the Sharpa repository ships no `pyproject`; T-Rex's pins a full VLA stack on Python 3.10 and torch 2.6), so the task's server and trainer put `third_party/sharpawave-deform-encoder` and `third_party/T-Rex` on the import path. The patch carries no copy of Sharpa or T-Rex code |
| Data: `tactile` and `tactile_force` modality categories, causal force window, delay augmentation, tactile dropout | `gr00t/data/types.py`, `gr00t/data/dataset/{lerobot_episode_loader,sharded_single_step_dataset,sharded_mixture_dataset}.py`, `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` |
| Training flags | `gr00t/configs/finetune_config.py`, `gr00t/experiment/launch_finetune.py` |
| Inference: per-step tactile refresh, slow/fast cascade, `server_info`, `--disable-tactile` | `gr00t/policy/gr00t_policy.py`, `gr00t/eval/run_gr00t_server.py` |
| Packaging and tests | `pyproject.toml` (a `tactile` extra: `timm`, `pytorch_kinematics`), `tests/fixtures/processor_config/processor_config.json`, `tests/gr00t/model/test_tactile_force_flow.py`, `tests/gr00t/policy/test_gr00t_policy.py` |

The two encoder wrappers accept the public weight files: the SharpaWave
encoder loads the `deform_encoder.*` tensors of the published SafeTensors
checkpoint (or of a `.pth` state dict with the same keys), and the T-Rex
wrapper extracts the `tactile_vqvae.*` tensors from the published VLA
checkpoint, replacing the decoder it never runs by an identity so that the
module's state dict matches the checkpoints finetuned so far.

## The design in one paragraph

Each fingertip contributes two tactile signals: the frozen SharpaWave
encoder's 512 features of its deformation map, modulated (FiLM) and positioned
by the fingertip's pose, which forward kinematics computes from the hand joints
recorded with the frame, and the frozen T-Rex code of its recent 6-axis force
history. The two are concatenated per fingertip, and these per-fingertip
features enter the N1.7 action head through an additional cross-attention
whose output projection starts at zero, so a freshly patched checkpoint is
numerically the vanilla one until finetuning moves those weights. Training
delays the tactile stream by a random amount of up to 16 frames and conditions
on whether the flow time is past `tau_split`; that is what lets inference
refresh tactile inputs once per denoising step and, in the slow/fast cascade,
re-plan the tail of a chunk with fresher data. Tactile dropout keeps the
vision-only behaviour of the same checkpoint usable (`--disable-tactile`).

## Regenerating the patch

The series lives as a branch of an Isaac-GR00T fork. From a checkout of that
branch, with the pinned revision fetched:

```bash
git diff 4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e <branch> -- gr00t tests pyproject.toml \
  > third_party/gr00t_n17_tactile.patch
git -C third_party/Isaac-GR00T-1.7-tactile apply --check third_party/gr00t_n17_tactile.patch
```

## Deployment-side patches

The deployment client (`i4h_tasks/gr00t_n17_tactile/client/`) drives the robot
through two more pinned checkouts of `third_party/setup.sh
tasks/gr00t_n17_tactile`. `third_party/xr_teleoperate` is the `teleop/`
directory of the isaac-for-healthcare fork of `unitreerobotics/xr_teleoperate`
at `1cb9c7fb` (branch `i4h/h2-sharpa`, tag `i4h-h2-sharpa-v2`): upstream's H2
branch `ebe257f0` plus two commits, one with the H2 arm controller changes, the
SharpaWave hand DDS client, the ROS 2 camera client, the DDS network
configuration, the body peer discovery and `teleop/utils/sharpa_tactile_wire.py`,
the tactile wire format the robot-side bridge shares with teleop recording (the
client reads it through its own vendored copy), and one adding the NVIDIA SPDX
headers to those files. It needs no patch: a change on
the teleop side is a commit on that branch and a new `XR_TELEOPERATE_REV`.
`third_party/unitree_sdk2_network_config.patch` applies to
`unitreerobotics/unitree_sdk2_python` at `794fb2b3` and adds the `networkConfig`
keyword to `ChannelFactoryInitialize`. To check it:

```bash
git -C third_party/unitree_sdk2_python apply --check third_party/unitree_sdk2_network_config.patch
```

## Open question for the review

Whether the series stays a patch in this repository or is offered to
`NVIDIA/Isaac-GR00T` as a pull request. It is written as a self-contained
addition either way.
