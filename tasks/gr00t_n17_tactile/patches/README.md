# GR00T N1.7 tactile patch series

This stack runs Isaac-GR00T N1.7 at the revision `tasks/gr00t_n17` already pins
(`4b1dca9d`, see `third_party/setup.sh`) plus a patch series that adds the
tactile path. The series is not part of this structure-first draft. It lands in
the code follow-up as one patch file in `third_party/`, applied by
`third_party/setup.sh` through `apply_patch_once`, the mechanism already used
for `gr00t_action_head_future_tokens.patch` on N1.5, onto a second checkout
`third_party/Isaac-GR00T-1.7-tactile`, so the vanilla stack is never touched.
The series was developed on the N1.7 release branch four days before the pinned
revision; the two differ only in README and demo metadata, so it applies as is.

With `use_tactile` off, the patched model, processor and loader behave exactly
as the vanilla ones. That is what the existing N1.7 tests keep checking.

## What the series changes

31 files, about 4.7k lines added, 37 removed. By concern:

| Concern | Files |
| --- | --- |
| Model: one token per fingertip, zero-initialised tactile cross-attention in the 16 cross-attention blocks of the action head | `gr00t/model/modules/tactile/{sat_encoder,spatial_anchor,fk,sharpa_encoder,trex_force_encoder}.py`, `gr00t/model/modules/dit.py`, `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`, `gr00t/model/gr00t_n1d7/setup.py`, `gr00t/configs/model/gr00t_n1d7_tactile.py` |
| Frozen third-party encoders | Loaded from their public upstreams at pinned revisions (see `../README.md`); the follow-up carries no copy of Sharpa or T-Rex code |
| Data: `tactile` and `tactile_force` modality categories, causal force window, delay augmentation, tactile dropout | `gr00t/data/types.py`, `gr00t/data/dataset/{lerobot_episode_loader,sharded_single_step_dataset,sharded_mixture_dataset}.py`, `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` |
| Training flags | `gr00t/configs/finetune_config.py`, `gr00t/experiment/launch_finetune.py` |
| Inference: per-step tactile refresh, slow/fast cascade, `server_info`, `--disable-tactile` | `gr00t/policy/gr00t_policy.py`, `gr00t/eval/run_gr00t_server.py` |
| Packaging and tests | `pyproject.toml`, `tests/fixtures/processor_config/processor_config.json`, `tests/gr00t/model/test_tactile_force_flow.py`, `tests/gr00t/policy/test_gr00t_policy.py` |

## The design in one paragraph

Each fingertip becomes one token: the frozen SharpaWave encoder's 512 features
of its deformation map, concatenated with the frozen T-Rex code of its recent
6-axis force history, modulated (FiLM) and positioned by the fingertip's pose,
which forward kinematics computes from the hand joints recorded with the frame.
The tokens enter the N1.7 action head through an additional cross-attention
whose output projection starts at zero, so a freshly patched checkpoint is
numerically the vanilla one until finetuning moves those weights. Training
delays the tactile stream by a random amount of up to 16 frames and conditions
on whether the flow time is past `tau_split`; that is what lets inference
refresh tactile inputs once per denoising step and, in the slow/fast cascade,
re-plan the tail of a chunk with fresher data. Tactile dropout keeps the
vision-only behaviour of the same checkpoint usable (`--disable-tactile`).

## Open question for the review

Whether the series stays a patch in this repository or is offered to
`NVIDIA/Isaac-GR00T` as a pull request. It is written as a self-contained
addition either way.
