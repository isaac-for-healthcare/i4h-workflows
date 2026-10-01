# Catheter Navigation: Imitation Learning and the Path to RL

How the IL pipeline for `endoluminal_navigation` is wired, how many demonstrations the
GR00T N1.7 task needs, what counts as success at each stage, how to validate a checkpoint,
and what RL post-training would require on top.

## Current state

Nothing is blocked on tooling. The pipeline is wired end to end and the recording schema
already matches the task contract exactly. The blocker is data: **there are no successful
demonstrations on disk.**

Seven teleoperated attempts exist across four recordings, all failed:

| Recording | Frames | Closest approach | Frames inside the 5 mm tolerance |
| --- | --- | --- | --- |
| `20260916_220249_fixed` demo_1 | 600 | 9.1 mm | 0 |
| `20260916_220249_fixed` demo_2 | 600 | 9.5 mm | 0 |
| `20260916_220249_fixed` demo_0 | 600 | 14.5 mm | 0 |
| `20260916_214025_ep1` | 600 | 45.2 mm | 0 |
| three others | 130–413 | 86–147 mm | 0 |

The last column is the one that matters. Arrival requires the tip within 5 mm of the target
for 15 consecutive steps, and across every recorded episode not one frame is inside that
tolerance. The behaviour that defines the task — the final approach and the hold — does not
appear anywhere in the data.

Six episodes on disk are flagged `success=True`, but all are `mode: demo`, the scripted
`basic/catheter_sweep` smoke task which has no navigation goal. Two of those are on the arm
workflow with 3-wide actions and 10-wide state, so they are not even schema-compatible with
the catheter task. They are not demonstrations of anything a policy should learn.

## The IL pipeline

Four stages. The manifests are the contract between them, which is why none of the stages
take a schema argument.

### 1. Record

```bash
RUN_DIR="runs/endoluminal_navigation/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

DISPLAY=:1 I4H_CATHETER_PROBE=30 ./run.sh endoluminal_navigation --teleop \
  --patient-twin ./data/TotalSegmentator/s0011/patient_twin.yaml \
  --episodes 1 --attempts 1 --episode-steps 600 \
  --record --record-failures \
  --run-dir "$RUN_DIR"
```

Writes workflow HDF5: 4-wide actions, 4-wide state, 256×256 fluoroscopy at 30 Hz, plus
per-frame diagnostics. `--patient-twin` is required — without it there is no centerline, the
route rail is never built, and insertion silently falls back to the tangent feed that buckles
the shaft.

Keyboard: `W`/`S` insert and withdraw at 9 mm/s, `A`/`D` rotate, `Z`/`C` curl the tip,
`1`–`4` select a C-arm projection, `Q`/`E` nudge the orbit, `R` resets. The viewport must
have focus or every key reads as unpressed.

### 2. Convert to LeRobot

```bash
DATASET_DIR="$RUN_DIR/lerobot/local/catheter_navigation"
uv run --project tools/dataset i4h-dataset convert \
  "$RUN_DIR/demos.hdf5" "$DATASET_DIR" \
  --robot catheter \
  --repo-id "local/catheter_navigation" \
  --successful-only \
  --task "Steer the catheter along the vessel and stop at the target branch"
```

`--successful-only` is **opt-in, not the default**. Omitting it converts failures too, which
is how you would otherwise train a policy on trajectories that never arrive.

The converter reads `arena/i4h_arena/embodiments/manifest/catheter.yaml` for the semantic
groups and writes `meta/modality.json`. The split matters: channels 0–2 are the `catheter`
group (insertion, rotation, tip bend) and channel 3 is `carm` alone, because the gantry only
changes the view. A policy can be trained on the catheter group with the C-arm as context.

### 3. Fine-tune

```bash
uv run --project tasks/gr00t_n17 i4h-tasks-gr00t-n17-train \
  --task gr00t_n17/catheter_navigation \
  --dataset "$DATASET_DIR" \
  --output-dir "$RUN_DIR/checkpoints" \
  --dry-run
```

Drop `--dry-run` to train. Defaults come from the `train:` block in
`tasks/gr00t_n17/i4h_tasks/gr00t_n17/manifest/catheter_navigation.yaml`:

| Setting | Value |
| --- | --- |
| `base_model` | `nvidia/GR00T-N1.7-3B` |
| `max_steps` | 10000 |
| `save_steps` | 1000 |
| `batch_size` | 32 |
| `modality_config` | `config_catheter` |
| `action_horizon` / `execution_steps` | 16 / 16 |
| `control_hz` | 30.0 |
| `image_size` | 256 × 256 |

`modality_config: config_catheter` selects `config_catheter.py` over the SO-ARM `config.py`,
because only one modality config can hold the `NEW_EMBODIMENT` tag in a process. Overrides
available on the CLI: `--max-steps`, `--save-steps`, `--batch-size`, `--num-gpus`,
`--base-model`, `--no-tune-visual`, `--no-tune-projector`, `--no-tune-diffusion-model`.

### 4. Validate

```bash
./run.sh endoluminal_navigation --mode policy_n17 \
  --checkpoint /absolute/path/to/checkpoint \
  --patient-twin ./data/TotalSegmentator/s0011/patient_twin.yaml \
  --episodes 20 --attempts 3 \
  --record verify.hdf5
```

`./scripts/e2e/run.sh --env endoluminal_navigation` chains all four stages through the
maintained driver if you would rather not run them by hand.

## How many episodes

Two independent estimates, and they agree closely enough to act on.

### From the training schedule

`config_catheter.py` sets `delta_indices=range(16)`, so each training sample is a 16-step
action chunk anchored at one frame. A 600-frame episode yields roughly 600 anchors, and the
default schedule draws 10000 × 32 = 320,000 samples:

| Episodes | Distinct anchors | Effective epochs |
| --- | --- | --- |
| 10 | ~6,000 | ~53 |
| 50 | ~30,000 | ~11 |
| 100 | ~60,000 | ~5 |
| 200 | ~120,000 | ~2.7 |

Below about 50 episodes the model sees each chunk dozens of times and memorizes rather than
generalizes.

### From GR00T's own guidance

`third_party/Isaac-GR00T-1.7/FAQ.md` gives post-training volumes by task class:

- Simple, fixed-location tasks: **~100 trajectories**
- Complex or multi-step scenes: ~500+
- High-DoF humanoid: ~2,000+
- Fine manipulation: ~100–500, ideally with human motion pre-training

Catheter navigation on a single twin with a single route is a fixed-location task by this
taxonomy, which puts it at **~100**. It is arguably fine manipulation given the 5 mm
tolerance, which would argue higher.

### Recommendation

**Target 100 successful episodes, but record 50 first and iterate.**

Three considerations push in different directions and are worth holding separately:

**Toward fewer.** One patient twin, one route, no anatomical variation. The task is close to
deterministic, and 50 episodes already gives a sane ~11 epochs.

**Toward more.** The decisive behaviour — inside 5 mm, held 15 steps — is about **3% of each
episode's frames**. At 50 episodes that is only ~750 frames of terminal-approach behaviour
covering the hardest part of the task. This rarity, not total frame count, is the real
argument for 100.

**Disk is the binding constraint.** Recordings run ~260 KB/frame, so a 600-frame episode is
~156 MB:

| Episodes | HDF5 size |
| --- | --- |
| 50 | ~8 GB |
| 100 | ~16 GB |
| 200 | ~31 GB |

With 20 GB free, 100 episodes does not fit alongside a LeRobot copy, a ~12 GB base model and
ten checkpoints. Freeing space is a prerequisite for the full target, not a detail.

GR00T's FAQ also recommends the iterative path directly: start with ~100 teleoperated
demonstrations, train, then use **HG-DAgger** — run the policy, intervene when it fails, and
add the corrected trajectories to the dataset. That covers out-of-distribution states pure
behaviour cloning misses, which for this task means the recovery behaviour after the tip
overshoots or catches a branch. Plan for a second recording round rather than trying to get
everything in the first.

## What success looks like from an IL perspective

Behaviour cloning fits a conditional distribution over action chunks given an observation. It
optimizes agreement with the demonstrator, **not task success**. Those come apart in specific
ways, so read them at four levels and do not let an earlier one stand in for a later one.

### Level 1: training loss converged

Says the optimizer worked. Says nothing about the policy. A loss curve cannot distinguish a
policy that navigates from one that has memorized the mean action of your demonstrations.
**Never report this as success.**

### Level 2: open-loop action agreement

Replay recorded observations and compare predicted chunks against ground-truth actions.
GR00T ships this as `gr00t/eval/open_loop_eval.py`, which writes a ground-truth-versus-
predicted plot with MSE metrics per trajectory. Per-channel error on insertion, rotation and
tip bend tells you the policy reproduces the demonstrator's commands at states the
demonstrator visited.

This is the first genuinely informative check, and it is still not success. It measures
agreement on the training distribution only. A policy can score well here and fail completely
closed-loop, because of **covariate shift**: once the policy drives, it reaches states no
demonstration contains, and its errors compound.

### Level 3: closed-loop rollout success

The policy drives the simulator and either satisfies the termination criterion or does not:
tip within 5 mm of the distal centerline for 15 consecutive control steps, inside 600 steps.
This is the real number. Report it as a fraction of attempted episodes.

Interpretation for this task specifically:

- **A policy that inserts but stops short** has learned the easy 97% of each episode and not
  the terminal approach. That is the expected first failure, and it is a data problem — see
  the 3% figure above.
- **A policy that overshoots or oscillates near the target** has learned insertion but not
  the hold. Also a data problem: your demonstrations need to contain the deliberate stop.
- **A policy that buckles the shaft** has learned demonstrator behaviour that was itself
  marginal. Check the demonstrations' `min_bend_radius_node` before blaming the policy.

### Level 4: rollout success plus physical plausibility

Success alone can be reached by a policy that jams the catheter through the vessel wall and
happens to end near the target. The per-frame diagnostics are in every recording precisely so
this is checkable:

| Diagnostic | What a good rollout looks like |
| --- | --- |
| `particles_outside` | stays at 0 |
| `worst_penetration_mm` | stays near 0 |
| `min_bend_radius_mm` | does not collapse; stays well above ~30 mm |
| `min_bend_radius_node` | not pinned at node 1, which indicates a proximal fold |
| `tip_target_distance_m` | decreases roughly monotonically |

A policy at level 3 but not level 4 is not a usable result for a medical workflow, however
good the success rate reads.

### A ceiling worth stating plainly

Behaviour cloning cannot exceed the demonstrator by much. If your demonstrations arrive by
luck after wandering, the policy learns to wander. The quality of 50 careful episodes beats
the quantity of 200 scrappy ones, and this is the main reason to fix the physics first —
which is now done — before collecting at volume.

## How to validate an IL checkpoint

Five gates in order, each catching a failure the previous one cannot see: the artifact
exists, inference loads it, it reproduces the operator's commands open-loop, it reaches the
target closed-loop, and the recorded diagnostics show it navigated rather than merely scored.

The runbook with the commands, pass criteria and failure-mode readings lives in
[`catheter-il-checkpoint-evaluation.md`](catheter-il-checkpoint-evaluation.md). Two rules
from it are worth repeating here because they are easy to break under pressure: never raise
the scene's 600-step cap to make a rollout pass, and never report "validated" without naming
which level was reached.

## Next steps for RL

### What exists today

Two maintained online-RL profiles, neither for this workflow:

| Workflow | Trainer | Starting artifact | Export |
| --- | --- | --- | --- |
| `ultrasound_probe_reach` | RSL-RL PPO | none, from scratch | TorchScript `policy.pt` → in-process Task |
| `assemble_trocar` | RLinf PPO actor/critic | local GR00T N1.5 SFT checkpoint | run bundle → GR00T export → remote Task |

`assemble_trocar` is the precedent to follow: it RL-post-trains a GR00T policy starting from a
supervised fine-tuned checkpoint. That is exactly the shape catheter navigation would take.

### The pinned RLinf already supports N1.7

Worth knowing before assuming a version blocker. `third_party/RLinf-a4b6abe` ships
`rlinf/models/embodiment/gr00t/gr00t_n1d7` alongside `gr00t_n1d5` and `gr00t_n1d6`, and there
is a working reference config at
`examples/embodiment/config/libero_spatial_ppo_gr00t_n1d7.yaml` using `model_type:
"gr00t_n1d7"` with `add_value_head: True`.

So the N1.5 pin is in the **i4h profile and adapter**, not in RLinf. The catheter task can
stay on N1.7; what is missing is an i4h profile, not upstream support.

### What catheter RL would require

Four pieces of work, roughly in dependency order.

**1. A reward function. Authored, not tuned.**
`CatheterEmbodiment.get_rewards_cfg()` binds the dense objective in
`arena/i4h_arena/medical/navigation_reward.py` to the scene's own route and lumen widths.
Seven terms: potential-based arc progress, exponential terminal approach, per-step arrival,
lateral offset, wall penetration, fold curvature, and action rate. The full specification,
including what terminates an episode and what does not, is in
[catheter-navigation-reward.md](catheter-navigation-reward.md).

What remains is tuning. The weights are sized against one episode rather than measured, and
the ordering — arriving worth more than traversing, traversing worth more than loitering —
is the intent while the exact numbers are not load-bearing.

**2. An `rl/profiles/endoluminal_navigation.yaml` plus a trainer config.**
Following the documented schema: `trainer: rlinf`, `algorithm: ppo_actor_critic`,
`adapter_module: i4h_rl.adapters.endoluminal_navigation`, `action_dof: 4`,
`state_dof: 4`, `cameras: [fluoroscopy]`, and train/eval task IDs registered against RLinf.
Plus `rl/config/endoluminal_navigation_ppo_gr00t.yaml` modelled on the trocar config but with
`model_type: "gr00t_n1d7"`.

**3. Vectorization, which is the real feasibility question.**
The trocar profile runs 64 parallel environments. This scene sets
`replicate_physics = False` and renders fluoroscopy through a Slang path per environment, and
each environment also carries an XPBD rod coupled to a deformable vessel. Whether 64 — or even
8 — environments are affordable on one A6000 is unknown and should be measured before any
profile work. Cost per environment-step here is far higher than for a Franka stacking cube.

**4. Two GPUs.**
The RLinf path isolates the model controller (Python 3.11, GR00T) from the simulator
(Python 3.12, Isaac Sim) as separate processes across two physical GPUs, bridged over a Unix
socket. This host currently has one A6000, so that is a hardware prerequisite, not a
configuration flag.

### How IL helps RL

This is the main reason to do the IL round properly even if RL is the goal.

**It provides the starting policy.** The `assemble_trocar` profile requires
`--model-path /path/to/gr00t-sft-checkpoint`; RLinf post-trains an existing policy rather than
training from scratch. A GR00T N1.7 SFT checkpoint on catheter demonstrations *is* that
artifact. Without it there is nothing for RL to start from, since training a 3B VLA from
random initialization by PPO is not a realistic option.

**It solves exploration, which is otherwise fatal here.** A randomly initialized policy
reaching a 5 mm target after 600 steps of a 4-DoF continuous action space is essentially
never going to happen by chance, so a sparse terminal reward yields zero gradient signal. An
IL-initialized policy already reaches 10 mm, which means non-zero success probability, which
means PPO has something to improve. IL converts an impossible exploration problem into a
tractable refinement problem.

**It anchors the policy against reward hacking.** PPO with a KL penalty against the reference
(SFT) policy — `kl_penalty: kl` in the trocar config — keeps the RL policy near demonstrated
behaviour while it optimizes. This is what stops it discovering that jamming the shaft through
the vessel wall reaches the target faster. For a medical workflow, that anchor is doing real
safety work, and its quality is exactly the quality of your demonstrations.

**It gives the value head a warm start.** `add_value_head: True` attaches a critic to the
GR00T actor. A policy whose visited states resemble demonstrated states produces returns the
critic can fit quickly, rather than the near-constant-zero returns a from-scratch policy
would generate.

**And RL is what exceeds the demonstrator.** BC is bounded by demonstration quality. RL
optimizes the actual objective, so it can find a smoother insertion profile or an earlier
steering commitment than any human produced — but only starting from a policy that already
mostly works.

### Suggested order

1. Drive 2–3 episodes to completion and confirm the hold triggers.
2. Free disk space.
3. Record 50 clean episodes; convert; fine-tune; validate to level 4.
4. Use HG-DAgger to add corrections where the policy fails; retrain toward 100 episodes.
5. Only once IL is rollout-validated: tune the reward weights, measure per-environment step
   cost, and decide whether RLinf post-training is affordable on available hardware.

Steps 1–4 have standing value on their own. Step 5 depends entirely on them.
