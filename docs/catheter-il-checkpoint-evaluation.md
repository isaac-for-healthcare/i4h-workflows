# Evaluating a Catheter Navigation IL Checkpoint

A runbook for deciding whether a GR00T N1.7 checkpoint fine-tuned on
`endoluminal_navigation` demonstrations is any good.

Companion to [`catheter-navigation-il-and-rl.md`](catheter-navigation-il-and-rl.md), which
covers the pipeline that produces the checkpoint, how many demonstrations it needs, and what
RL post-training would add on top.

## Why five gates instead of one number

A checkpoint can fail in five different ways, and each way is invisible to the check below
it. Run them in order and stop at the first failure: a twenty-episode rollout that discovers
the checkpoint never loaded is an hour spent on a question that takes ten seconds.

The reason this matters more here than for a typical model: the policy is trained by
**behaviour cloning**, which optimizes *predict what the operator typed*, not *reach the
target*. Those two objectives genuinely come apart, so a healthy loss curve and a healthy
success rate can each be misleading, in opposite directions.

| Gate | Question | Cost |
| --- | --- | --- |
| 1 | Is there a real artifact on disk? | seconds |
| 2 | Can inference load it? | ~10 s |
| 3 | Does it reproduce the operator's commands? | minutes |
| 4 | Does it reach the target when it drives? | ~1 h |
| 5 | Did it navigate, or just score? | minutes |

Throughout, `$RUN_DIR` is the run directory from recording, and checkpoint paths must be
absolute.

## Gate 1: a real artifact on disk

A training run that exits cleanly proves files were written, nothing more. Trainers write
both a final model at the output root *and* numbered subdirectories, and an output directory
is not a checkpoint.

```bash
ls -la "$RUN_DIR/checkpoints"
du -sh "$RUN_DIR/checkpoints"/*
```

**Pass:** you can name one exact path, and its size is plausible — roughly base-model-sized
(~12 GB for `nvidia/GR00T-N1.7-3B`), or smaller if only adapters were tuned. A directory of a
few megabytes means the weights did not save.

## Gate 2: inference loads it

Training and inference are different code paths. Loading through the inference backend
catches modality mismatches, wrong embodiment tags and action-space disagreements before
Isaac Sim ever starts.

```bash
uv run --project tasks/gr00t_n17 python -m i4h_tasks.gr00t_n17.server \
  --namespace "smoke-$$" \
  --preload gr00t_n17/catheter_navigation \
  --checkpoint /absolute/path/to/checkpoint \
  --preload-only
```

`--preload-only` loads the requested checkpoints and exits without serving.

**Pass:** exit status 0, no traceback. The usual failure is the policy expecting a different
action width than the 4 the catheter task produces, which surfaces as a shape error.

## Gate 3: open-loop agreement

Replay observations from recordings the policy has already seen and compare its predicted
action chunks against what the operator actually did. GR00T ships this as
`third_party/Isaac-GR00T-1.7/gr00t/eval/open_loop_eval.py`, which writes a
ground-truth-versus-predicted plot with per-trajectory MSE.

Read the error **per channel** rather than as one number, because the channels are not
equally important:

| Channel | Meaning | Large error means |
| --- | --- | --- |
| 0 | insertion | has not learned when to push versus hold |
| 1 | rotation | has not learned the steering |
| 2 | tip bend | has not learned the shaping |
| 3 | C-arm | harmless — the gantry only changes the view |

**Pass:** low, flat error on channels 0–2.

Be precise about what this proves: the policy agrees with the operator *at states the
operator visited*. Once the policy drives, it reaches states no demonstration contains and
its errors compound. That is **covariate shift**, and it is why a policy can score well here
and still fail completely at gate 4.

## Gate 4: closed-loop rollout

The policy drives the simulator and either satisfies the termination criterion or does not:
tip within 5 mm of the distal centerline for 15 consecutive control steps, inside 600 steps.

```bash
./run.sh endoluminal_navigation --mode policy_n17 \
  --checkpoint /absolute/path/to/checkpoint \
  --patient-twin ./data/TotalSegmentator/s0011/patient_twin.yaml \
  --episodes 20 --attempts 3 \
  --record verify.hdf5
```

Run it visibly the first time. Thirty seconds of watching the catheter tells you what a
success rate over twenty episodes only suggests.

**Pass:** exit status 0 and the final `N/N episodes succeeded` line. Report attempts and
retries separately — 20/20 at three attempts each is a very different policy from 20/20
first try.

Two constraints that are not negotiable:

- `--patient-twin` is required. Without it there is no centerline, the route rail is never
  built, and insertion falls back to the tangent feed that buckles the shaft — you would be
  testing the policy against broken physics.
- Never raise the scene's 600-step cap to make a rollout pass. `--episode-steps` may only
  lower it.

### Reading the failure mode

The way it fails tells you what to fix:

- **Inserts but stops short.** Learned the easy 97% of each episode and not the terminal
  approach. This is the expected first failure and it is a *data* problem: the arrival hold
  is only about 3% of each episode's frames.
- **Overshoots or oscillates near the target.** Learned insertion but not the deliberate
  stop. Also data — the demonstrations need to contain a clean hold.
- **Buckles the shaft.** Learned demonstrator behaviour that was itself marginal. Check the
  demonstrations' `min_bend_radius_node` before blaming the policy.

## Gate 5: navigated, or merely scored

This is the gate that gets skipped. A policy can satisfy the success term by driving the
catheter through the vessel wall and ending up near the target. The per-frame diagnostics are
recorded precisely so that is checkable.

```bash
uv run --project tools/dataset i4h-dataset inspect "$RUN_DIR/verify.hdf5" --segments
uv run --project tools/dataset i4h-dataset actions "$RUN_DIR/verify.hdf5"
```

Then read the numbers out directly:

```python
import h5py
from i4h_common.episode import episodes

with h5py.File("verify.hdf5") as handle:
    for ep in episodes(handle):
        print(ep.name, "success:", ep.success)
        print("  wall breaches:", ep.diagnostic("particles_outside").max())
        print("  worst penetration mm:", ep.diagnostic("worst_penetration_mm").max())
        print("  min bend radius mm:", ep.diagnostic("min_bend_radius_mm").min())
        print("  at node:", ep.diagnostic("min_bend_radius_node").min())
        distance = ep.diagnostic("tip_target_distance_m")
        print("  approach mm:", 1000 * distance[0], "->", 1000 * distance.min())
```

**Pass:**

| Diagnostic | Healthy rollout |
| --- | --- |
| `particles_outside` | stays at 0 |
| `worst_penetration_mm` | stays near 0 |
| `min_bend_radius_mm` | stays well above ~30 mm |
| `min_bend_radius_node` | never pinned at node 1, which indicates a proximal fold |
| `tip_target_distance_m` | decreases roughly monotonically, rather than wandering in |

A checkpoint that passes gate 4 but fails gate 5 is not a usable result for a medical
workflow, however good the success rate reads.

## Reporting

State which level was reached, using these words:

- **Structurally valid** — `show` and `lint` pass.
- **Launchable** — the mode starts and the checkpoint preloads (gate 2).
- **Rollout-validated** — requested episodes complete and the recorded evidence passes
  inspection (gates 4 and 5 together).

Do not report "validated" without naming the level.

## Prerequisite

None of this can run yet. Gate 3 onward needs a checkpoint, and a checkpoint needs training
data in which the tip actually arrives. Across every teleoperated episode recorded so far,
not one frame is inside the 5 mm tolerance, so the behaviour the evaluation is checking for
does not yet exist in the demonstrations. See
[`catheter-navigation-il-and-rl.md`](catheter-navigation-il-and-rl.md) for the current data
inventory and the recording plan.
