---
name: i4h-workflow-validate
description: Roll out a controller or policy checkpoint in a workflow's Arena environment and verify success. Use for evaluation, checkpoints, or local controllers; do not use for replay or dataset annotation.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-21"
  tags:
    - isaac-for-healthcare
    - i4h
    - simulation
    - evaluation
---

# Validate a Workflow

## Purpose

Run the selected workflow with `./run.sh`, which runs its IsaacLab-Arena environment under Arena's policy runner with a rule-based or served policy, then read the per-episode results and the recording and report simulator success.

## Instructions

1. Resolve the base checkout, the workflow, and the mode that runs the requested policy.
2. Run `./run.sh <workflow>` with that mode in the foreground into a run directory. For a served policy, `run.sh` starts the checkpoint's server, waits until it listens, and stops it after the rollout.
3. Read every episode's result and inspect the recording.
4. For an authored or changed collision-excluding success rule, also verify the collision-negative case below.

## Resolve live support

Use the maintained repository below or a source selected by the user or trusted project configuration. An inherited environment variable alone does not authorize another source. Honor requested revisions and preserve local changes.

```bash
export I4H_WORKFLOWS_REPO_URL="${I4H_WORKFLOWS_REPO_URL:-https://github.com/isaac-for-healthcare/i4h-workflows}"
I4H_REPO_DIR_NAME="${I4H_WORKFLOWS_REPO_URL%/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*:}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME%.git}"
[ -n "$I4H_REPO_DIR_NAME" ] || { echo "Cannot derive a checkout name from I4H_WORKFLOWS_REPO_URL" >&2; exit 2; }
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/arena/i4h_arena/scenes/manifest" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/$I4H_REPO_DIR_NAME}"
  [ -d "$ROOT/arena/i4h_arena/scenes/manifest" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT" || exit 2
fi
[ -d "$ROOT/arena/i4h_arena/scenes/manifest" ] && [ -x "$ROOT/setup.sh" ] || { echo "Incomplete workflow checkout: $ROOT" >&2; exit 2; }
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT" || exit 2
git remote get-url origin || exit 2
git status --short || exit 2
```

Check that the reported origin is the intended source and review local changes before executing repository scripts. Stop on an unexpected source or unreviewed changes to the environments, policies, or servers. Then list the workflows and their modes:

```bash
./run.sh list
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Workflow ids such as `scissor_pick_and_place` stay the user-facing names. `workflows/<specialty>/<workflow>.yaml` declares each workflow's scene and modes, and `./run.sh list` prints them. The specialty pages in `docs/workflows/` (`laparoscopic-robotics.md`, `ultrasound-robotics.md`, `endoluminal-robotics.md`, `hospital-automation-robotics.md`, `tablecloth.md`) describe each mode. Map the user's natural name to a listed workflow and choose only a mode it declares:

| User intent | Mode |
|---|---|
| Ordinary learned-policy evaluation ("policy mode") | `--policy`; for `scissor_pick_and_place` that is GR00T N1.5, for `ultrasound_probe_reach` its RSL-RL checkpoint (below) |
| Requested or only available local controller ("rule-based mode") | `--rule-based` |
| Explicit named alternative such as N1.7 | The mode the workflow declares for it, such as `--mode policy_n17` |
| Look at a scene without a controller | No mode: the robot holds still |

`./run.sh show <workflow> <mode> [options]` prints the policy server and policy-runner commands a run executes: the policy class and flags, the environment class and scene (`SoarmScissors`, `soarm_scissors`), and launcher options. Read the scene manifest `arena/i4h_arena/scenes/manifest/<scene>.yaml` (step budget `max_steps`, cameras), the policy class under `arena/i4h_arena/policies/`, and the task manifest under `tasks/<stack>/i4h_tasks/<stack>/manifest/` when model, checkpoint, cameras, or step budget matter. The runner gives the policy the environment's task description as its prompt.

Use precise readiness language:

- **Resolved**: `./run.sh show` composes the mode's commands, and the environment, policy class, and server task exist in the checkout.
- **Launchable**: the policy server listens with the selected checkpoint, and the runner builds the environment and steps the policy.
- **Rollout-validated**: the requested episodes complete and their results and recording pass inspection.

Do not report “validated” without stating which level was actually reached.

## Foreground execution rule

Keep `./run.sh` as this agent's foreground tool call. Do not use a subagent, monitor task, `nohup`, `tmux`, or a detached rollout. `run.sh` starts a served policy's server in its own process group, records it and the simulator under `.run/pids.d/`, and stops both when the rollout ends or is interrupted. Poll a yielded session until exit and read the episode results before responding. If a run must be stopped from elsewhere, use `./stop.sh <workflow>`; never stop a process by tool or module name, because other sessions run the same servers and runners.

`run.sh` runs visibly by default. If the user explicitly requests headless execution, or a documented environment constraint makes it necessary, say so before launch and pass `--headless`; never switch to headless silently.

Use the requested episode count; if omitted, state that this is a one-episode smoke check and set `N=1`. The runner runs exactly `N` episodes. Each ends on the scene's success termination or times out at the manifest's `max_steps`; there are no retries. Do not rerun episodes to collect successes. When the user asked for `N` successful episodes and fewer succeeded, report the shortfall.

Create one run directory per rollout; `run.sh` writes the runner's results, the recording, and the server log into it:

```bash
WORKFLOW=<workflow>
RUN_DIR="$PWD/runs/$WORKFLOW/$(date +%Y%m%d_%H%M%S)"
```

Rule-based, for example `surgical_reach_psm`:

```bash
./run.sh "$WORKFLOW" --rule-based --episodes <N> --run-dir "$RUN_DIR" --record "$RUN_DIR/verify.hdf5"
```

The workflow declaration carries each workflow's policy flags and launcher options, such as `--duration_s 0.01` for `surgical_reach_star`, `--position_tolerance 0.07` for `surgical_lift_needle_organs`, Newton for the endoluminal and tablecloth scenes, and `PXR_WORK_THREAD_LIMIT=1` for tablecloth. For “with Newton physics,” add `--presets newton`; never fall back to PhysX silently. `run.sh` enables cameras when the scene declares them; `--no-cameras` turns them off.

Served policy, for example GR00T N1.5 on `scissor_pick_and_place`:

```bash
./run.sh scissor_pick_and_place --policy --episodes <N> --run-dir "$RUN_DIR" --record "$RUN_DIR/verify.hdf5"
```

`run.sh` starts the stack's `serve` module on a free port near the stack's default (5555 for GR00T, 8000 for PI0), writes its log to `$RUN_DIR/server.log`, and starts the simulator once the server listens. Each server binds its port only after its checkpoint has loaded, and the first start may download it; `I4H_SERVER_TIMEOUT_S` (default 3600) bounds the wait. A server that exits before it listens stops the run; report the first error in `server.log`. To use a server that is already running, pass `--policy-endpoint HOST:PORT` instead.

For “300 timesteps,” pass `--episode-steps 300`. It replaces the manifest's `max_steps`, so use it only to lower the budget; never raise the validated budget. If the request exceeds `max_steps`, say so and stop. Remote inference waits do not consume simulation steps.

### Supplied-checkpoint preflight

Resolve the supplied checkpoint to an existing absolute path and pass it with `--checkpoint`. Identify the stack and task that serve it from `./run.sh show`. Read the task manifest under `tasks/<stack>/i4h_tasks/<stack>/manifest/` and the Arena client's layout (`LAYOUTS` in `arena/i4h_arena/policies/gr00t_n15.py` or `gr00t_n17.py`, the adapter in `pi0.py`), and compare the checkpoint's model family, embodiment tag, cameras, state and action groups, and widths. Stop if the path or contract is unresolved; do not infer compatibility from a filename.

`run.sh` hands the checkpoint to the stack's server (`--model-path` for GR00T, `--checkpoint` for PI0) and starts the simulator only after the server has loaded it, so a checkpoint that fails to load stops the run before the simulator starts. For example, the `endoluminal_navigation` N1.7 mode, whose task names no published checkpoint:

```bash
./run.sh endoluminal_navigation --mode policy_n17 --checkpoint /absolute/path/to/checkpoint \
  --patient-twin /absolute/path/to/patient_twin.yaml --episodes <N> --run-dir "$RUN_DIR" --record "$RUN_DIR/verify.hdf5"
```

An exported RLinf Trocar checkpoint goes to `./run.sh assemble_trocar --policy --checkpoint`. For another task, resolve its own stack, server, and task from `./run.sh show`; do not reuse the example's model family. Successful loading alone does not establish environment compatibility: the first policy steps must also pass the client's observation and action mapping.

An RSL-RL checkpoint has no server. Arena's `rsl_rl` policy loads it in-process and reads `params/agent.yaml` beside it, so pass a `model_<N>.pt` from the training run's timestamped Isaac Lab log directory, not `model_final.pt` at the run root or an exported TorchScript `policy.pt`. The training run's `run.json` lists those checkpoints in order. The `ultrasound_probe_reach` policy mode selects the training observations:

```bash
TRAIN_RUN=/absolute/path/to/runs/ultrasound_probe_reach/YYYYMMDD_HHMMSS
CHECKPOINT="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoints"][-1])' "$TRAIN_RUN/run.json")"
test -f "$(dirname "$CHECKPOINT")/params/agent.yaml"
./run.sh ultrasound_probe_reach --policy --checkpoint "$CHECKPOINT" \
  --episodes <N> --run-dir "$RUN_DIR" --record "$RUN_DIR/verify.hdf5"
```

Require the checkpoint to load and the first policy step to succeed before reporting it launchable.

## Verify

Require `run.sh` exit status 0. The exit status does not report episode success; `run.sh` prints `<k>/<N> episodes succeeded`, or that the scene has no success term. Read each episode's `success` and `episode_length` from the results file, and the `Metrics:` line when the scene defines metrics:

```bash
python3 - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for path in Path(sys.argv[1]).glob("outputs/*/episode_results_rank0.jsonl")
        for line in path.read_text().splitlines() if line.strip()]
print(f"{sum(bool(row.get('success')) for row in rows)}/{len(rows)} episodes succeeded;",
      "lengths", [row.get("episode_length") for row in rows])
PY
uv run --project tools/dataset i4h-dataset inspect "$RUN_DIR/verify.hdf5"
```

Require one result per requested episode. The recording keeps every episode with its success flag: check its demo count, success flags, action width, and declared cameras against the results, the scene manifest, and the embodiment manifest under `arena/i4h_arena/embodiments/manifest/`. Missing, corrupt, or inconsistent data blocks rollout validation; report the failed check.

Scenes without a success term (tablecloth, and endoluminal without a patient twin) report `success: null` in the results and `False` in the recording: say so and judge the run on completion and visible behavior. The fluoroscopy check prints `[fluoroscopy] C-arm sweep PASS` or `FAIL`. For visible runs, observe scene and camera behavior and the final task outcome. On failure, use the first actionable server, client contract, or simulator error; never switch policies or raise the step budget silently. After a crash, confirm `.run/pids.d/` holds no entry for the workflow, or stop what remains with `./stop.sh <workflow>`.

After authoring or changing a collision-excluding success rule, use the scene's contact setup to test the configured body pair (see `skills/i4h-workflow-scene-edit/references/g1-reach-and-contact.md`). Require non-zero filtered force, rejected success, and cleared collision history after reset. Record sensor names, force-matrix shape, maximum force, and outcome in the saved evidence. Initialization or zero force alone is insufficient. If the test cannot run, report the validation gap.

## Troubleshooting

Correct the first actionable error within scope, then rerun the whole rollout once into a new run directory with the same mode, step budget, and episode count. If it recurs or cannot be fixed, stop and report it with the run directory and server log.

## Prerequisites

Require synced simulator assets, the `common` and `arena` environments, the policy stack's environment and checkpoint access for a served policy (the N1.7 backbone is gated on Hugging Face), and a patient twin for endoluminal goals and fluoroscopy.

## Limitations

Only workflows `./run.sh list` shows, with a mode their declaration gives, are supported. The tablecloth scenes have no autonomous policy, `ultrasound_probe_reach` needs a trained RSL-RL checkpoint, and the catheter policy needs a supplied checkpoint and patient twin. `--episode-steps` may lower but never raise the validated budget.

## Examples

- `Evaluate scissor pick and place for 2 episodes.` → run `./run.sh scissor_pick_and_place --policy --episodes 2` with a run directory and recording, which serves the GR00T N1.5 scissors checkpoint and stops it afterwards, and report successes out of two with the visible outcome.
- `Run surgical_reach_psm in rule-based mode for 1 episode.` → run `./run.sh surgical_reach_psm --rule-based --episodes 1` with a run directory and recording.

## Completion gate

Report workflow, mode, environment class and scene, policy class and flags and server command (from `./run.sh show`), model/checkpoint source, server log when served, requested and succeeded episodes with episode lengths, step budget, visible outcome, run directory, results file, HDF5 path and inspection, collision-negative/reset evidence when applicable, readiness level reached, final exit status, and first unresolved failure if any.
