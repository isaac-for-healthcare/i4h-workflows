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

Run the selected workflow's IsaacLab-Arena environment under Arena's policy runner with a rule-based or served policy, read the per-episode results and the recording, and report simulator success.

## Instructions

1. Resolve the base checkout, the workflow's environment, and its policy.
2. For a served policy, start its server, wait for its ready line, and keep its PID.
3. Run the policy runner in the foreground into a run directory.
4. Read every episode's result, inspect the recording, and stop the server by its PID; for an authored or changed collision-excluding success rule, also verify the collision-negative case below.

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

Check that the reported origin is the intended source and review local changes before executing repository scripts. Stop on an unexpected source or unreviewed changes to the environments, policies, or servers. Then list the scenes:

```bash
ls arena/i4h_arena/scenes/manifest
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Workflow ids such as `scissor_pick_and_place` stay the user-facing names. The specialty pages in `docs/workflows/` (`laparoscopic-robotics.md`, `ultrasound-robotics.md`, `endoluminal-robotics.md`, `hospital-automation-robotics.md`, `tablecloth.md`) map each workflow to its environment class and scene (`SoarmScissors`, `soarm_scissors`), its rule-based policy class with any per-workflow flags, and its served policy: the server command and the Arena client with its flags. Treat those pages as the authoritative table; do not duplicate it here. Map the user's natural name to a listed workflow and choose only a policy the page gives for it:

| User intent | Policy |
|---|---|
| Ordinary learned-policy evaluation ("policy mode") | The page's first served policy and its server; for `scissor_pick_and_place` that is GR00T N1.5. An RL-trained scene such as `ultrasound_probe_reach` uses its RSL-RL checkpoint (below) |
| Requested or only available local controller ("rule-based mode") | The page's rule-based class under `i4h_arena.policies` with the flags the page gives it |
| Explicit named alternative such as N1.7 | That stack's server and client from the same page |
| Look at a scene without a controller | `i4h_arena.policies.hold.HoldPolicy` |

Read the scene manifest `arena/i4h_arena/scenes/manifest/<scene>.yaml` (step budget `max_steps`, cameras), the policy class under `arena/i4h_arena/policies/`, and the task manifest under `tasks/<stack>/i4h_tasks/<stack>/manifest/` when model, checkpoint, cameras, or step budget matter. The runner gives the policy the environment's task description as its prompt.

Use precise readiness language:

- **Resolved**: the environment, policy class, and server task exist in the checkout (for example, the serve module's `--help` lists the task).
- **Launchable**: the policy server reports ready with the selected checkpoint and the runner builds the environment and steps the policy.
- **Rollout-validated**: the requested episodes complete and their results and recording pass inspection.

Do not report “validated” without stating which level was actually reached.

## Foreground execution rule

Keep the policy runner as this agent's foreground tool call. Do not use a subagent, monitor task, `nohup`, `tmux`, or a detached rollout. The only background process is a served policy's server, started in the same foreground command as the rollout with its PID kept and stopped by that PID afterwards. Poll a yielded session until exit and read the episode results before responding. Never stop a process by tool or module name: other sessions run the same servers and runners.

Run visibly by default with `--viz kit`. If the user explicitly requests headless execution, or a documented environment constraint makes it necessary, say so before launch and omit `--viz kit`; never switch to headless silently.

Use the requested episode count; if omitted, state that this is a one-episode smoke check and set `N=1`. The runner runs exactly `N` episodes. Each ends on the scene's success termination or times out at the manifest's `max_steps`; there are no retries. Do not rerun episodes to collect successes. When the user asked for `N` successful episodes and fewer succeeded, report the shortfall.

Create one run directory per rollout and pass it to both the runner and the recorder:

```bash
WORKFLOW=<workflow>
RUN_DIR="$PWD/runs/$WORKFLOW/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
```

Launcher and policy flags go before the scene name; environment flags (`--episode_steps`, recording paths, scene flags such as `--patient_twin` or `--rl_observations`) go after it.

Rule-based, for example `surgical_reach_psm`:

```bash
arena/.venv/bin/python -m isaaclab_arena.evaluation.policy_runner \
  --policy_type i4h_arena.policies.reach.CartesianReachPolicy \
  --num_episodes <N> --enable_cameras --viz kit --output_base_dir "$RUN_DIR/outputs" \
  --external_environment_class_path i4h_arena.environments:PsmReach psm_reach \
  --recorder_dataset_export_dir_path "$RUN_DIR" --recorder_dataset_filename verify
```

Copy each workflow's policy class, flags, and launcher options from its page, such as `--duration_s 0.01` for `surgical_reach_star`, `--position_tolerance 0.07` for `surgical_lift_needle_organs`, `--presets newton` for the endoluminal and tablecloth scenes, and `PXR_WORK_THREAD_LIMIT=1` for tablecloth. For “with Newton physics,” add `--presets newton` before the scene name; never fall back to PhysX silently. Add `--enable_cameras` whenever the policy reads cameras or the recording should hold camera frames.

Served policy, for example GR00T N1.5 on `scissor_pick_and_place`. Run the block as one foreground command so the server's PID stays in scope:

```bash
SERVER_LOG="$RUN_DIR/server.log"
PYTHONUNBUFFERED=1 tasks/gr00t_n15/.venv/bin/python -m i4h_tasks.gr00t_n15.serve scissor_pick_and_place \
  --port 5555 >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "policy server pid $SERVER_PID"
READY=0
for _ in $(seq 360); do
  if grep -q "Server is ready" "$SERVER_LOG"; then READY=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 5
done
if [ "$READY" = 1 ]; then
  arena/.venv/bin/python -m isaaclab_arena.evaluation.policy_runner \
    --policy_type i4h_arena.policies.gr00t_n15.Gr00tN15RemotePolicy --embodiment so101_scissors \
    --remote_port 5555 --num_episodes <N> --enable_cameras --viz kit --output_base_dir "$RUN_DIR/outputs" \
    --external_environment_class_path i4h_arena.environments:SoarmScissors soarm_scissors \
    --recorder_dataset_export_dir_path "$RUN_DIR" --recorder_dataset_filename verify
  echo "policy runner exit $?"
else
  tail -n 40 "$SERVER_LOG"
fi
kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
```

Swap in the server and client from the workflow's page. The GR00T servers (N1.5, N1.6, N1.7) print `Server is ready`; the PI0 server compiles with a warm-up inference and then logs `server listening on`. Keep `PYTHONUNBUFFERED=1`, or a printed ready line can sit in Python's output buffer instead of the log. The first start may download the checkpoint, so allow it time.

The PI0 client defaults to port 8000 and the GR00T clients to 5555; if another session holds the port (`ss -ltn`), pass another one to both `--port` and `--remote_port`. GR00T N1.6 runs GR00T's own `run_gr00t_server.py` from the `tasks/gr00t_n16` environment with Arena's GR00T client, exactly as the hospital page shows. If the N1.5 client's first inference times out, raise its `--timeout_s`. If the block is interrupted, stop the server with `kill <printed pid>`.

For “300 timesteps,” pass `--episode_steps 300` after the scene name. It replaces the manifest's `max_steps`, so use it only to lower the budget; never raise the validated budget. If the request exceeds `max_steps`, say so and stop. Remote inference waits do not consume simulation steps.

### Supplied-checkpoint preflight

Resolve the supplied checkpoint to an existing absolute path. Identify the stack and task that serve it from the workflow's page. Read the task manifest under `tasks/<stack>/i4h_tasks/<stack>/manifest/` and the Arena client's layout (`LAYOUTS` in `arena/i4h_arena/policies/gr00t_n15.py` or `gr00t_n17.py`, the adapter in `pi0.py`), and compare the checkpoint's model family, embodiment tag, cameras, state and action groups, and widths. Stop if the path or contract is unresolved; do not infer compatibility from a filename.

Load-check a served checkpoint before starting a simulator: start its stack's `serve` module with the checkpoint in the background as above, wait for the ready line, then stop it by its PID. For example, the `gr00t_n17/catheter_navigation` task uses:

```bash
PYTHONUNBUFFERED=1 tasks/gr00t_n17/.venv/bin/python -m i4h_tasks.gr00t_n17.serve catheter_navigation \
  --model-path /absolute/path/to/checkpoint --port 5555 >"$RUN_DIR/load-check.log" 2>&1 &
```

`Server is ready` confirms the checkpoint loaded; the specialty page gives the matching Arena rollout. A server that exits before its ready line failed to load: report the first error in its log. Pass a GR00T checkpoint with `--model-path` (including an exported RLinf Trocar checkpoint to `gr00t_n15.serve assemble_trocar`) and a PI0 checkpoint with `--checkpoint`. For another task, resolve its own stack, server, and task from the live declarations; do not reuse the example's model family. Successful loading alone does not establish environment compatibility: the first policy steps must also pass the client's observation and action mapping.

An RSL-RL checkpoint has no server. Arena's `rsl_rl` policy loads it in-process and reads `params/agent.yaml` beside it, so pass a `model_<N>.pt` from the training run's timestamped Isaac Lab log directory, not `model_final.pt` at the run root or an exported TorchScript `policy.pt`. The training run's `run.json` lists those checkpoints in order. Select the environment's training observations with its scene flag, `--rl_observations` for `ultrasound_probe_reach`:

```bash
TRAIN_RUN=/absolute/path/to/runs/ultrasound_probe_reach/YYYYMMDD_HHMMSS
CHECKPOINT="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoints"][-1])' "$TRAIN_RUN/run.json")"
test -f "$(dirname "$CHECKPOINT")/params/agent.yaml"
arena/.venv/bin/python -m isaaclab_arena.evaluation.policy_runner \
  --policy_type rsl_rl --checkpoint_path "$CHECKPOINT" \
  --num_episodes <N> --viz kit --output_base_dir "$RUN_DIR/outputs" \
  --external_environment_class_path i4h_arena.environments:UltrasoundProbeReach ultrasound_probe_reach \
  --rl_observations --recorder_dataset_export_dir_path "$RUN_DIR" --recorder_dataset_filename verify
```

Require the checkpoint to load and the first policy step to succeed before reporting it launchable.

## Verify

Require policy-runner exit status 0. The exit status does not report episode success; read each episode's `success` and `episode_length` from the results file, and the `Metrics:` line when the scene defines metrics:

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

Scenes without a success term (tablecloth, and endoluminal without a patient twin) report `success: null` in the results and `False` in the recording: say so and judge the run on completion and visible behavior. The fluoroscopy check prints `[fluoroscopy] C-arm sweep PASS` or `FAIL`. For visible runs, observe scene and camera behavior and the final task outcome. On failure, use the first actionable server, client contract, or simulator error; never switch policies or raise the step budget silently. After a crash, confirm the server and runner PIDs have exited.

After authoring or changing a collision-excluding success rule, use the scene's contact setup to test the configured body pair (see `skills/i4h-workflow-scene-edit/references/g1-reach-and-contact.md`). Require non-zero filtered force, rejected success, and cleared collision history after reset. Record sensor names, force-matrix shape, maximum force, and outcome in the saved evidence. Initialization or zero force alone is insufficient. If the test cannot run, report the validation gap.

## Troubleshooting

Correct the first actionable error within scope, then rerun the whole rollout once into a new run directory with the same policy, step budget, and episode count. If it recurs or cannot be fixed, stop and report it with the run directory and server log.

## Prerequisites

Require synced simulator assets, the `arena` environment, the policy stack's environment and checkpoint access for a served policy (the N1.7 backbone is gated on Hugging Face), and a patient twin for endoluminal goals and fluoroscopy.

## Limitations

Only scenes under `arena/i4h_arena/scenes/manifest` with a policy their specialty page documents are supported. The tablecloth scenes have no autonomous policy, `ultrasound_probe_reach` needs a trained RSL-RL checkpoint, and the catheter policy needs a supplied checkpoint and patient twin. `--episode_steps` may lower but never raise the validated budget.

## Examples

- `Evaluate scissor pick and place for 2 episodes.` → serve the GR00T N1.5 scissors checkpoint, roll out `SoarmScissors soarm_scissors` for two recorded episodes, stop the server by PID, and report successes out of two with the visible outcome.
- `Run surgical_reach_psm in rule-based mode for 1 episode.` → run `CartesianReachPolicy` in `PsmReach psm_reach` for one recorded episode.

## Completion gate

Report workflow, environment class and scene, policy class and flags, server command, ready line and PID when served, model/checkpoint source, requested and succeeded episodes with episode lengths, step budget, visible outcome, run directory, results file, HDF5 path and inspection, collision-negative/reset evidence when applicable, server stopped, readiness level reached, final exit status, and first unresolved failure if any.
