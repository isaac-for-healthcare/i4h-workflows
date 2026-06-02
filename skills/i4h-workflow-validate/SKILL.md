---
name: i4h-workflow-validate
version: "0.6.0"
description: Roll out a policy against an env and record verification episodes. Use when the user asks to validate, evaluate, or rollout a policy or checkpoint.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - validation
    - policy-rollout
---

# i4h Workflow — Validate

## Purpose

Roll out a policy against an env and record verification episodes to an HDF5. Use when the user asks to validate, evaluate, or rollout a policy or checkpoint.

## Base Code

These steps drive the i4h-workflows base code (the `workflows/agentic/` tree). To reuse an existing checkout, set `I4H_WORKFLOWS` to its path (no clone happens). Otherwise this resolves the current repo, or clones to `~/i4h-workflows` — pick that default without prompting. Run every command below from the resolved root:

```bash
# Resolve the i4h-workflows base code (provides workflows/agentic/).
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/agentic" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/agentic" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

- **Env config (source of truth):** `workflows/agentic/config/environments/<env>.yaml` — read it for the `<env>` defaults: `policy.model_repo`/`model_revision`, `policy.task_description`, `policy.health_port`, and `arena.max_timesteps`.
- Validation runs the policy daemon and Arena together; both processes are required.
- The policy daemon is headless. Arena is the only process that opens the sim window.
- Do not run the VLM annotator unless the user asks for success labels.
- `assemble_trocar` is inference-only — validate its YAML default model or a compatible N1.5 checkpoint.

## Inputs

- `ENV_ID`: env YAML id.
- `EPISODES`: `1` for sanity, more for real eval.
- `MAX_TIMESTEPS`: `200` for sanity. Use env YAML defaults (1500 locomanip / 500 scissor / 250 ultrasound) only for real success-rate measurement.
- `MODEL_PATH` (optional): path to a `checkpoint-NNNN/` directory containing `model-0000{N}-of-*.safetensors`, `experiment_cfg/`, and `processor/`. Omit to use YAML `policy.model_repo`.

## Run

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
EPISODES=1
MAX_TIMESTEPS=200
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/eval_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"

"${REPO_ROOT}/workflows/agentic/policy/run.sh" --env "${ENV_ID}" \
  > "${RUN_DIR}/logs/policy.log" 2>&1 &
POLICY_PID=$!

until grep -qE "policy ready|Traceback|Error|FAILED" "${RUN_DIR}/logs/policy.log" 2>/dev/null; do sleep 2; done
grep -qE "Traceback|Error|FAILED" "${RUN_DIR}/logs/policy.log" && {
  tail -30 "${RUN_DIR}/logs/policy.log"
  kill "${POLICY_PID}" 2>/dev/null
  exit 1
}

"${REPO_ROOT}/workflows/agentic/arena/run.sh" --env "${ENV_ID}" \
  --episodes "${EPISODES}" \
  --max-timesteps "${MAX_TIMESTEPS}" \
  --max-attempts 1 \
  --record-to "${RUN_DIR}/data/verify.hdf5" \
  2>&1 | tee "${RUN_DIR}/logs/arena.log"

"${REPO_ROOT}/workflows/agentic/stop.sh" policy --env "${ENV_ID}" || kill "${POLICY_PID}" 2>/dev/null || true
```

For a checkpoint, append `--model-path "${MODEL_PATH}"` to the policy launch.

## Notes

- Launch the policy daemon first, wait for `policy ready`, then launch Arena.
- `--record-to` must be absolute. The recorder resolves relative paths against `workflows/agentic/arena` (its CWD) and produces a nested orphan dir.
- `--max-attempts` defaults to 1 for locomanip-family envs.

## Optional Annotation

Run only on request:

```bash
"${REPO_ROOT}/workflows/agentic/annotator/run.sh" \
  --env "${ENV_ID}" \
  --output "${RUN_DIR}/annotations.jsonl" \
  offline \
  --hdf5-path "${RUN_DIR}/data/verify.hdf5"
```

## Verify

- `verify.hdf5` exists under `${RUN_DIR}/data/`.
- Arena log shows `run complete: N/M episodes succeeded`.
- Policy log contains no `Traceback`.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (`.venv` present); the `policy/run.sh` and `arena/run.sh` launches depend on it.
- An `ENV_ID` matching an env YAML id.
- A model source: either the env YAML `policy.model_repo` default, or a `MODEL_PATH` pointing at a `checkpoint-NNNN/` dir (`model-0000{N}-of-*.safetensors`, `experiment_cfg/`, `processor/`).

## Limitations

- Both the policy daemon and Arena are required; the daemon is headless and Arena is the only process that opens the sim window.
- `assemble_trocar` is inference-only — validate its YAML default model or a compatible N1.5 checkpoint.
- `--record-to` must be absolute; relative paths resolve against `workflows/agentic/arena` and produce a nested orphan dir.
- The VLM annotator is optional and run only on request; it is not part of the default rollout.

## Troubleshooting

- **Error:** `.venv` / import fails or `run.sh` missing - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** policy log shows `Traceback` / `Error` / `FAILED` before `policy ready` - Cause: the policy daemon failed to start (e.g. bad model source). Fix: inspect `${RUN_DIR}/logs/policy.log`; verify `ENV_ID` / `MODEL_PATH`.
- **Error:** Arena starts before the daemon is ready - Cause: launch order. Fix: launch the policy daemon first and wait for `policy ready`, then launch Arena.
- **Error:** `verify.hdf5` lands in a nested orphan dir - Cause: relative `--record-to`. Fix: pass an absolute path under `${RUN_DIR}/data/`.

## Final Response

Report env, model source, episodes saved vs requested, HDF5 path, log paths.
