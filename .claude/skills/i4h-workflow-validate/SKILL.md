---
name: i4h-workflow-validate
description: Roll out a policy against an env and record verification episodes. Use when the user asks to validate, evaluate, or rollout a policy or checkpoint.
---

# i4h Workflow — Validate

## Basics

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
REPO_ROOT="$(git rev-parse --show-toplevel)"
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

## Final Response

Report env, model source, episodes saved vs requested, HDF5 path, log paths.
