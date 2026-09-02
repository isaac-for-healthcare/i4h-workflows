#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

WORKFLOWS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKFLOWS_ROOT"
export I4H_WORKFLOWS="$WORKFLOWS_ROOT"

ENV_ID=scissor_pick_and_place
DRY_RUN=0
SKIP_MIMIC=0
SKIP_ANNOTATE=0
SKIP_REPLAY=0
SKIP_VIZ=0
RUN_DIR=""
RECORD_EPISODES="${RECORD_EPISODES:-3}"
RECORD_ATTEMPTS="${RECORD_ATTEMPTS:-3}"
MIMIC_EPISODES="${MIMIC_EPISODES:-3}"
FINETUNE_STEPS="${FINETUNE_STEPS:-500}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-32}"

usage() {
  cat <<EOF
Usage: scripts/e2e/run.sh [ENV] [options]

Runs setup -> policy record -> mimic -> annotate/filter -> replay -> convert ->
visualize -> finetune -> validate.

Options:
  --env ENV              workflow name (default: scissor_pick_and_place)
  --run-dir PATH         output directory
  --dry-run              resolve and print every stage without executing it
  --skip-mimic           use policy recordings directly
  --skip-annotate        trust simulator success and skip VLM filtering
  --skip-replay          skip visual replay
  --skip-viz             skip the LeRobot HTML server
  -h, --help
EOF
}

while (($#)); do
  case "$1" in
    --env) ENV_ID="$2"; shift 2 ;;
    --env=*) ENV_ID="${1#*=}"; shift ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --run-dir=*) RUN_DIR="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-mimic) SKIP_MIMIC=1; shift ;;
    --skip-annotate) SKIP_ANNOTATE=1; shift ;;
    --skip-replay) SKIP_REPLAY=1; shift ;;
    --skip-viz) SKIP_VIZ=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) ENV_ID="$1"; shift ;;
  esac
done

LIGHT=(env -u VIRTUAL_ENV uv run --project workflows --no-sync)
if ! META_TEXT=$("${LIGHT[@]}" python - "$ENV_ID" <<'PY'
import sys
from i4h_engine.loader import resolve_workflow
from i4h_engine.registry import default_registry

name = sys.argv[1]
try:
    workflow = resolve_workflow(name, "policy")
except KeyError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(2) from None
registry = default_registry()
remote = []
for node in workflow.graph.nodes:
    spec = node.spec or registry.tasks.get(node.task_id)
    if spec is not None and spec.runtime == "remote":
        remote.append(spec)
if len(remote) != 1:
    raise SystemExit(f"{name}: e2e needs exactly one policy task, found {len(remote)}")
spec = remote[0]
print(spec.project)
print(spec.id)
print(spec.requires.get("embodiment", ""))
print(spec.effective_prompt)
print("1" if spec.trainable else "0")
PY
); then
  echo "[e2e] $ENV_ID does not provide a policy mode; use ./run.sh $ENV_ID --rule-based" >&2
  exit 2
fi
readarray -t META <<<"$META_TEXT"

STACK="${META[0]}"
TASK_ID="${META[1]}"
ROBOT="${META[2]}"
TASK_DESCRIPTION="${META[3]}"
TRAINABLE="${META[4]}"

RUN_DIR="${RUN_DIR:-runs/${ENV_ID}/$(date +%Y%m%d_%H%M%S)}"
case "$RUN_DIR" in /*) ;; *) RUN_DIR="$WORKFLOWS_ROOT/$RUN_DIR" ;; esac
DATA_DIR="$RUN_DIR/data"
LOG_DIR="$RUN_DIR/logs"
RECORDING="$DATA_DIR/recording.hdf5"
EXPANDED="$DATA_DIR/expanded.hdf5"
FILTERED="$DATA_DIR/filtered.hdf5"
DATASET="$RUN_DIR/lerobot/local/${ENV_ID}_e2e"
CHECKPOINTS="$RUN_DIR/checkpoints"
VERIFY="$DATA_DIR/verify.hdf5"
VIZ_STATE="$RUN_DIR/viz-state"
VLLM_CONTAINER="${I4H_ANNOTATOR_VLLM_CONTAINER:-i4h-workflows-annotator-vllm}"
VLLM_OWNED=0

stage() { printf '\n========== %s ==========\n' "$1"; }
run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  [ "$DRY_RUN" -eq 1 ] || "$@"
}
cleanup() {
  if [ "$VLLM_OWNED" -eq 1 ]; then
    tools/annotator/scripts/vllm.sh stop >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

cat <<EOF
[e2e] env=$ENV_ID stack=$STACK task=$TASK_ID robot=$ROBOT trainable=$TRAINABLE
[e2e] run_dir=$RUN_DIR
EOF

if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$DATA_DIR" "$LOG_DIR"
  exec > >(tee -a "$LOG_DIR/workflow.log") 2>&1
  echo "[e2e] log=$LOG_DIR/workflow.log"
fi

stage setup
run ./setup.sh

stage record
run ./run.sh "$ENV_ID" --policy --episodes "$RECORD_EPISODES" --attempts "$RECORD_ATTEMPTS" \
  --run-dir "$RUN_DIR/record" --record "$RECORDING"

stage mimic
if [ "$SKIP_MIMIC" -eq 1 ]; then
  run cp "$RECORDING" "$EXPANDED"
else
  run uv run --project tools/mimic i4h-mimic "$RECORDING" "$EXPANDED" \
    --episodes "$MIMIC_EPISODES" --include-source --successful-only
fi

stage annotate
if [ "$SKIP_ANNOTATE" -eq 1 ]; then
  run cp "$EXPANDED" "$FILTERED"
else
  if [ "$DRY_RUN" -eq 0 ] && ! docker ps --filter "name=^/${VLLM_CONTAINER}$" --format '{{.ID}}' | grep -q .; then
    VLLM_OWNED=1
  fi
  run tools/annotator/scripts/vllm.sh ensure
  run uv run --project tools/annotator i4h-annotator --task "$TASK_DESCRIPTION" \
    offline "$EXPANDED" --write --filter "$FILTERED"
  if [ "$VLLM_OWNED" -eq 1 ]; then
    run tools/annotator/scripts/vllm.sh stop
    VLLM_OWNED=0
  fi
fi

stage replay
[ "$SKIP_REPLAY" -eq 1 ] || run ./run.sh "$ENV_ID" --replay "$FILTERED" --episode 0 --run-dir "$RUN_DIR/replay"

stage convert
run uv run --project tools/dataset i4h-dataset convert "$FILTERED" "$DATASET" \
  --robot "$ROBOT" --repo-id "local/${ENV_ID}_e2e" --successful-only --task "$TASK_DESCRIPTION"

stage visualize
[ "$SKIP_VIZ" -eq 1 ] || run tools/dataset/scripts/viz.sh "$DATASET" --state-dir "$VIZ_STATE"

stage finetune
if [ "$TRAINABLE" -eq 0 ]; then
  echo "[e2e] $TASK_ID is inference-only; skipping finetune and checkpoint validation"
else
  run uv run --project "tasks/$STACK" "i4h-tasks-${STACK//_/-}-train" \
    --task "$TASK_ID" --dataset "$DATASET" --output-dir "$CHECKPOINTS" \
    --max-steps "$FINETUNE_STEPS" --save-steps "$FINETUNE_STEPS" \
    --batch-size "$FINETUNE_BATCH_SIZE"

  if [ "$DRY_RUN" -eq 1 ]; then
    CHECKPOINT="<latest-checkpoint>"
  else
    CHECKPOINT="$(find "$CHECKPOINTS" -type d \( -name 'checkpoint-*' -o -path '*/finetune/*' \) -printf '%T@ %p\n' \
      | sort -nr | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print}')"
    if [ -z "$CHECKPOINT" ]; then
      echo "[e2e] no checkpoint found under $CHECKPOINTS" >&2
      exit 1
    fi
  fi

  stage validate
  run ./run.sh "$ENV_ID" --policy --checkpoint "$CHECKPOINT" --episodes 1 --attempts 3 \
    --headless --run-dir "$RUN_DIR/validate" --record "$VERIFY"
fi

stage summary
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[e2e] dry run: no stages executed"
else
  echo "[e2e] complete: $RUN_DIR"
fi
