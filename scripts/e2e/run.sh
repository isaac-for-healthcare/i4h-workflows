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
HEADLESS=()
RUN_DIR=""
RECORD_EPISODES="${RECORD_EPISODES:-3}"
MIMIC_EPISODES="${MIMIC_EPISODES:-3}"
FINETUNE_STEPS="${FINETUNE_STEPS:-500}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-32}"
VALIDATE_EPISODES="${VALIDATE_EPISODES:-3}"

usage() {
  cat <<EOF
Usage: scripts/e2e/run.sh [options]

Runs setup -> policy record -> mimic -> annotate/filter -> replay -> convert ->
visualize -> finetune -> validate. The simulator stages run the workflow with
./run.sh: its policy mode under Arena's policy runner, against the policy
server of the workflow's checkpoint, and replay with Isaac Lab's replay_demos.py.

Options:
  --env ENV              workflow name: scissor_pick_and_place (default) or ultrasound_liver_scan
  --run-dir PATH         output directory
  --dry-run              resolve and print every stage without executing it
  --skip-mimic           use policy recordings directly
  --skip-annotate        trust simulator success and skip VLM filtering
  --skip-replay          skip visual replay
  --skip-viz             skip the LeRobot HTML server
  --headless             record and replay without the interactive viewport
                         (checkpoint validation always runs headless)
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
    --headless) HEADLESS=(--headless); shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) ENV_ID="$1"; shift ;;
  esac
done

# Each workflow's scene and the policy task its policy mode serves
# (workflows/<specialty>/<workflow>.yaml).
case "$ENV_ID" in
  scissor_pick_and_place)
    SCENE=soarm_scissors
    TASK_ID=gr00t_n15/scissor_pick_and_place
    ;;
  ultrasound_liver_scan)
    SCENE=panda_phantom
    TASK_ID=openpi_pi0/ultrasound_liver_scan
    ;;
  *)
    echo "[e2e] $ENV_ID has no e2e pipeline; supported: scissor_pick_and_place, ultrasound_liver_scan" >&2
    exit 2
    ;;
esac
STACK="${TASK_ID%%/*}"

LIGHT=(env -u VIRTUAL_ENV uv run --project common --no-sync)
META_TEXT=$("${LIGHT[@]}" python - "$TASK_ID" "$SCENE" <<'PY'
import sys
from pathlib import Path

from i4h_common.manifest import load_scene_spec
from i4h_common.training import task_spec

spec = task_spec(sys.argv[1])
scene = load_scene_spec(Path("arena/i4h_arena/scenes/manifest") / f"{sys.argv[2]}.yaml")
# The annotate stage grades against the scene's visible success criterion when it
# declares one; the policy's language prompt may name no end state.
annotate = scene.annotate
prompt = " ".join((spec.prompt or spec.summary).split())
print(spec.embodiment)
print(prompt)
print("1" if spec.trainable else "0")
print((annotate and annotate.criterion) or prompt)
print(",".join(annotate.cameras) if annotate else "")
print(annotate.judge if annotate else "")
print((annotate and annotate.frames) or "")
PY
)
readarray -t META <<<"$META_TEXT"

ROBOT="${META[0]}"
TASK_DESCRIPTION="${META[1]}"
TRAINABLE="${META[2]}"
ANNOTATE_CRITERION="${META[3]}"
ANNOTATE_CAMERAS="${META[4]:-}"
ANNOTATE_JUDGE="${META[5]:-}"
ANNOTATE_FRAMES="${META[6]:-}"
ANNOTATE_ARGS=(--task "$ANNOTATE_CRITERION")
[ -z "$ANNOTATE_CAMERAS" ] || ANNOTATE_ARGS+=(--camera "$ANNOTATE_CAMERAS")
[ -z "$ANNOTATE_JUDGE" ] || ANNOTATE_ARGS+=(--judge "$ANNOTATE_JUDGE")
[ -z "$ANNOTATE_FRAMES" ] || ANNOTATE_ARGS+=(--frames "$ANNOTATE_FRAMES")

RUN_DIR="${RUN_DIR:-runs/${ENV_ID}/$(date +%Y%m%d_%H%M%S)}"
case "$RUN_DIR" in /*) ;; *) RUN_DIR="$WORKFLOWS_ROOT/$RUN_DIR" ;; esac
DATA_DIR="$RUN_DIR/data"
LOG_DIR="$RUN_DIR/logs"
RECORDING="$DATA_DIR/recording.hdf5"
EXPANDED="$DATA_DIR/expanded.hdf5"
FILTERED="$DATA_DIR/filtered.hdf5"
DATASET="$RUN_DIR/lerobot/local/${ENV_ID}_e2e"
CHECKPOINTS="$RUN_DIR/checkpoints"
VERIFY_DIR="$RUN_DIR/validate"
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
# Fail the stage unless at least one episode of the run in $1 succeeded.
require_success() {
  printf '+ require a successful episode under %q\n' "$1"
  [ "$DRY_RUN" -eq 1 ] && return
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

results = [json.loads(line) for path in Path(sys.argv[1]).glob("outputs/*/episode_results_rank0.jsonl") for line in path.read_text().splitlines() if line.strip()]
successes = sum(bool(result.get("success")) for result in results)
print(f"[e2e] {successes}/{len(results)} validation episodes succeeded")
raise SystemExit(0 if successes else 1)
PY
}
cleanup() {
  if [ "$VLLM_OWNED" -eq 1 ]; then
    tools/annotator/scripts/vllm.sh stop >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

cat <<EOF
[e2e] env=$ENV_ID scene=$SCENE stack=$STACK task=$TASK_ID robot=$ROBOT trainable=$TRAINABLE
[e2e] annotate: judge=${ANNOTATE_JUDGE:-terminal} cameras=${ANNOTATE_CAMERAS:-first} criterion="$ANNOTATE_CRITERION"
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
# run.sh starts the checkpoint's policy server, waits for it, and stops it afterwards.
run ./run.sh "$ENV_ID" --policy --episodes "$RECORD_EPISODES" ${HEADLESS[@]+"${HEADLESS[@]}"} \
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
  run uv run --project tools/annotator i4h-annotator "${ANNOTATE_ARGS[@]}" \
    offline "$EXPANDED" --write --filter "$FILTERED"
  if [ "$VLLM_OWNED" -eq 1 ]; then
    run tools/annotator/scripts/vllm.sh stop
    VLLM_OWNED=0
  fi
fi

stage replay
# Replay runs without cameras: Arena's registration callback starts Kit before
# Isaac Lab can enable camera rendering.
[ "$SKIP_REPLAY" -eq 1 ] || run ./run.sh "$ENV_ID" --replay "$FILTERED" --episode 0 \
  ${HEADLESS[@]+"${HEADLESS[@]}"} --run-dir "$RUN_DIR/replay"

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
  run ./run.sh "$ENV_ID" --policy --checkpoint "$CHECKPOINT" --episodes "$VALIDATE_EPISODES" --headless \
    --run-dir "$VERIFY_DIR" --record "$VERIFY_DIR/verify.hdf5"
  require_success "$VERIFY_DIR"
fi

stage summary
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[e2e] dry run: no stages executed"
else
  echo "[e2e] complete: $RUN_DIR"
fi
