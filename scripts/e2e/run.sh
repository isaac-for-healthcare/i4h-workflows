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
VIZ=(--viz kit)
RUN_DIR=""
PORT="${I4H_POLICY_PORT:-8000}"
RECORD_EPISODES="${RECORD_EPISODES:-3}"
MIMIC_EPISODES="${MIMIC_EPISODES:-3}"
FINETUNE_STEPS="${FINETUNE_STEPS:-500}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-32}"
VALIDATE_EPISODES="${VALIDATE_EPISODES:-3}"

usage() {
  cat <<EOF
Usage: scripts/e2e/run.sh [options]

Runs setup -> policy record -> mimic -> annotate/filter -> replay -> convert ->
visualize -> finetune -> validate. The simulator stages run the workflow's
IsaacLab-Arena environment with Arena's policy runner and Isaac Lab's
replay_demos.py, against the policy server of the workflow's checkpoint.

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
    --headless) VIZ=(); shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) ENV_ID="$1"; shift ;;
  esac
done

# Each workflow's environment, its policy task, the Arena client for that task's
# server, the server itself, and the server's flag for a finetuned checkpoint.
case "$ENV_ID" in
  scissor_pick_and_place)
    SCENE=soarm_scissors
    ENVIRONMENT=(--external_environment_class_path i4h_arena.environments:SoarmScissors "$SCENE")
    TASK_ID=gr00t_n15/scissor_pick_and_place
    CLIENT=(--policy_type i4h_arena.policies.gr00t_n15.Gr00tN15RemotePolicy --embodiment so101_scissors --remote_port "$PORT")
    SERVER=(tasks/gr00t_n15/.venv/bin/python -m i4h_tasks.gr00t_n15.serve scissor_pick_and_place --port "$PORT")
    CHECKPOINT_FLAG=--model-path
    ;;
  ultrasound_liver_scan)
    SCENE=panda_phantom
    ENVIRONMENT=(--external_environment_class_path i4h_arena.environments:PandaPhantom "$SCENE")
    TASK_ID=openpi_pi0/ultrasound_liver_scan
    CLIENT=(--policy_type i4h_arena.policies.pi0.LiverScanPi0Policy --remote_port "$PORT")
    SERVER=(tasks/openpi_pi0/.venv/bin/python -m i4h_tasks.openpi_pi0.serve ultrasound_liver_scan --port "$PORT")
    CHECKPOINT_FLAG=--checkpoint
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
SERVER_PID=""

ARENA=(env -u VIRTUAL_ENV arena/.venv/bin/python)
POLICY_RUNNER=("${ARENA[@]}" -m isaaclab_arena.evaluation.policy_runner)
REPLAY_DEMOS=(
  "${ARENA[@]}" third_party/IsaacLab-e3f7d9f/scripts/tools/replay_demos.py
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback
  "${ENVIRONMENT[@]:0:2}" --task "$SCENE"
)

stage() { printf '\n========== %s ==========\n' "$1"; }
run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  [ "$DRY_RUN" -eq 1 ] || "$@"
}
# Start the policy server in the background; the Arena client waits until it answers.
serve() {
  local log="$1"
  shift
  printf '+'
  printf ' %q' "$@"
  printf ' > %q &\n' "$log"
  [ "$DRY_RUN" -eq 1 ] && return
  "$@" >"$log" 2>&1 &
  SERVER_PID=$!
  # Each server binds its port only once its checkpoint is loaded, which takes a
  # download on first use; the client would otherwise time out waiting for it.
  local deadline=$((SECONDS + ${I4H_E2E_SERVER_TIMEOUT_S:-3600}))
  until python3 -c 'import socket, sys; socket.create_connection(("127.0.0.1", int(sys.argv[1])), 1).close()' "$PORT" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[e2e] policy server exited before it was ready; see $log" >&2
      exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "[e2e] policy server not ready after ${I4H_E2E_SERVER_TIMEOUT_S:-3600}s; see $log" >&2
      exit 1
    fi
    sleep 5
  done
}
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
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
  stop_server
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
serve "$LOG_DIR/server-record.log" "${SERVER[@]}"
run "${POLICY_RUNNER[@]}" "${CLIENT[@]}" --num_episodes "$RECORD_EPISODES" --enable_cameras \
  ${VIZ[@]+"${VIZ[@]}"} --output_base_dir "$RUN_DIR/record" "${ENVIRONMENT[@]}" \
  --recorder_dataset_export_dir_path "$DATA_DIR" --recorder_dataset_filename "$(basename "$RECORDING" .hdf5)"
stop_server

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
[ "$SKIP_REPLAY" -eq 1 ] || run "${REPLAY_DEMOS[@]}" --dataset_file "$FILTERED" --select_episodes 0 \
  ${VIZ[@]+"${VIZ[@]}"}

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
  serve "$LOG_DIR/server-validate.log" "${SERVER[@]}" "$CHECKPOINT_FLAG" "$CHECKPOINT"
  run "${POLICY_RUNNER[@]}" "${CLIENT[@]}" --num_episodes "$VALIDATE_EPISODES" --enable_cameras \
    --output_base_dir "$VERIFY_DIR/outputs" "${ENVIRONMENT[@]}" \
    --recorder_dataset_export_dir_path "$VERIFY_DIR" --recorder_dataset_filename verify
  stop_server
  require_success "$VERIFY_DIR"
fi

stage summary
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[e2e] dry run: no stages executed"
else
  echo "[e2e] complete: $RUN_DIR"
fi
