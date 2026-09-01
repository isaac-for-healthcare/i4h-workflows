#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
#   ./run.sh list                       what can I run?
#   ./run.sh show  <workflow>               render the graph          (no Isaac)
#   ./run.sh lint  <workflow>               validate it               (no Isaac)
#   ./run.sh robot-pd <workflow> [options]  inspect and tune joint PD drives
#   ./run.sh <workflow> [mode] [options]    run it
#
# Run modes (choose one; with none, the scene just opens and renders):
#   --policy              AI policy: roll out the trained checkpoint
#   --rule-based          run the workflow's rule-based controller
#   --teleop [DEVICE]     manual control; bare flag means keyboard
#   --replay PATH         playback: replay a recording
#   --idle                view only: open the scene and render (the default)
#   --live                keep idle scene + Python bridge open for authoring
#   --mode NAME           any workflow-specific run mode
#
# Common options:
#   --episodes N          successful episodes requested
#   --attempts N          retries allowed per requested episode
#   --episode-steps N     lower per-episode step cap; never raises the workflow cap
#   --record [PATH]       write HDF5 episodes; default RUN_DIR/demos.hdf5
#   --record-failures     retain failed attempts too
#   --run-dir PATH        put launcher/backend/arena logs here; created automatically
#   --checkpoint PATH     override the configured policy checkpoint
#   --prompt TEXT         override the policy instruction
#   --policy-endpoint HOST:PORT connect to an external policy service over Zenoh
#   --headless            run without the interactive viewport
#   --presets NAME        physics backend: physx (the default) or newton
#   --device NAME         CUDA device for the simulation (default cuda:0)
#   --python-server       enable the Isaac Sim Python bridge on port 8226
#   --fluoro-backend NAME override automatic synthetic/patient-backed Slang selection
#   --fluoro-device NAME  Slang device for fluoroscopy (vulkan or cuda)
#   --patient-twin PATH   patient-twin YAML used by medical simulation sensors
#
# Compatibility option names: --max-attempts, --timesteps, --record-to,
# --save-all-episodes, and --episode-index.
#
# run.sh resolves the workflow, lints it, launches any backend the workflow needs, then
# launches arena — and tears all of it down on exit. `./stop.sh all` kills strays.
set -euo pipefail

CALLER_CWD="$(pwd -P)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export I4H_WORKFLOWS="$ROOT"

LIGHT_PROJECT=workflows
PIDFILE=".run/pids"
mkdir -p .run


light() { uv run --project "$LIGHT_PROJECT" "$@"; }

usage() { awk 'NR > 3 && /^set /{exit} NR > 3{sub(/^# ?/, ""); print}' "$ROOT/run.sh"; }

[ $# -eq 0 ] && { usage; exit 1; }

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  list)      shift; light python -m i4h_engine.cli list "$@"; exit $? ;;
  show)      shift; light python -m i4h_engine.cli show "$@"; exit $? ;;
  lint)      shift; light python -m i4h_engine.cli lint "$@"; exit $? ;;
  robot-pd|pd-tune|pd-diagnostics)
    shift
    exec env -u VIRTUAL_ENV uv run --project arena python arena/scripts/robot_pd_tune.py "$@"
    ;;
esac

WORKFLOW="$1"; shift

# -- parse just enough to know what to launch; arena owns the rest -------
MODE=""
DRY_RUN=0
NO_BACKEND=0
NAMESPACE=""
RUN_DIR=""
RUN_DIR_REQUESTED=0
RECORD=""
RECORD_REQUESTED=0
PATIENT_TWIN=""
CHECKPOINT=""
POLICY_ENDPOINT=""
ARENA_ARGS=()

caller_path() {
  case "$1" in
    /*) realpath -m -- "$1" ;;
    *) realpath -m -- "$CALLER_CWD/$1" ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)     MODE="$2";     shift 2 ;;
    --policy)   MODE=policy;   shift ;;
    --rule-based) MODE=rule-based; shift ;;
    --idle)     MODE=idle;     shift ;;
    --live)
      MODE=idle
      ARENA_ARGS+=(--python-server --idle-seconds 86400)
      shift ;;
    --teleop)
      MODE=teleop
      if [ $# -ge 2 ] && [[ "$2" != --* ]]; then
        ARENA_ARGS+=(--teleop-device "$2"); shift 2
      else shift; fi ;;
    --replay)
      MODE=replay
      ARENA_ARGS+=(--dataset "$(caller_path "$2")"); shift 2 ;;
    --no-backend) NO_BACKEND=1; shift ;;
    --policy-endpoint) POLICY_ENDPOINT="$2"; shift 2 ;;
    --policy-endpoint=*) POLICY_ENDPOINT="${1#*=}"; shift ;;
    --dry-run)    DRY_RUN=1; ARENA_ARGS+=("$1"); shift ;;
    --namespace)  NAMESPACE="$2"; ARENA_ARGS+=("$1" "$2"); shift 2 ;;
    --run-dir)    RUN_DIR="$2"; RUN_DIR_REQUESTED=1; shift 2 ;;
    --run-dir=*)  RUN_DIR="${1#*=}"; RUN_DIR_REQUESTED=1; shift ;;
    --record|--record-to)
      RECORD_REQUESTED=1
      if [ $# -ge 2 ] && [[ "$2" != --* ]]; then
        RECORD="$2"; shift 2
      else
        RECORD="demos.hdf5"; shift
      fi
      ;;
    --record=*|--record-to=*)
      RECORD="${1#*=}"; RECORD_REQUESTED=1; shift ;;
    --patient-twin)
      PATIENT_TWIN="$(caller_path "$2")"; shift 2 ;;
    --patient-twin=*)
      PATIENT_TWIN="$(caller_path "${1#*=}")"; shift ;;
    --checkpoint)
      CHECKPOINT="$2"
      # A local path is relative to the developer's invocation directory, not
      # the repository root. Preserve remote
      # model identifiers that do not resolve to a local file or directory.
      if [[ "$CHECKPOINT" != /* ]] && [ -e "$CALLER_CWD/$CHECKPOINT" ]; then
        CHECKPOINT="$(caller_path "$CHECKPOINT")"
      fi
      ARENA_ARGS+=("$1" "$CHECKPOINT")
      shift 2
      ;;
    *) ARENA_ARGS+=("$1"); shift ;;
  esac
done
if [ -n "$POLICY_ENDPOINT" ]; then
  case "$POLICY_ENDPOINT" in
    */*) ;;
    *:*) POLICY_ENDPOINT="tcp/$POLICY_ENDPOINT" ;;
    *) POLICY_ENDPOINT="tcp/$POLICY_ENDPOINT:7448" ;;
  esac
  export I4H_ZENOH_CONNECT="$POLICY_ENDPOINT"
  NO_BACKEND=1
  echo "==> external policy $POLICY_ENDPOINT"
fi
# No default here: an unnamed mode resolves to idle, which opens the scene
# and renders without driving anything.
[ -z "$NAMESPACE" ] && NAMESPACE="$WORKFLOW"

# Backend logs, the arena log, and caller-selected recordings share one run
# directory. An explicit relative path belongs to the caller, not to the
# launcher's internal working directory; the automatic path remains canonical.
if [ "$RUN_DIR_REQUESTED" -eq 1 ]; then
  [ -n "$RUN_DIR" ] || { echo "!! --run-dir requires a non-empty path" >&2; exit 2; }
  RUN_DIR="$(caller_path "$RUN_DIR")"
  mkdir -p "$RUN_DIR"
else
  RUN_PARENT="$ROOT/runs/${WORKFLOW}"
  RUN_STEM="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$RUN_PARENT"
  RUN_DIR="$RUN_PARENT/$RUN_STEM"
  suffix=0
  while ! mkdir "$RUN_DIR" 2>/dev/null; do
    suffix=$((suffix + 1))
    RUN_DIR="$(printf '%s/%s_%02d' "$RUN_PARENT" "$RUN_STEM" "$suffix")"
  done
fi
export I4H_RUN_DIR="$RUN_DIR"
echo "==> run dir $RUN_DIR"

if [ "$RECORD_REQUESTED" -eq 1 ]; then
  [ -n "$RECORD" ] || { echo "!! --record= requires a non-empty path" >&2; exit 2; }
  case "$RECORD" in
    /*) RECORD="$(realpath -m -- "$RECORD")" ;;
    *) RECORD="$(realpath -m -- "$RUN_DIR/$RECORD")" ;;
  esac
  mkdir -p "$(dirname "$RECORD")"
  export I4H_RECORD_PATH="$RECORD"
  ARENA_ARGS+=(--record "$RECORD")
  echo "==> recording $RECORD"
fi

if [ -n "$PATIENT_TWIN" ]; then
  ARENA_ARGS+=(--patient-twin "$PATIENT_TWIN")
  echo "==> patient twin $PATIENT_TWIN"
fi

RUN_METADATA="$RUN_DIR/run.json"
RUN_LOG="$RUN_DIR/i4h_arena.log"
export I4H_RUN_METADATA="$RUN_METADATA"
light python - "$RUN_METADATA" "$WORKFLOW" "$MODE" "$RUN_DIR" "$RECORD" "$PATIENT_TWIN" "$CALLER_CWD" "$ROOT/run.sh" <<'PY'
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

metadata, workflow, mode, run_dir, recording, patient_twin, caller_cwd, launcher = sys.argv[1:]
Path(metadata).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "workflow": workflow,
            "mode": mode or "idle",
            "run_dir": run_dir,
            "recording": recording or None,
            "patient_twin": patient_twin or None,
            "caller_cwd": caller_cwd,
            "launcher": launcher,
            "created_at": datetime.now(UTC).isoformat(),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
{
  printf 'I4H_RUN_DIR=%s\n' "$RUN_DIR"
  printf 'I4H_RUN_METADATA=%s\n' "$RUN_METADATA"
  if [ -n "$RECORD" ]; then
    printf 'I4H_RECORD_PATH=%s\n' "$RECORD"
  fi
} > "$RUN_LOG"
echo "==> run metadata $RUN_METADATA"

# -- 1-4: resolve, lint. Milliseconds, in the light venv, before anything heavy.
echo "==> lint $WORKFLOW [${MODE:-default}]"
light python -m i4h_engine.cli lint "$WORKFLOW" --mode "$MODE"

# -- 5: launch the backends this workflow actually needs ---------------------
: > "$PIDFILE"
cleanup() {
  if [ -s "$PIDFILE" ]; then
    while read -r pid; do
      kill "$pid" 2>/dev/null || true
    done < "$PIDFILE"
  fi
  rm -f "$PIDFILE"
}
trap cleanup EXIT INT TERM

if [ "$NO_BACKEND" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  backends=$(light python - "$WORKFLOW" "$MODE" <<'PY'
import sys
from i4h_engine.loader import resolve_workflow
from i4h_engine.registry import default_registry

workflow = resolve_workflow(sys.argv[1], sys.argv[2])
registry = default_registry()
seen = set()
for node in workflow.graph.nodes:
    spec = node.spec or registry.tasks.get(node.task_id)
    if spec is not None and spec.runtime == "remote" and spec.backend:
        key = (spec.backend.project, spec.backend.entry)
        if key not in seen:
            seen.add(key)
            print(f"{spec.backend.project}\t{spec.backend.entry}\t{spec.id}")
PY
)
  backend_logs=()
  while IFS=$'\t' read -r project entry task_id; do
    [ -z "$project" ] && continue
    if [ ! -d "$project/.venv" ]; then
      if [ -n "${I4H_VENV_ROOT:-}" ]; then
        echo "==> syncing persistent backend environment $project"
        I4H_SETUP_PROJECTS="$project" I4H_THIRD_PARTY_TARGET="$project" ./setup.sh
      else
        echo "!! $project is not synced. Run: ./setup.sh" >&2
        exit 1
      fi
    fi
    module="${entry%%:*}"
    func="${entry##*:}"
    echo "==> backend $project ($entry)"
    # PI0 is JAX; without a pin it claims every visible GPU and leaves nothing
    # for the Isaac process it is supposed to be serving.
    backend_env=(env -u VIRTUAL_ENV)
    case "$project" in
      *openpi*) backend_env+=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}") ;;
    esac
    backend_log="$RUN_DIR/backend-$(basename "$project").log"
    backend_checkpoint_args=()
    [ -n "$CHECKPOINT" ] && backend_checkpoint_args+=(--checkpoint "$CHECKPOINT")
    "${backend_env[@]}" uv run --project "$project" python -c \
      "import importlib; getattr(importlib.import_module('$module'),'$func')()" \
      --namespace "$NAMESPACE" --preload "$task_id" "${backend_checkpoint_args[@]}" \
      > "$backend_log" 2>&1 &
    echo $! >> "$PIDFILE"
    backend_logs+=("$backend_log")
  done <<< "$backends"

  # Wait for every checkpoint to be resident before the simulator starts so
  # inference startup never consumes the episode's simulation-step budget.
  for log in "${backend_logs[@]}"; do
    echo "==> waiting for $(basename "$log" .log) to load"
    backend_ready=0
    for _ in $(seq 600); do
      if grep -q "ready for specs" "$log" 2>/dev/null; then
        backend_ready=1
        break
      fi
      grep -qE "Traceback|Error:" "$log" 2>/dev/null && { echo "!! backend failed; see $log" >&2; exit 1; }
      sleep 1
    done
    if [ "$backend_ready" -ne 1 ]; then
      echo "!! backend readiness timed out; see $log" >&2
      exit 1
    fi
  done
fi

# -- 6: arena ------------------------------------------------------------
# The Arena CLI deliberately keeps dry-run discovery Isaac-free. Run it in
# the already-synced light environment so CPU CI does not need the large Arena
# dependency tree or third-party checkouts.
if [ "$DRY_RUN" -eq 1 ]; then
  echo "==> arena $WORKFLOW [${MODE:-default}]"
  PYTHONPATH="$ROOT/arena${PYTHONPATH:+:$PYTHONPATH}" \
    env -u VIRTUAL_ENV uv run --project workflows --no-sync python -m i4h_arena.cli \
      --workflow "$WORKFLOW" --mode "$MODE" --namespace "$NAMESPACE" "${ARENA_ARGS[@]}" \
      2>&1 | tee -a "$RUN_LOG"
  exit 0
fi

if [ ! -d "arena/.venv" ] && [ "$DRY_RUN" -eq 0 ]; then
  echo "!! arena is not synced. Run: ./setup.sh" >&2
  exit 1
fi

# Kit blocks on an interactive EULA prompt otherwise, which turns any
# non-interactive run — CI, a backgrounded launch — into a silent hang.
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# Kit also blocks on a second modal — the Omniverse telemetry consent — which
# the EULA variable does not cover. It is answered by a config file, so write a
# decline once rather than have every fresh machine hang on a dialog.
privacy="${HOME}/.nvidia-omniverse/config/privacy.toml"
if [ ! -f "$privacy" ]; then
  mkdir -p "$(dirname "$privacy")"
  printf '[privacy]\nperformance = false\npersonalization = false\nusage = false\n' > "$privacy"
  echo "==> wrote $privacy (declining telemetry; edit to opt in)"
fi

# Isaac Sim and torch each ship a libgomp; whichever loads second loses, and
# the symptom is a crash deep in OpenMP rather than anything that names the
# clash. Preloading one settles it.
if [ "$(uname -m)" = "aarch64" ]; then
  sys_libgomp="$(ls /lib/*/libgomp.so.1 2>/dev/null | head -1 || true)"
  [ -n "$sys_libgomp" ] && export LD_PRELOAD="${sys_libgomp}${LD_PRELOAD:+:$LD_PRELOAD}"
  export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=2000000}"
else
  torch_libgomp="$(env -u VIRTUAL_ENV uv run --project arena --no-sync python -c \
    'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib" / "libgomp.so.1")' 2>/dev/null || true)"
  if [ -n "$torch_libgomp" ] && [ -e "$torch_libgomp" ]; then
    export LD_PRELOAD="${torch_libgomp}${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi

echo "==> arena $WORKFLOW [${MODE:-default}]"
env -u VIRTUAL_ENV uv run --project arena python -m i4h_arena.cli \
  --workflow "$WORKFLOW" --mode "$MODE" --namespace "$NAMESPACE" "${ARENA_ARGS[@]}" \
  2>&1 | tee -a "$RUN_LOG"
