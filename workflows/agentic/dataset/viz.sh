#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATASET_PROJECT="$REPO_ROOT/workflows/agentic/dataset"

usage() {
  cat <<EOF
Usage:
  viz.sh [REPO_ID|DATASET_DIR] [--port N] [--host H] [--episodes "0 2 5"] [--state-dir DIR]
  viz.sh --stop [--port N] [--state-dir DIR]

Serves a LeRobot dataset with lerobot's HTML visualizer.
EOF
}

MODE=start
TARGET=""
PORT=""
HOST="127.0.0.1"
EPISODES=""
STATE_DIR="${LEROBOT_VIZ_STATE_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --host=*) HOST="${1#*=}"; shift ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --episodes=*) EPISODES="${1#*=}"; shift ;;
    --state-dir) STATE_DIR="$2"; shift 2 ;;
    --state-dir=*) STATE_DIR="${1#*=}"; shift ;;
    --stop) MODE=stop; shift ;;
    -h|--help) usage; exit 0 ;;
    -*)
      echo "[lerobot-viz] unknown flag: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$TARGET" ]]; then
        echo "[lerobot-viz] unexpected positional arg: $1" >&2
        exit 2
      fi
      TARGET="$1"
      shift
      ;;
  esac
done

resolve_state_dir() {
  if [[ -n "$STATE_DIR" ]]; then
    mkdir -p "$STATE_DIR"
    return
  fi

  if mkdir "/tmp/.lerobot-viz-probe-$$" 2>/dev/null; then
    rmdir "/tmp/.lerobot-viz-probe-$$"
    STATE_DIR="/tmp"
  else
    STATE_DIR="$REPO_ROOT/.lerobot-viz"
    mkdir -p "$STATE_DIR"
  fi
}

stop_viz() {
  local port="${PORT:-9090}"
  local pid_file="$STATE_DIR/lerobot-viz.$port.pid"

  if [[ ! -f "$pid_file" ]]; then
    echo "[lerobot-viz] no pid file at $pid_file" >&2
    return 1
  fi

  local pid
  pid="$(<"$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
    echo "[lerobot-viz] stopped pid $pid (port $port)"
  else
    echo "[lerobot-viz] removing stale pid file $pid_file"
  fi
  rm -f "$pid_file"
}

resolve_dataset() {
  local cache_root="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

  if [[ -z "$TARGET" ]]; then
    if [[ ! -d "$cache_root" ]]; then
      echo "[lerobot-viz] no dataset target and cache root does not exist: $cache_root" >&2
      exit 1
    fi
    TARGET="$(python3 - "$cache_root" <<'PYEOF'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser()
datasets = [p.parent for p in root.glob("**/meta") if p.is_dir()]
if datasets:
    print(max(datasets, key=lambda p: p.stat().st_mtime).relative_to(root))
PYEOF
)"
    if [[ -z "$TARGET" ]]; then
      echo "[lerobot-viz] no LeRobot datasets found under $cache_root" >&2
      exit 1
    fi
  fi

  if [[ "$TARGET" == /* || "$TARGET" == ./* || "$TARGET" == ../* ]]; then
    if [[ ! -d "$TARGET/meta" ]]; then
      echo "[lerobot-viz] not a LeRobot dataset directory: $TARGET" >&2
      exit 1
    fi
    local dataset_dir parent name
    dataset_dir="$(cd "$TARGET" && pwd)"
    if [[ "$dataset_dir" == "$cache_root"/* ]]; then
      REPO_ID="${dataset_dir#$cache_root/}"
    else
      parent="$(dirname "$dataset_dir")"
      name="$(basename "$dataset_dir")"
      export HF_LEROBOT_HOME="$parent"
      mkdir -p "$parent/local"
      ln -sfn "../$name" "$parent/local/$name"
      REPO_ID="local/$name"
    fi
  else
    REPO_ID="$TARGET"
    [[ "$REPO_ID" == */* ]] || REPO_ID="local/$REPO_ID"
  fi
}

choose_port() {
  if [[ -n "$PORT" ]]; then
    return
  fi
  for candidate in $(seq 9090 9099); do
    if ! ss -tln 2>/dev/null | awk 'NR>1 {print $4}' | grep -qE ":$candidate$"; then
      PORT="$candidate"
      return
    fi
  done
  echo "[lerobot-viz] no free port in 9090..9099; pass --port N" >&2
  exit 1
}

start_viz() {
  if [[ ! -d "$DATASET_PROJECT/.venv" ]]; then
    echo "[lerobot-viz] dataset venv missing; run workflows/agentic/setup.sh first" >&2
    exit 1
  fi

  resolve_dataset
  choose_port

  local cache_root="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
  if [[ ! -d "$cache_root/$REPO_ID" ]]; then
    echo "[lerobot-viz] dataset not found: $cache_root/$REPO_ID" >&2
    exit 1
  fi

  local pid_file="$STATE_DIR/lerobot-viz.$PORT.pid"
  local log_file="$STATE_DIR/lerobot-viz.$PORT.log"
  local output_dir="$STATE_DIR/lerobot-viz.$PORT"
  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    echo "[lerobot-viz] already running on port $PORT (pid $(<"$pid_file"))" >&2
    exit 1
  fi

  rm -rf "$output_dir/static"
  mkdir -p "$output_dir"
  local cmd=(
    uv run python -m lerobot.scripts.visualize_dataset_html
    --repo-id "$REPO_ID"
    --output-dir "$output_dir"
    --serve 1
    --host "$HOST"
    --port "$PORT"
  )
  # shellcheck disable=SC2206
  [[ -n "$EPISODES" ]] && cmd+=(--episodes $EPISODES)

  echo "[lerobot-viz] starting: (cd $DATASET_PROJECT && ${cmd[*]})"
  if command -v setsid >/dev/null 2>&1; then
    setsid bash -c 'cd "$1" && shift; exec "$@"' lerobot-viz "$DATASET_PROJECT" "${cmd[@]}" > "$log_file" 2>&1 &
  else
    (cd "$DATASET_PROJECT" && "${cmd[@]}") > "$log_file" 2>&1 &
  fi
  local pid=$!
  echo "$pid" > "$pid_file"
  disown "$pid" 2>/dev/null || true

  local url="http://$HOST:$PORT/"
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[lerobot-viz] server died during startup; tail of $log_file:" >&2
      tail -40 "$log_file" >&2 || true
      rm -f "$pid_file"
      exit 1
    fi
    local code
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 2 "$url" 2>/dev/null || echo 000)"
    [[ "$code" == 200 || "$code" =~ ^3[0-9]{2}$ ]] && break
    sleep 1
  done

  cat <<EOF
[lerobot-viz] running
  URL:       $url
  repo-id:   $REPO_ID
  PID:       $pid
  PID file:  $pid_file
  log:       $log_file
  output:    $output_dir

Stop with:
  $REPO_ROOT/workflows/agentic/dataset/viz.sh --stop --port $PORT --state-dir "$STATE_DIR"
EOF
}

resolve_state_dir
if [[ "$MODE" == stop ]]; then
  stop_viz
else
  start_viz
fi
