#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Manage the scene-edit BRIDGE lifecycle for the one-shot local agent.
#
# The bridge (`arena/run.sh --env <env> --bridge`) is a long-running FOREGROUND
# process. Launching it directly blocks the agent's (tmux) shell forever and floods
# stdout with Isaac logs. This helper delegates to arena/run.sh ensure-bridge, which
# launches a detached --keep-open bridge, waits until it reports ready, and LEAVES IT
# RUNNING so the agent can then issue `/script` + `curl` edits across separate calls.
# Stop it only via `stop` (= arena/stop.sh).
#
# Usage:
#   ./local-agent/bridge.sh start  <env_id>   # detached launch + wait-for-ready; prints RUN_DIR=...
#   ./local-agent/bridge.sh status <env_id>   # is the bridge HTTP API up?
#   ./local-agent/bridge.sh stop   <env_id>   # stop the arena (the only way to stop the bridge)
#   ./local-agent/bridge.sh rundir            # echo the latest scene-edit RUN_DIR
#
# Run `start` PLAINLY — it takes several minutes (cold Isaac start). Do NOT wrap it in
# a short `timeout`, or it gets killed before the bridge is ready.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A="$ROOT/workflows/agentic"

cmd="${1:-}"; ENV="${2:-}"
case "$cmd" in
  start)
    [ -n "$ENV" ] || { echo "usage: bridge.sh start <env_id>" >&2; exit 2; }
    RUN="$A/runs/scene_edit_${ENV}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN/logs" "$RUN/scripts" "$RUN/captures"
    ln -sfn "$RUN" "$A/runs/.latest"
    LOG="$RUN/logs/bridge.log"; : > "$LOG"
    BRIDGE_URL="$("$A/arena/run.sh" bridge-url --env "$ENV")"
    HEALTH="$BRIDGE_URL/health"
    # Detached launch — survives this script exiting; logs to file (not the agent's stdout).
    echo "[bridge] launching env=$ENV detached (log: $LOG)"
    echo "RUN_DIR=$RUN"
    if ! "$A/arena/run.sh" ensure-bridge --env "$ENV" --log "$LOG"; then
      echo "[bridge] FAILED before ready; see $LOG"
      "$A/arena/stop.sh" --env "$ENV" >/dev/null 2>&1
      exit 1
    fi
    deadline=$((SECONDS + 480))
    while (( SECONDS < deadline )); do
      if grep -qiE "scene-edit bridge ready" "$LOG" || curl -sf "$HEALTH" >/dev/null 2>&1; then
        echo "[bridge] READY — edit via curl $BRIDGE_URL ; write /script files under $RUN/scripts/"
        exit 0
      fi
      if grep -qiE "\[agentic-arena\] failed with|Traceback \(most recent" "$LOG"; then
        echo "[bridge] FAILED before ready:"
        grep -niE "failed with|Error|Traceback|No module|cannot import|line [0-9]+, in" "$LOG" | head -20
        "$A/arena/stop.sh" --env "$ENV" >/dev/null 2>&1
        exit 1
      fi
      sleep 5
    done
    echo "[bridge] TIMEOUT waiting for ready; see $LOG"
    "$A/arena/stop.sh" --env "$ENV" >/dev/null 2>&1
    exit 1
    ;;
  status)
    [ -n "$ENV" ] || { echo "usage: bridge.sh status <env_id>" >&2; exit 2; }
    HEALTH="$("$A/arena/run.sh" bridge-url --env "$ENV")/health"
    curl -sf "$HEALTH" >/dev/null 2>&1 && echo "bridge: ready" || { echo "bridge: not ready"; exit 1; }
    ;;
  stop)
    [ -n "$ENV" ] || { echo "usage: bridge.sh stop <env_id>" >&2; exit 2; }
    exec "$A/arena/stop.sh" --env "$ENV"
    ;;
  rundir)
    readlink -f "$A/runs/.latest"
    ;;
  *)
    echo "usage: bridge.sh <start|status|stop|rundir> <env_id>" >&2; exit 2
    ;;
esac
