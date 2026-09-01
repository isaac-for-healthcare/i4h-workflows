#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Manage one detached root-level run.sh --live session for Local Agent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$ROOT/.run/local-agent-bridge"
PORT=8226
STARTUP_TIMEOUT="${I4H_AGENT_BRIDGE_TIMEOUT:-900}"
mkdir -p "$STATE_DIR"

usage() {
    echo "usage: ./local-agent/bridge.sh start <workflow> [run-dir]" >&2
    echo "       ./local-agent/bridge.sh <status|stop|rundir> <workflow>" >&2
}

validate_workflow() {
    [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]] || {
        echo "bridge: invalid workflow id '$1'" >&2
        exit 2
    }
}

pid_file() { printf '%s/%s.pid\n' "$STATE_DIR" "$1"; }
run_file() { printf '%s/%s.run-dir\n' "$STATE_DIR" "$1"; }

port_ready() {
    python3 - "$PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

managed_pid() {
    local file
    file="$(pid_file "$1")"
    [[ -s "$file" ]] || return 1
    cat "$file"
}

default_run_dir() {
    local workflow="$1" base candidate suffix
    base="$ROOT/runs/$workflow/$(date +%Y%m%d_%H%M%S)"
    candidate="$base"
    suffix=2
    while [[ -e "$candidate" ]]; do
        candidate="${base}_$(printf '%02d' "$suffix")"
        suffix=$((suffix + 1))
    done
    printf '%s\n' "$candidate"
}

start_bridge() {
    local workflow="$1" requested_run_dir="${2:-}" pid run_dir log deadline
    validate_workflow "$workflow"
    if pid="$(managed_pid "$workflow")" && kill -0 "$pid" 2>/dev/null; then
        echo "bridge: $workflow already managed by pid $pid" >&2
        exit 1
    fi
    if port_ready; then
        echo "bridge: port $PORT is already in use by an unmanaged session" >&2
        exit 1
    fi

    if [[ -n "$requested_run_dir" ]]; then
        run_dir="$(realpath -m "$requested_run_dir")"
        case "$run_dir" in
            "$ROOT/runs/$workflow"/*) ;;
            *) echo "bridge: run directory must be under $ROOT/runs/$workflow/" >&2; exit 2 ;;
        esac
    else
        run_dir="$(default_run_dir "$workflow")"
    fi
    mkdir -p "$run_dir"
    log="$run_dir/local-agent-live.log"
    echo "$run_dir" > "$(run_file "$workflow")"

    setsid "$ROOT/run.sh" "$workflow" --live --run-dir "$run_dir" >"$log" 2>&1 &
    pid=$!
    echo "$pid" > "$(pid_file "$workflow")"
    echo "[bridge] launching workflow=$workflow pid=$pid"
    echo "RUN_DIR=$run_dir"

    deadline=$((SECONDS + STARTUP_TIMEOUT))
    while ((SECONDS < deadline)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[bridge] FAILED before ready; see $log" >&2
            tail -80 "$log" >&2 || true
            rm -f "$(pid_file "$workflow")"
            exit 1
        fi
        if port_ready; then
            echo "[bridge] READY — 127.0.0.1:$PORT"
            return
        fi
        sleep 2
    done

    echo "[bridge] FAILED: readiness timed out after ${STARTUP_TIMEOUT}s; see $log" >&2
    stop_bridge "$workflow" || true
    exit 1
}

status_bridge() {
    local workflow="$1" pid
    validate_workflow "$workflow"
    if pid="$(managed_pid "$workflow")" && kill -0 "$pid" 2>/dev/null && port_ready; then
        echo "bridge: ready workflow=$workflow pid=$pid port=$PORT"
        return
    fi
    echo "bridge: not ready workflow=$workflow"
    return 1
}

stop_bridge() {
    local workflow="$1" pid
    validate_workflow "$workflow"
    if ! pid="$(managed_pid "$workflow")" || ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$(pid_file "$workflow")"
        echo "bridge: no managed session for $workflow"
        return
    fi
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "bridge: $workflow did not stop cleanly; run ./stop.sh all for broad cleanup" >&2
        return 1
    fi
    rm -f "$(pid_file "$workflow")"
    echo "bridge: stopped workflow=$workflow"
}

show_rundir() {
    local workflow="$1" file
    validate_workflow "$workflow"
    file="$(run_file "$workflow")"
    [[ -s "$file" ]] || {
        echo "bridge: no run directory recorded for $workflow" >&2
        exit 1
    }
    cat "$file"
}

command="${1:-}"
workflow="${2:-}"
[[ -n "$command" && -n "$workflow" ]] || {
    usage
    exit 2
}

case "$command" in
    start)
        [[ "$#" -le 3 ]] || { usage; exit 2; }
        start_bridge "$workflow" "${3:-}"
        ;;
    status)
        [[ "$#" -eq 2 ]] || { usage; exit 2; }
        status_bridge "$workflow"
        ;;
    stop)
        [[ "$#" -eq 2 ]] || { usage; exit 2; }
        stop_bridge "$workflow"
        ;;
    rundir)
        [[ "$#" -eq 2 ]] || { usage; exit 2; }
        show_rundir "$workflow"
        ;;
    *) usage; exit 2 ;;
esac
