#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stop anything run.sh started. run.sh already tears down on a clean exit; this
# is for the times it did not get one.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ "$#" -ne 1 ] || [ "$1" != "all" ]; then
  echo "usage: ./stop.sh all" >&2
  exit 2
fi

stopped=0

if [ -s .run/pids ]; then
  while read -r pid; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "stopping tracked pid $pid"
      kill "$pid" 2>/dev/null && stopped=$((stopped + 1))
    fi
  done < .run/pids
  rm -f .run/pids
fi

# Match on where the process actually lives, not on its command line. The uv
# parent invokes `--project tasks/gr00t_n15` relatively, so a $ROOT-prefixed
# pattern never matched it and the backend survived every stop.
in_tree() {
  local pid="$1" cwd exe
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
  case "$cwd" in "$ROOT"|"$ROOT"/*) return 0 ;; esac
  case "$exe" in "$ROOT"/*) return 0 ;; esac
  return 1
}

# Only ever the interpreters this tree starts. Without this a shell whose
# command line merely mentions one of the patterns — including the shell
# running this script — matches and gets killed.
runnable() {
  case "$(cat "/proc/$1/comm" 2>/dev/null || true)" in
    python*|uv|kit|isaac*) return 0 ;;
    *) return 1 ;;
  esac
}

ours() {
  # Anything this tree starts: the arena CLI, a policy backend, or the uv
  # wrapper around either. Kit children die with their parent.
  pgrep -f "i4h_arena\.cli|i4h_tasks\.gr00t_n1|i4h_tasks\.openpi_pi0|tasks/gr00t_n1|tasks/openpi_pi0" 2>/dev/null || true
}

for signal in TERM KILL; do
  targets=()
  while read -r pid; do
    [ -z "$pid" ] && continue
    [ "$pid" = "$$" ] && continue
    runnable "$pid" || continue
    in_tree "$pid" || continue
    targets+=("$pid")
  done < <(ours)
  [ ${#targets[@]} -eq 0 ] && break
  for pid in "${targets[@]}"; do
    echo "stopping pid $pid (SIG$signal)"
    kill "-$signal" "$pid" 2>/dev/null && [ "$signal" = TERM ] && stopped=$((stopped + 1))
  done
  [ "$signal" = TERM ] && sleep 2
done

echo "stopped $stopped process(es)"
