#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# OpenCode shell adapter backed by one persistent tmux session.
set -uo pipefail

SESSION="${I4H_TMUX_SESSION:-i4h_local_agent}"
RUNDIR="${TMPDIR:-/tmp}/i4h-tmux-shell.$(id -u)"
mkdir -p "${RUNDIR}"

cmd=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|-lc|-ic|-cl|-li|-il) shift; cmd="${1:-}"; shift || true ;;
        -*) shift ;;
        *) cmd="$1"; shift ;;
    esac
done

[[ -z "${cmd}" ]] && exec /usr/bin/env bash -l

# Guardrail: refuse destructive git ONLY when it targets protected infra (skills/,
# local-agent/) or is a BROAD reset with no workflows/ path (e.g. `git reset --hard`,
# `git checkout .`, `git stash`) — those discard the user's uncommitted skill/config
# edits or everything. ALLOW it scoped to a workflows/ path so the agent can still revert
# its own contamination of committed workflow code (its normal recovery). Self-protecting:
# reverting this file (`git checkout local-agent/...`) is itself blocked.
if printf '%s' "${cmd}" | grep -qiE '(^|[;&|])[[:space:]]*git[[:space:]]+(checkout|restore|reset|stash|clean)([[:space:]]|$)'; then
    if printf '%s' "${cmd}" | grep -qiE 'skills/|local-agent/' \
       || ! printf '%s' "${cmd}" | grep -qiE 'workflows/'; then
        echo "tmux-shell: BLOCKED destructive git on protected/broad paths (skills/, local-agent/, or no workflows/ path). It discards uncommitted skill/config work. To revert your own change to committed WORKFLOW code, scope it to a path: 'git checkout workflows/agentic/...'. To undo any other edit, edit the file directly." >&2
        exit 1
    fi
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux-shell: install tmux" >&2; exit 127; }

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION}" -x 220 -y 50 -c "${I4H_TMUX_CWD:-$PWD}"
    tmux send-keys -t "${SESSION}" 'export PS1="" PROMPT_COMMAND="" HISTFILE=/dev/null' Enter
    [[ -n "${I4H_TMUX_CWD:-}" ]] && tmux send-keys -t "${SESSION}" "cd $(printf '%q' "${I4H_TMUX_CWD}")" Enter
    # Export the forwarded env ONCE, at session creation. Re-sending it on every
    # command (the old behavior) meant ~11 send-keys per call — a large surface for a
    # dropped Enter, which leaves the real command typed-but-unsubmitted and hangs the poll.
    for v in I4H_WORKFLOWS REPO_ROOT I4H_AGENT_BASE_URL I4H_AGENT_VL_BASE_URL I4H_AGENT_VL_MODEL I4H_AGENT_VL_API_KEY CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER DISPLAY XAUTHORITY PATH; do
        [[ -n "${!v:-}" ]] && tmux send-keys -t "${SESSION}" "export ${v}=$(printf '%q' "${!v}")" Enter
    done
    sleep 0.3
else
    # Reuse: reset the shell to a clean prompt. A prior command may have left a stuck
    # foreground process (OpenCode timed out while it ran, or it waited on stdin) or a
    # half-typed line; either way new keystrokes would otherwise pile onto it.
    tmux send-keys -t "${SESSION}" C-c 2>/dev/null   # interrupt any leftover foreground process
    tmux send-keys -t "${SESSION}" C-u 2>/dev/null   # clear any half-typed input line
    sleep 0.15
fi

id="$$.${RANDOM}"
cmdf="${RUNDIR}/cmd.${id}.sh"
out="${RUNDIR}/out.${id}"
rc="${RUNDIR}/rc.${id}"
started="${RUNDIR}/started.${id}"
printf '%s\n' "${cmd}" > "${cmdf}"
: > "${out}"

# The `started` sentinel is touched the instant this line actually executes, so the
# poll loop can tell "command is running" (started exists) from "command never ran"
# (Enter was dropped and it sits buffered at the prompt).
runline="touch $(printf '%q' "${started}") ; { source $(printf '%q' "${cmdf}") ; } > $(printf '%q' "${out}") 2>&1 ; echo \$? > $(printf '%q' "${rc}")"
tmux send-keys -t "${SESSION}" "${runline}" Enter

# Poll for completion. Never hang forever:
#   - session ended            -> emit partial output, exit 1
#   - command never started     -> Enter likely dropped; re-send once, then reset+exit 1
# Once `started` exists the command is genuinely running, so we wait indefinitely for
# it (a multi-minute build is fine) — only the session-ended guard applies.
grace="${I4H_TMUX_START_GRACE:-25}"   # 0.2s ticks (~5s) to see `started` before acting
waited=0; resent=0
while [[ ! -f "${rc}" ]]; do
    tmux has-session -t "${SESSION}" 2>/dev/null || {
        [[ -f "${out}" ]] && sed -n '1,200p' "${out}"
        echo "tmux-shell: session ended" >&2
        rm -f "${cmdf}" "${out}" "${started}"
        exit 1
    }
    if [[ ! -f "${started}" ]]; then
        waited=$((waited+1))
        if (( waited >= grace )); then
            if (( resent == 0 )); then
                tmux send-keys -t "${SESSION}" C-u 2>/dev/null   # clear the buffered partial line
                tmux send-keys -t "${SESSION}" "${runline}" Enter
                resent=1; waited=0
            else
                echo "tmux-shell: command did not execute (pane idle, no result); resetting session" >&2
                tmux kill-session -t "${SESSION}" 2>/dev/null
                rm -f "${cmdf}" "${out}" "${started}"
                exit 1
            fi
        fi
    fi
    sleep 0.2
done

code="$(sed -n '1p' "${rc}" 2>/dev/null || echo 1)"
[[ -f "${out}" ]] && sed -n '1,100000p' "${out}"
rm -f "${cmdf}" "${out}" "${rc}" "${started}"
exit "${code:-1}"
