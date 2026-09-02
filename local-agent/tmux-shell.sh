#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# OpenCode shell adapter backed by one persistent tmux session.
set -uo pipefail

SESSION="${I4H_TMUX_SESSION:-i4h_local_agent}"
REPO_ROOT="${I4H_AGENT_REPO_ROOT:-$PWD}"
RUNDIR="${REPO_ROOT}/.run/local-agent-tmux.$(id -u)"
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

if printf '%s' "${cmd}" | grep -qiE '(^|[;&|])[[:space:]]*git[[:space:]]+(checkout|restore|reset|stash|clean)([[:space:]]|$)'; then
    echo "tmux-shell: BLOCKED destructive git. Undo changes with targeted file edits." >&2
    exit 1
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux-shell: install tmux" >&2; exit 127; }

# OpenCode may issue shell tools concurrently. One tmux pane cannot safely accept
# interleaved commands, so serialize each complete command while preserving the
# pane's working directory and exported environment between calls.
lock_dir="${RUNDIR}/session.lock.d"
while ! mkdir "${lock_dir}" 2>/dev/null; do
    owner="$(sed -n '1p' "${lock_dir}/owner" 2>/dev/null || true)"
    if [[ -n "${owner}" ]] && ! kill -0 "${owner}" 2>/dev/null; then
        rm -f "${lock_dir}/owner"
        rmdir "${lock_dir}" 2>/dev/null || true
        continue
    fi
    sleep 0.1
done
printf '%s\n' "$$" > "${lock_dir}/owner"
release_lock() {
    rm -f "${lock_dir}/owner"
    rmdir "${lock_dir}" 2>/dev/null || true
}
trap release_lock EXIT INT TERM

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION}" -x 220 -y 50 -c "${I4H_TMUX_CWD:-$PWD}"
    tmux send-keys -t "${SESSION}" 'export PS1="" PROMPT_COMMAND="" HISTFILE=/dev/null' Enter
    [[ -n "${I4H_TMUX_CWD:-}" ]] && tmux send-keys -t "${SESSION}" "cd $(printf '%q' "${I4H_TMUX_CWD}")" Enter
    for variable in I4H_LOCAL_AGENT I4H_WORKFLOWS REPO_ROOT I4H_AGENT_BASE_URL I4H_AGENT_VL_BASE_URL I4H_AGENT_VL_MODEL I4H_AGENT_VL_API_KEY CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER DISPLAY XAUTHORITY PATH; do
        [[ -n "${!variable:-}" ]] && tmux send-keys -t "${SESSION}" "export ${variable}=$(printf '%q' "${!variable}")" Enter
    done
    sleep 0.3
else
    tmux send-keys -t "${SESSION}" C-c 2>/dev/null
    tmux send-keys -t "${SESSION}" C-u 2>/dev/null
    sleep 0.15
fi

id="$$.${RANDOM}"
cmd_file="${RUNDIR}/cmd.${id}.sh"
output_file="${RUNDIR}/out.${id}"
rc_file="${RUNDIR}/rc.${id}"
started_file="${RUNDIR}/started.${id}"
printf '%s\n' "${cmd}" > "${cmd_file}"
: > "${output_file}"

runline="touch $(printf '%q' "${started_file}") ; { source $(printf '%q' "${cmd_file}") ; } > $(printf '%q' "${output_file}") 2>&1 ; echo \$? > $(printf '%q' "${rc_file}")"
tmux send-keys -t "${SESSION}" "${runline}" Enter

grace="${I4H_TMUX_START_GRACE:-25}"
waited=0
resent=0
while [[ ! -f "${rc_file}" ]]; do
    tmux has-session -t "${SESSION}" 2>/dev/null || {
        [[ -f "${output_file}" ]] && sed -n '1,200p' "${output_file}"
        echo "tmux-shell: session ended" >&2
        rm -f "${cmd_file}" "${output_file}" "${started_file}"
        exit 1
    }
    if [[ ! -f "${started_file}" ]]; then
        waited=$((waited + 1))
        if (( waited >= grace )); then
            if (( resent == 0 )); then
                tmux send-keys -t "${SESSION}" C-u 2>/dev/null
                tmux send-keys -t "${SESSION}" "${runline}" Enter
                resent=1
                waited=0
            else
                echo "tmux-shell: command did not execute; resetting session" >&2
                tmux kill-session -t "${SESSION}" 2>/dev/null
                rm -f "${cmd_file}" "${output_file}" "${started_file}"
                exit 1
            fi
        fi
    fi
    sleep 0.2
done

code="$(sed -n '1p' "${rc_file}" 2>/dev/null || echo 1)"
[[ -f "${output_file}" ]] && sed -n '1,100000p' "${output_file}"
rm -f "${cmd_file}" "${output_file}" "${rc_file}" "${started_file}"
exit "${code:-1}"
