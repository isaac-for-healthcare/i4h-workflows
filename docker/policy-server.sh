#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
  echo "usage: i4h-policy <workflow|task-id> [--stack NAME] [--listen ENDPOINT] [--checkpoint PATH] [--namespace NAME] [--verbose]" >&2
}

require_value() {
  [ "$2" -ge 2 ] || { echo "$1 requires a value" >&2; usage; exit 2; }
}

stack="${I4H_POLICY_STACK:-gr00t_n15}"
target=""
namespace=""
checkpoint="${I4H_POLICY_CHECKPOINT:-}"
connect="${I4H_ZENOH_CONNECT:-}"
listen="${I4H_POLICY_LISTEN:-tcp/0.0.0.0:7448}"
verbose=0

while [ $# -gt 0 ]; do
  case "$1" in
    --connect) require_value "$1" "$#"; connect="$2"; listen=""; shift 2 ;;
    --connect=*) connect="${1#*=}"; listen=""; shift ;;
    --stack) require_value "$1" "$#"; stack="$2"; shift 2 ;;
    --stack=*) stack="${1#*=}"; shift ;;
    --listen) require_value "$1" "$#"; listen="$2"; connect=""; shift 2 ;;
    --listen=*) listen="${1#*=}"; connect=""; shift ;;
    --checkpoint) require_value "$1" "$#"; checkpoint="$2"; shift 2 ;;
    --checkpoint=*) checkpoint="${1#*=}"; shift ;;
    --namespace) require_value "$1" "$#"; namespace="$2"; shift 2 ;;
    --namespace=*) namespace="${1#*=}"; shift ;;
    --verbose) verbose=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage; exit 2 ;;
    *)
      [ -z "$target" ] || { echo "only one workflow or task ID may be provided" >&2; usage; exit 2; }
      target="$1"
      shift
      ;;
  esac
done

[ -n "$target" ] || { usage; exit 2; }

case "$target" in
  */*)
    task_stack="${target%%/*}"
    stack="$task_stack"
    task="$target"
    workflow="${target#*/}"
    ;;
  *)
    workflow="$target"
    task="$stack/$workflow"
    ;;
esac

case "$stack" in
  gr00t_n15|gr00t_n16|gr00t_n17|openpi_pi0) ;;
  *) echo "unsupported policy stack: $stack" >&2; exit 2 ;;
esac

project="tasks/$stack"
manifest="$project/i4h_tasks/$stack/manifest/${task#*/}.yaml"
[ -f "$manifest" ] || { echo "policy task $task is not available" >&2; exit 2; }
[ -n "$namespace" ] || namespace="$workflow"
if [ -n "$connect" ]; then
  export I4H_ZENOH_CONNECT="$connect"
else
  unset I4H_ZENOH_CONNECT
fi
if [ -n "$listen" ]; then
  export I4H_ZENOH_LISTEN="$listen"
else
  unset I4H_ZENOH_LISTEN
fi

args=(--namespace "$namespace" --preload "$task")
[ -z "$checkpoint" ] || args+=(--checkpoint "$checkpoint")
[ "$verbose" -ne 1 ] || args+=(--verbose)

echo "==> policy $stack serving $task on namespace $namespace"
exec env -u VIRTUAL_ENV uv run --project "$project" --no-sync python -m "i4h_tasks.${stack}.server" "${args[@]}"
