#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="${I4H_WORKFLOWS:-/workspace}"
STATE_DIR="${I4H_STATE_DIR:-/opt/i4h-state}"
IMAGE_FLAVOR="${I4H_IMAGE_FLAVOR:-default}"
export I4H_WORKFLOWS="$ROOT"
if [ "$IMAGE_FLAVOR" = full ]; then
  export I4H_VENV_ROOT="${I4H_BAKED_VENV_ROOT:?full image is missing I4H_BAKED_VENV_ROOT}"
  export UV_PYTHON_INSTALL_DIR="${I4H_BAKED_PYTHON_INSTALL_DIR:?full image is missing I4H_BAKED_PYTHON_INSTALL_DIR}"
  export UV_NO_SYNC=1
else
  export I4H_VENV_ROOT="${I4H_VENV_ROOT:-$STATE_DIR/venvs}"
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$STATE_DIR/python}"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STATE_DIR/uv-cache}"
export HF_HOME="${HF_HOME:-$STATE_DIR/huggingface}"
DEFAULT_SETUP_FILE="${I4H_DEFAULT_SETUP_FILE:-/opt/i4h-default-setup-projects}"

if [ -f "$DEFAULT_SETUP_FILE" ]; then
  export I4H_SETUP_PROJECTS="${I4H_SETUP_PROJECTS:-$(<"$DEFAULT_SETUP_FILE")}"
fi

mkdir -p \
  "$I4H_VENV_ROOT" \
  "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" \
  "$HF_HOME" \
  "$STATE_DIR/runs" \
  "$STATE_DIR/isaac/cache" \
  "$STATE_DIR/isaac/compute-cache" \
  "$STATE_DIR/isaac/config" \
  "$STATE_DIR/isaac/data"
cd "$ROOT"

if [ "${1:-}" = i4h-annotator ]; then
  export I4H_SETUP_PROJECTS=tools/annotator
  shift
  set -- uv run --no-sync --project tools/annotator i4h-annotator "$@"
fi

if [ "${1:-}" = ./run.sh ]; then
  case "${2:-}" in
    list|show|lint) export I4H_SETUP_PROJECTS=workflows ;;
  esac
fi

if [ "${1:-}" = i4h-policy ] || [ "${1:-}" = i4h-workflows-policy ]; then
  shift
  policy_stack="${I4H_POLICY_STACK:-gr00t_n15}"
  expect_stack=0
  for arg in "$@"; do
    if [ "$expect_stack" -eq 1 ]; then
      policy_stack="$arg"
      expect_stack=0
      continue
    fi
    case "$arg" in
      --stack) expect_stack=1 ;;
      --stack=*) policy_stack="${arg#*=}" ;;
      */*) policy_stack="${arg%%/*}"; break ;;
    esac
  done
  export I4H_SETUP_PROJECTS="tasks/$policy_stack"
  export I4H_THIRD_PARTY_TARGET="tasks/$policy_stack"
  set -- "${I4H_POLICY_COMMAND:-$ROOT/docker/policy-server.sh}" "$@"
fi

skip_setup="${I4H_SKIP_SETUP:-0}"
if [ "$IMAGE_FLAVOR" = full ]; then
  baked_marker="${I4H_FULL_SETUP_MARKER:-/opt/i4h-full/setup-complete}"
  if [ ! -f "$baked_marker" ] || [ "$(<"$baked_marker")" != "$(<"$ROOT/.i4h-dependency-fingerprint")" ]; then
    echo "full image environments do not match this source tree" >&2
    exit 1
  fi
  skip_setup=1
  echo "==> using environments included in the full image"
fi

if [ "$skip_setup" != "1" ]; then
  dependency_fingerprint="$(<"$ROOT/.i4h-dependency-fingerprint")"
  runtime_selection="${I4H_SETUP_PROJECTS:-all}"
  selection_key="$(printf '%s' "$runtime_selection" | sha256sum | cut -c1-16)"
  runtime_fingerprint="${dependency_fingerprint}:${runtime_selection}"
  state_fingerprint="$STATE_DIR/dependency-fingerprint-$selection_key"

  # A single state volume may be shared by successive containers. Serialize
  # environment creation so two first starts cannot mutate the same uv cache.
  exec 9>"$STATE_DIR/setup.lock"
  flock 9
  if [ -f "$state_fingerprint" ] \
    && [ "$(<"$state_fingerprint")" = "$runtime_fingerprint" ] \
    && ./setup.sh links >/dev/null 2>&1; then
    echo "==> reusing persistent i4h environments from $I4H_VENV_ROOT"
  else
    echo "==> syncing persistent i4h environments (uv cache: $UV_CACHE_DIR)"
    ./setup.sh all
    printf '%s\n' "$runtime_fingerprint" > "$state_fingerprint"
  fi
  flock -u 9
fi

if [ "$#" -eq 0 ]; then
  set -- bash
fi
exec "$@"
