#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ALL_ENVS=(scissor_pick_and_place locomanip_tray_pick_and_place locomanip_push_cart assemble_trocar)

subproject_for() {
  case "$1" in
    scissor_pick_and_place) echo gr00t_n15 ;;
    locomanip_tray_pick_and_place) echo gr00t_n16 ;;
    locomanip_push_cart) echo gr00t_n16 ;;
    assemble_trocar) echo gr00t_n15 ;;
    *) return 1 ;;
  esac
}

ENV=""; FORCE=0
for a in "$@"; do
  case "${prev:-}" in --env) ENV="$a" ;; esac
  case "$a" in
    --env=*) ENV="${a#--env=}" ;;
    --force|-f) FORCE=1 ;;
  esac
  prev="$a"
done

has_engines() {
  compgen -G "${POLICY_DIR}/$1/trt_engines/$2/engines/*.engine" > /dev/null
}

build_one() {
  local env="$1" sub
  sub="$(subproject_for "${env}")" || { echo "unsupported env '${env}'" >&2; return 2; }
  if (( ! FORCE )) && has_engines "${sub}" "${env}"; then
    echo "[build_trt] ${env}: engines already present in ${sub}/trt_engines/${env}/engines — skip (--force to rebuild)"
    return 0
  fi
  echo "=============================================="
  echo "[build_trt] ${env}  (${sub})"
  echo "=============================================="
  "${POLICY_DIR}/${sub}/scripts/build_trt_engines.sh" --env "${env}"
}

if [[ -z "${ENV}" ]]; then
  for env in "${ALL_ENVS[@]}"; do build_one "${env}"; done
else
  build_one "${ENV}"
fi
