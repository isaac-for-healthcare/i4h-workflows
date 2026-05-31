#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Centralized fast checkouts for agentic third-party source trees.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${SCRIPT_DIR}"

LOG_PREFIX="agentic third_party setup"

ISAACLAB_REV="2107794a21bae76a578e896ee9424377c24c2ab0"
ISAACLAB_DIR="IsaacLab-2107794"
LEISAAC_REV="cd61a20c75f7b72c347538089602201349af6dc8"
LEISAAC_DIR="leisaac-cd61a20"
ISAACLAB_ARENA_REV="dba09956588dddae52897820686efd329d85da12"
ISAACLAB_ARENA_DIR="IsaacLab-Arena-dba0995"

GR00T_15_REV="17a77ebf646cf13460cdbc8f49f9ec7d0d63bcb1"
GR00T_15_DIR="Isaac-GR00T-1.5"
GR00T_16_REV="e8e625f4f21898c506a1d8f7d20a289c97a52acf"
GR00T_16_DIR="Isaac-GR00T-1.6"
GR00T_17_REV="4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e"
GR00T_17_DIR="Isaac-GR00T-1.7"

OPENPI_REV="581e07d73af36d336cef1ec9d7172553b2332193"
OPENPI_DIR_NAME="openpi-581e07d"
LEROBOT_REV="6674e368249472c91382eb54bb8501c94c7f0c56"
LEROBOT_DIR="lerobot-6674e36"

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
export GIT_LFS_SKIP_SMUDGE=1

mkdir -p "${THIRD_PARTY_DIR}"

apply_patch_once() {
  local label="$1"
  local repo_dir="$2"
  local patch_file="$3"

  if git -C "${repo_dir}" apply --check -R "${patch_file}" >/dev/null 2>&1; then
    echo "[${LOG_PREFIX}] ${label} patch already applied"
    return
  fi
  echo "[${LOG_PREFIX}] applying ${label} patch"
  git -C "${repo_dir}" apply "${patch_file}"
}

checkout_ref() {
  local name="$1"
  local url="$2"
  local ref="$3"
  local repo_dir="${THIRD_PARTY_DIR}/${name}"
  local current_commit target_commit

  if [[ ! -d "${repo_dir}/.git" ]]; then
    echo "[${LOG_PREFIX}] initializing ${name} @ ${ref}"
    mkdir -p "${repo_dir}"
    git -C "${repo_dir}" init
    git -C "${repo_dir}" remote add origin "${url}"
  fi
  if ! git -C "${repo_dir}" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
    git -C "${repo_dir}" fetch --depth=1 --no-tags --filter=blob:none origin "${ref}"
  fi
  target_commit="$(git -C "${repo_dir}" rev-parse "${ref}^{commit}")"
  current_commit="$(git -C "${repo_dir}" rev-parse --verify HEAD 2>/dev/null || true)"
  if [[ "${current_commit}" != "${target_commit}" ]]; then
    git -C "${repo_dir}" checkout -f "${ref}"
  fi
}

wait_for_checkouts() {
  local failed=0
  local pid

  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  if (( failed )); then
    echo "[${LOG_PREFIX}] one or more third-party checkouts failed" >&2
    exit 1
  fi
}

checkout_pids=()
checkouts=(
  "${ISAACLAB_DIR}|https://github.com/isaac-sim/IsaacLab.git|${ISAACLAB_REV}"
  "${LEISAAC_DIR}|https://github.com/LightwheelAI/leisaac.git|${LEISAAC_REV}"
  "${ISAACLAB_ARENA_DIR}|https://github.com/isaac-sim/IsaacLab-Arena.git|${ISAACLAB_ARENA_REV}"
  "${GR00T_15_DIR}|https://github.com/NVIDIA/Isaac-GR00T.git|${GR00T_15_REV}"
  "${GR00T_16_DIR}|https://github.com/NVIDIA/Isaac-GR00T.git|${GR00T_16_REV}"
  "${GR00T_17_DIR}|https://github.com/NVIDIA/Isaac-GR00T.git|${GR00T_17_REV}"
  "${OPENPI_DIR_NAME}|https://github.com/Physical-Intelligence/openpi.git|${OPENPI_REV}"
  "${LEROBOT_DIR}|https://github.com/huggingface/lerobot.git|${LEROBOT_REV}"
)

for spec in "${checkouts[@]}"; do
  IFS="|" read -r name url ref <<<"${spec}"
  checkout_ref "${name}" "${url}" "${ref}" &
  checkout_pids+=("$!")
done

wait_for_checkouts "${checkout_pids[@]}"

apply_patch_once "leisaac HDF5/CUDA" \
  "${THIRD_PARTY_DIR}/${LEISAAC_DIR}" \
  "${THIRD_PARTY_DIR}/leisaac_hdf5_cuda_fix.patch"

apply_patch_once "IsaacLab-Arena G1 WBC default_base_height cfg" \
  "${THIRD_PARTY_DIR}/${ISAACLAB_ARENA_DIR}" \
  "${THIRD_PARTY_DIR}/isaaclab_arena_wbc_default_base_height.patch"

apply_patch_once "LeRobot datasets>=4 + pyav VideoReader compat" \
  "${THIRD_PARTY_DIR}/${LEROBOT_DIR}" \
  "${THIRD_PARTY_DIR}/lerobot_datasets_v4_compat.patch"

apply_patch_once "Isaac-GR00T N1.5 action-head future tokens" \
  "${THIRD_PARTY_DIR}/${GR00T_15_DIR}" \
  "${THIRD_PARTY_DIR}/gr00t_action_head_future_tokens.patch"

OPENPI_DIR="${THIRD_PARTY_DIR}/${OPENPI_DIR_NAME}"
OPENPI_PYPROJECT="${OPENPI_DIR}/pyproject.toml"
OPENPI_UTILS="${OPENPI_DIR}/src/openpi/training/utils.py"
OPENPI_LEROBOT_SOURCE="lerobot = { path = \"../${LEROBOT_DIR}\", editable = true }"

if grep -q "opt_state: optax\.OptState" "${OPENPI_UTILS}"; then
  echo "[${LOG_PREFIX}] patching ${OPENPI_UTILS}"
  sed -i -e 's/opt_state: optax\.OptState/opt_state: Any/' "${OPENPI_UTILS}"
fi
if grep -q '"jax\[cuda12\]==0\.5\.0"' "${OPENPI_PYPROJECT}"; then
  echo "[${LOG_PREFIX}] patching ${OPENPI_PYPROJECT} (jax 0.5.0 -> 0.5.3)"
  sed -i -e 's/"jax\[cuda12\]==0\.5\.0"/"jax[cuda12]==0.5.3"/' "${OPENPI_PYPROJECT}"
fi

LEROBOT_PYPROJECT="${THIRD_PARTY_DIR}/${LEROBOT_DIR}/pyproject.toml"
if grep -q -E '(^|")pyav' "${LEROBOT_PYPROJECT}"; then
  echo "[${LOG_PREFIX}] patching lerobot pyproject pyav -> av"
  sed -i -E -e 's/^pyav([[:space:]]*=)/av\1/g' -e 's/"pyav/"av/g' "${LEROBOT_PYPROJECT}"
fi

if ! grep -Fxq "${OPENPI_LEROBOT_SOURCE}" "${OPENPI_PYPROJECT}"; then
  echo "[${LOG_PREFIX}] rewriting openpi's lerobot source to local path"
  sed -i -E -e "s#^lerobot = \\{ (git|path) = .*#${OPENPI_LEROBOT_SOURCE}#" "${OPENPI_PYPROJECT}"
fi

if [[ ! -f "${OPENPI_DIR}/src/openpi/train.py" && -f "${OPENPI_DIR}/scripts/train.py" ]]; then
  cp "${OPENPI_DIR}/scripts/train.py" "${OPENPI_DIR}/src/openpi/train.py"
fi
if [[ ! -f "${OPENPI_DIR}/src/openpi/compute_norm_stats.py" && -f "${OPENPI_DIR}/scripts/compute_norm_stats.py" ]]; then
  cp "${OPENPI_DIR}/scripts/compute_norm_stats.py" "${OPENPI_DIR}/src/openpi/compute_norm_stats.py"
fi
