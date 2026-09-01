#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Centralized pinned checkouts for workflow third-party source trees.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${SCRIPT_DIR}"
WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_PREFIX="i4h-workflows third_party setup"
TARGET="${1:-all}"
[ "$#" -le 1 ] || { echo "usage: third_party/setup.sh [all|arena|tasks/<policy>]" >&2; exit 2; }

ISAACLAB_REV="ffff603eafc6b74264a5261cc0183d6a65390d78"
ISAACLAB_DIR="IsaacLab-ffff603"
LEISAAC_REV="cd61a20c75f7b72c347538089602201349af6dc8"
LEISAAC_DIR="leisaac-cd61a20"
ISAACLAB_ARENA_REV="0a1b8c2345691c2f225b4a01b96dbe4d0aeb221c"
ISAACLAB_ARENA_DIR="IsaacLab-Arena-0a1b8c2"
ISAACSIM_SKILLS_REV="045ca8b59622b99a408092124377c66346e8d9c2"
ISAACSIM_SKILLS_DIR="IsaacSim-045ca8b"
# The three i4h component repositories track main so that workflow integration always builds
# against current upstream. Their directory names carry no revision for that reason: the uv
# source paths in arena/ and tools/patient_twin/ point here and must stay valid as main moves.
# Export the matching *_REF variable to pin one to a commit when bisecting a break.
I4H_PHYSICS_SIM_REF="${I4H_PHYSICS_SIM_REF:-main}"
I4H_PHYSICS_SIM_DIR="i4h-physics-simulation-internal"
I4H_SENSOR_SIM_REF="${I4H_SENSOR_SIM_REF:-main}"
I4H_SENSOR_SIM_DIR="i4h-sensor-simulation-internal"
I4H_DIGITAL_TWIN_REF="${I4H_DIGITAL_TWIN_REF:-main}"
I4H_DIGITAL_TWIN_DIR="i4h-digital-twin-internal"

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
RLINF_REV="a4b6abe205d7942f45cf3e8843c3e72ce818729d"
RLINF_DIR="RLinf-a4b6abe"

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
export GIT_LFS_SKIP_SMUDGE=1

mkdir -p "${THIRD_PARTY_DIR}"

# Private component repositories follow the transport used to check out this repository.
# Jenkins checks out over HTTPS and supplies an askpass credential around setup.sh, while
# developers who use an SSH origin can reuse their existing SSH agent. Archive/container
# builds have no root origin, so they retain the existing HTTPS behavior.
root_origin="$(git -C "${WORKFLOW_ROOT}" remote get-url origin 2>/dev/null || true)"
case "${root_origin}" in
  git@*:*|ssh://*|git+ssh://*) INTERNAL_GIT_TRANSPORT="ssh" ;;
  *) INTERNAL_GIT_TRANSPORT="https" ;;
esac
case "${INTERNAL_GIT_TRANSPORT}" in
  https) INTERNAL_GITHUB_BASE="https://github.com/isaac-for-healthcare" ;;
  ssh) INTERNAL_GITHUB_BASE="git@github.com:isaac-for-healthcare" ;;
esac
echo "[${LOG_PREFIX}] private repository transport: ${INTERNAL_GIT_TRANSPORT}"

ensure_origin_url() {
  local repo_dir="$1"
  local url="$2"
  local current_url

  current_url="$(git -C "${repo_dir}" remote get-url origin 2>/dev/null || true)"
  if [[ -z "${current_url}" ]]; then
    git -C "${repo_dir}" remote add origin "${url}"
  elif [[ "${current_url}" != "${url}" ]]; then
    git -C "${repo_dir}" remote set-url origin "${url}"
  fi
}

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
  fi
  ensure_origin_url "${repo_dir}" "${url}"
  if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
    # A commit never moves, so fetch it once and reuse it on every later run.
    if ! git -C "${repo_dir}" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
      git -C "${repo_dir}" fetch --depth=1 --no-tags --filter=blob:none origin "${ref}"
    fi
    target_commit="$(git -C "${repo_dir}" rev-parse "${ref}^{commit}")"
  else
    # A branch moves, so fetch it on every run and resolve through FETCH_HEAD. `fetch origin
    # main` writes no local ref, and once any commit of that name existed a name lookup would
    # freeze the checkout at whatever main was the day the machine was first set up.
    git -C "${repo_dir}" fetch --depth=1 --no-tags --filter=blob:none origin "${ref}"
    target_commit="$(git -C "${repo_dir}" rev-parse FETCH_HEAD)"
  fi
  current_commit="$(git -C "${repo_dir}" rev-parse --verify HEAD 2>/dev/null || true)"
  if [[ "${current_commit}" != "${target_commit}" ]]; then
    git -C "${repo_dir}" checkout -f "${target_commit}"
    echo "[${LOG_PREFIX}] ${name} @ ${ref} is now ${target_commit}"
  fi
}

checkout_sparse_ref() {
  local name="$1"
  local url="$2"
  local ref="$3"
  local sparse_path="$4"
  local repo_dir="${THIRD_PARTY_DIR}/${name}"
  local current_commit target_commit

  if [[ ! -d "${repo_dir}/.git" ]]; then
    echo "[${LOG_PREFIX}] initializing ${name} @ ${ref} (${sparse_path} only)"
    mkdir -p "${repo_dir}"
    git -C "${repo_dir}" init
  fi
  ensure_origin_url "${repo_dir}" "${url}"
  git -C "${repo_dir}" sparse-checkout init --cone
  git -C "${repo_dir}" sparse-checkout set "${sparse_path}"
  if ! git -C "${repo_dir}" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
    git -C "${repo_dir}" fetch --depth=1 --no-tags --filter=blob:none origin "${ref}"
  fi
  target_commit="$(git -C "${repo_dir}" rev-parse "${ref}^{commit}")"
  current_commit="$(git -C "${repo_dir}" rev-parse --verify HEAD 2>/dev/null || true)"
  if [[ "${current_commit}" != "${target_commit}" ]]; then
    git -C "${repo_dir}" checkout -f "${ref}"
  fi
}

arena_checkouts=(
  "${ISAACLAB_DIR}|https://github.com/isaac-sim/IsaacLab.git|${ISAACLAB_REV}"
  "${LEISAAC_DIR}|https://github.com/LightwheelAI/leisaac.git|${LEISAAC_REV}"
  "${ISAACLAB_ARENA_DIR}|https://github.com/isaac-sim/IsaacLab-Arena.git|${ISAACLAB_ARENA_REV}"
  "${I4H_PHYSICS_SIM_DIR}|${INTERNAL_GITHUB_BASE}/i4h-physics-simulation-internal.git|${I4H_PHYSICS_SIM_REF}"
  "${I4H_SENSOR_SIM_DIR}|${INTERNAL_GITHUB_BASE}/i4h-sensor-simulation-internal.git|${I4H_SENSOR_SIM_REF}"
  # tools/patient_twin, not arena, consumes this one. It rides along with the arena target so
  # that a scoped `I4H_THIRD_PARTY_TARGET=arena` still produces a runnable twin pipeline.
  "${I4H_DIGITAL_TWIN_DIR}|${INTERNAL_GITHUB_BASE}/i4h-digital-twin-internal.git|${I4H_DIGITAL_TWIN_REF}"
)
policy_checkouts=(
  "${GR00T_15_DIR}|https://github.com/NVIDIA/Isaac-GR00T.git|${GR00T_15_REV}"
  "${GR00T_16_DIR}|https://github.com/NVIDIA/Isaac-GR00T.git|${GR00T_16_REV}"
  "${GR00T_17_DIR}|https://github.com/NVIDIA/Isaac-GR00T.git|${GR00T_17_REV}"
  "${OPENPI_DIR_NAME}|https://github.com/Physical-Intelligence/openpi.git|${OPENPI_REV}"
  "${LEROBOT_DIR}|https://github.com/huggingface/lerobot.git|${LEROBOT_REV}"
  "${RLINF_DIR}|https://github.com/RLinf/RLinf.git|${RLINF_REV}"
)

checkouts=()
case "$TARGET" in
  all)
    checkouts=("${arena_checkouts[@]}" "${policy_checkouts[@]}")
    checkout_sparse_ref \
      "${ISAACSIM_SKILLS_DIR}" \
      "https://github.com/isaac-sim/IsaacSim.git" \
      "${ISAACSIM_SKILLS_REV}" \
      "skills"
    ;;
  arena)
    checkouts=("${arena_checkouts[@]}")
    checkout_sparse_ref \
      "${ISAACSIM_SKILLS_DIR}" \
      "https://github.com/isaac-sim/IsaacSim.git" \
      "${ISAACSIM_SKILLS_REV}" \
      "skills"
    ;;
  tasks/gr00t_n15)
    checkouts=("${policy_checkouts[0]}")
    ;;
  tasks/gr00t_n16)
    # N1.6 imports reusable configuration from IsaacLab-Arena.
    checkouts=("${arena_checkouts[2]}" "${policy_checkouts[1]}")
    ;;
  tasks/gr00t_n17)
    checkouts=("${policy_checkouts[2]}")
    ;;
  tasks/openpi_pi0)
    checkouts=("${policy_checkouts[3]}" "${policy_checkouts[4]}")
    ;;
  *)
    echo "unsupported third-party target: $TARGET" >&2
    exit 2
    ;;
esac

for spec in "${checkouts[@]}"; do
  IFS="|" read -r name url ref <<<"${spec}"
  checkout_ref "${name}" "${url}" "${ref}"
done

if [[ -d "${THIRD_PARTY_DIR}/${LEISAAC_DIR}/.git" ]]; then
  apply_patch_once "leisaac HDF5/CUDA" \
    "${THIRD_PARTY_DIR}/${LEISAAC_DIR}" \
    "${THIRD_PARTY_DIR}/leisaac_hdf5_cuda_fix.patch"

  apply_patch_once "LeIsaac Isaac Lab 3 ActionTermCfg export" \
    "${THIRD_PARTY_DIR}/${LEISAAC_DIR}" \
    "${THIRD_PARTY_DIR}/leisaac_isaaclab3_action_term_cfg.patch"
fi

if [[ -d "${THIRD_PARTY_DIR}/${ISAACLAB_ARENA_DIR}/.git" ]]; then
  apply_patch_once "IsaacLab-Arena G1 WBC default_base_height cfg" \
    "${THIRD_PARTY_DIR}/${ISAACLAB_ARENA_DIR}" \
    "${THIRD_PARTY_DIR}/isaaclab_arena_wbc_default_base_height.patch"

  apply_patch_once "IsaacLab-Arena Newton import" \
    "${THIRD_PARTY_DIR}/${ISAACLAB_ARENA_DIR}" \
    "${THIRD_PARTY_DIR}/isaaclab_arena_newton_import.patch"

  apply_patch_once "IsaacLab-Arena lazy registration" \
    "${THIRD_PARTY_DIR}/${ISAACLAB_ARENA_DIR}" \
    "${THIRD_PARTY_DIR}/isaaclab_arena_lazy_registration.patch"
fi

if [[ -d "${THIRD_PARTY_DIR}/${LEROBOT_DIR}/.git" ]]; then
  apply_patch_once "LeRobot datasets>=4 + pyav VideoReader compat" \
    "${THIRD_PARTY_DIR}/${LEROBOT_DIR}" \
    "${THIRD_PARTY_DIR}/lerobot_datasets_v4_compat.patch"
fi

if [[ -d "${THIRD_PARTY_DIR}/${GR00T_15_DIR}/.git" ]]; then
  apply_patch_once "Isaac-GR00T N1.5 action-head future tokens" \
    "${THIRD_PARTY_DIR}/${GR00T_15_DIR}" \
    "${THIRD_PARTY_DIR}/gr00t_action_head_future_tokens.patch"
fi

OPENPI_DIR="${THIRD_PARTY_DIR}/${OPENPI_DIR_NAME}"
OPENPI_PYPROJECT="${OPENPI_DIR}/pyproject.toml"
OPENPI_UTILS="${OPENPI_DIR}/src/openpi/training/utils.py"
OPENPI_LEROBOT_SOURCE="lerobot = { path = \"../${LEROBOT_DIR}\", editable = true }"

if [[ -d "${OPENPI_DIR}/.git" ]] && grep -q "opt_state: optax\.OptState" "${OPENPI_UTILS}"; then
  echo "[${LOG_PREFIX}] patching ${OPENPI_UTILS}"
  sed -i -e 's/opt_state: optax\.OptState/opt_state: Any/' "${OPENPI_UTILS}"
fi
if [[ -d "${OPENPI_DIR}/.git" ]] && grep -q '"jax\[cuda12\]==0\.5\.0"' "${OPENPI_PYPROJECT}"; then
  echo "[${LOG_PREFIX}] patching ${OPENPI_PYPROJECT} (jax 0.5.0 -> 0.5.3)"
  sed -i -e 's/"jax\[cuda12\]==0\.5\.0"/"jax[cuda12]==0.5.3"/' "${OPENPI_PYPROJECT}"
fi

LEROBOT_PYPROJECT="${THIRD_PARTY_DIR}/${LEROBOT_DIR}/pyproject.toml"
if [[ -f "${LEROBOT_PYPROJECT}" ]] && grep -q -E '(^|")pyav' "${LEROBOT_PYPROJECT}"; then
  echo "[${LOG_PREFIX}] patching lerobot pyproject pyav -> av"
  sed -i -E -e 's/^pyav([[:space:]]*=)/av\1/g' -e 's/"pyav/"av/g' "${LEROBOT_PYPROJECT}"
fi

if [[ -f "${OPENPI_PYPROJECT}" ]] && ! grep -Fxq "${OPENPI_LEROBOT_SOURCE}" "${OPENPI_PYPROJECT}"; then
  echo "[${LOG_PREFIX}] rewriting openpi's lerobot source to local path"
  sed -i -E -e "s#^lerobot = \\{ (git|path) = .*#${OPENPI_LEROBOT_SOURCE}#" "${OPENPI_PYPROJECT}"
fi

if [[ ! -f "${OPENPI_DIR}/src/openpi/train.py" && -f "${OPENPI_DIR}/scripts/train.py" ]]; then
  cp "${OPENPI_DIR}/scripts/train.py" "${OPENPI_DIR}/src/openpi/train.py"
fi
if [[ ! -f "${OPENPI_DIR}/src/openpi/compute_norm_stats.py" && -f "${OPENPI_DIR}/scripts/compute_norm_stats.py" ]]; then
  cp "${OPENPI_DIR}/scripts/compute_norm_stats.py" "${OPENPI_DIR}/src/openpi/compute_norm_stats.py"
fi
