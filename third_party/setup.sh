#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Centralized pinned checkouts for workflow third-party source trees.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${SCRIPT_DIR}"

LOG_PREFIX="i4h-workflows third_party setup"
TARGET="${1:-all}"
[ "$#" -le 1 ] || { echo "usage: third_party/setup.sh [all|arena|tasks/<policy>]" >&2; exit 2; }

# Isaac Lab v3.0.0 GA. It requires Isaac Sim 6.2.0.0 (arena/pyproject.toml) and warp-lang 1.17.0.
ISAACLAB_REV="e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"
ISAACLAB_DIR="IsaacLab-e3f7d9f"
LEISAAC_REV="cd61a20c75f7b72c347538089602201349af6dc8"
LEISAAC_DIR="leisaac-cd61a20"
# IsaacLab-Arena release/0.3.1. Upstream validates it on Isaac Lab 91a0906f (pre-GA) and Isaac
# Sim 6.1; here it runs on Isaac Lab v3.0.0 GA instead.
ISAACLAB_ARENA_REV="c7edf3a9d5d8276565208f72422f8f8a0806d6cf"
ISAACLAB_ARENA_DIR="IsaacLab-Arena-c7edf3a"
ISAACSIM_SKILLS_REV="74a9356b20250dd60337d90b4b4d913c9454d32b"
ISAACSIM_SKILLS_DIR="IsaacSim-74a9356"
# Release component versions for v0.9.0: physics simulation, sensor simulation and digital
# twin default to the commits their v0.9.0 tags name. Every component repository is public.
# Keep the checkout directory names stable because uv path sources refer to them. Export
# *_REF to bisect or test another component revision; the solver compatibility check still
# applies.
I4H_PHYSICS_SIM_REF="${I4H_PHYSICS_SIM_REF:-50056196775fc0ed8cca03bf16144fe71d183ca7}"
I4H_PHYSICS_SIM_DIR="i4h-physics-simulation-internal"
I4H_SENSOR_SIM_REF="${I4H_SENSOR_SIM_REF:-5d6ba8547a4c6e25af7eaf551f6a34371ccd6350}"
I4H_SENSOR_SIM_DIR="i4h-sensor-simulation-internal"
I4H_DIGITAL_TWIN_REF="${I4H_DIGITAL_TWIN_REF:-6b6470cff156ba15b573ae1b4b44c3a77654b8b3}"
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
  elif git -C "${repo_dir}" rev-parse --verify --quiet "refs/tags/${ref}^{commit}" >/dev/null; then
    # A release tag does not move either. Its first fetch recorded it below, so later runs
    # resolve it offline. That includes a container's first start, which reuses the
    # checkouts that the image build fetched.
    target_commit="$(git -C "${repo_dir}" rev-parse "refs/tags/${ref}^{commit}")"
  else
    # A branch moves, so fetch it on every run and resolve through FETCH_HEAD. `fetch origin
    # main` writes no local ref, and once any commit of that name existed a name lookup would
    # freeze the checkout at whatever main was the day the machine was first set up. Peel to
    # the commit: for an annotated release tag FETCH_HEAD names the tag object, which never
    # equals HEAD, so every rerun would force a checkout over local edits.
    git -C "${repo_dir}" fetch --depth=1 --no-tags --filter=blob:none origin "${ref}"
    target_commit="$(git -C "${repo_dir}" rev-parse "FETCH_HEAD^{commit}")"
    # FETCH_HEAD records whether the remote resolved the name as a tag or a branch. Keep a tag
    # so the branch above finds it next time; a branch gets no local ref and is fetched again.
    if grep -qsF $'\t'"tag '${ref}' of " "$(git -C "${repo_dir}" rev-parse --absolute-git-dir)/FETCH_HEAD"; then
      git -C "${repo_dir}" update-ref "refs/tags/${ref}" "${target_commit}"
    fi
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
  "${I4H_PHYSICS_SIM_DIR}|https://github.com/isaac-for-healthcare/i4h-physics-simulation.git|${I4H_PHYSICS_SIM_REF}"
  "${I4H_SENSOR_SIM_DIR}|https://github.com/isaac-for-healthcare/i4h-sensor-simulation.git|${I4H_SENSOR_SIM_REF}"
  # tools/patient_twin, not arena, consumes this one. It rides along with the arena target so
  # that a scoped `I4H_THIRD_PARTY_TARGET=arena` still produces a runnable twin pipeline.
  "${I4H_DIGITAL_TWIN_DIR}|https://github.com/isaac-for-healthcare/i4h-digital-twin.git|${I4H_DIGITAL_TWIN_REF}"
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

# A ref that resolves cleanly can still be the wrong solver. The catheter scenes need a
# coupled-arm contract that i4h-physics-simulation has carried only since v0.9.0,
# so an older pin or a bisect ref can land a tree without it, and then setup succeeds and
# the run fails much later as a missing attribute during solver initialization -- which
# reads like a scene bug. Fail here instead, while the ref that produced the tree is still
# in hand and before Isaac Sim or RLinf has started.
#
# A presence check on the source rather than a signature check: these are Warp solver methods
# whose call sites are built at runtime, so there is nothing importable to introspect without
# standing up the solver, and `proximal_reaction` may be a property rather than a def. A ref
# that lacks the contract does not mention the names at all, which is the case worth catching.
if [[ -d "${THIRD_PARTY_DIR}/${I4H_PHYSICS_SIM_DIR}/.git" ]]; then
  missing_solver_api=()
  for symbol in register_rod set_root_pose_gpu proximal_reaction; do
    grep -rqw --include='*.py' "${symbol}" "${THIRD_PARTY_DIR}/${I4H_PHYSICS_SIM_DIR}" \
      || missing_solver_api+=("${symbol}")
  done
  if [[ "${#missing_solver_api[@]}" -gt 0 ]]; then
    echo "[${LOG_PREFIX}] i4h-physics-simulation @ ${I4H_PHYSICS_SIM_REF} does not carry the" >&2
    echo "[${LOG_PREFIX}] coupled-arm solver contract: ${missing_solver_api[*]}" >&2
    echo "[${LOG_PREFIX}] The catheter arm scenes need physics simulation v0.9.0 or later. Unset" >&2
    echo "[${LOG_PREFIX}] I4H_PHYSICS_SIM_REF to use the release default, or set it to a commit that" >&2
    echo "[${LOG_PREFIX}] carries the contract." >&2
    exit 1
  fi
  echo "[${LOG_PREFIX}] i4h-physics-simulation carries the coupled-arm solver contract"

  # Isaac Lab 3.0 no longer holds the finalized ModelBuilder on NewtonManager._builder, where
  # the coupled arm manager reads drive-body names from; it lives in the backend registry.
  apply_patch_once "i4h-physics-simulation Isaac Lab 3 ModelBuilder lookup" \
    "${THIRD_PARTY_DIR}/${I4H_PHYSICS_SIM_DIR}" \
    "${THIRD_PARTY_DIR}/i4h_physics_simulation_isaaclab3_builder.patch"
fi

if [[ -d "${THIRD_PARTY_DIR}/${ISAACLAB_DIR}/.git" ]]; then
  # Isaac Lab 3.0 resolves the Pink IK mesh_path as a single file, but Pinocchio takes a mesh
  # package directory; the H2 + Sharpa URDF needs one for its relative mesh paths.
  apply_patch_once "Isaac Lab Pink IK local mesh directory" \
    "${THIRD_PARTY_DIR}/${ISAACLAB_DIR}" \
    "${THIRD_PARTY_DIR}/isaaclab_pink_ik_local_mesh_dir.patch"
fi

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

  # Arena 0.3.1 was validated on pre-GA Isaac Lab 91a0906f. On v3.0.0, resets and recording pass
  # slice(None) env_ids, mdp.image became per-modality terms (image_rgb, image_depth, ...), and the
  # legacy Panda with its shoulder/forearm/hand actuator groups moved to FRANKA_PANDA_LEGACY_HIGH_PD_CFG.
  apply_patch_once "IsaacLab-Arena Isaac Lab 3.0 GA slice env_ids" \
    "${THIRD_PARTY_DIR}/${ISAACLAB_ARENA_DIR}" \
    "${THIRD_PARTY_DIR}/isaaclab_arena_isaaclab3_slice_env_ids.patch"

  apply_patch_once "IsaacLab-Arena Isaac Lab 3.0 GA image observation terms" \
    "${THIRD_PARTY_DIR}/${ISAACLAB_ARENA_DIR}" \
    "${THIRD_PARTY_DIR}/isaaclab_arena_isaaclab3_image_terms.patch"

  apply_patch_once "IsaacLab-Arena Isaac Lab 3.0 GA legacy Franka actuator groups" \
    "${THIRD_PARTY_DIR}/${ISAACLAB_ARENA_DIR}" \
    "${THIRD_PARTY_DIR}/isaaclab_arena_isaaclab3_franka_legacy.patch"
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

if [[ -d "${THIRD_PARTY_DIR}/${RLINF_DIR}/.git" ]]; then
  apply_patch_once "RLinf video encoding and collective error propagation" \
    "${THIRD_PARTY_DIR}/${RLINF_DIR}" \
    "${THIRD_PARTY_DIR}/rlinf_training_failures.patch"

  apply_patch_once "RLinf GR00T N1.7 patch embedding and optimizer progress" \
    "${THIRD_PARTY_DIR}/${RLINF_DIR}" \
    "${THIRD_PARTY_DIR}/rlinf_gr00t_training.patch"
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
