#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sync every uv project. Each remains independent with its own lock:
#
#   ./setup.sh                everything (same as: ./setup.sh all)
#   ./setup.sh all            everything
#   ./setup.sh links          restore .venv links for an external venv root
#   ./setup.sh clean          delete caches, venvs and third_party checkouts
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git is required for third_party checkouts" >&2; exit 1; }

LIGHT=(common engine tasks/basic tasks/rsl_rl workflows rl)
TASKS=(tasks/ik tasks/teleop tasks/gr00t_n15 tasks/gr00t_n16 tasks/gr00t_n17 tasks/openpi_pi0)
TOOLS=(tools/mimic tools/dataset tools/cosmos tools/annotator tools/patient_twin)
all_projects=("${LIGHT[@]}" "${TASKS[@]}" "${TOOLS[@]}" arena)
ordered=("${all_projects[@]}")

# A specialized container command can select only the environment it needs.
# The default remains the complete host setup.
if [ -n "${I4H_SETUP_PROJECTS:-}" ]; then
  read -r -a requested_projects <<<"$I4H_SETUP_PROJECTS"
  ordered=()
  for requested in "${requested_projects[@]}"; do
    known=0
    for project in "${all_projects[@]}"; do
      if [ "$requested" = "$project" ]; then
        known=1
        break
      fi
    done
    [ "$known" -eq 1 ] || { echo "unknown I4H_SETUP_PROJECTS entry: $requested" >&2; exit 2; }
    ordered+=("$requested")
  done
  [ "${#ordered[@]}" -gt 0 ] || { echo "I4H_SETUP_PROJECTS selected no projects" >&2; exit 2; }
fi

# Containers keep environments and uv's cache on the same persistent volume.
# Project-local .venv symlinks preserve every existing launcher and tool
# contract while the actual environments survive container replacement.
VENV_ROOT="${I4H_VENV_ROOT:-}"
if [ -n "$VENV_ROOT" ]; then
  case "$VENV_ROOT" in
    /*) ;;
    *) echo "I4H_VENV_ROOT must be an absolute path: $VENV_ROOT" >&2; exit 2 ;;
  esac
  [ "$VENV_ROOT" != "/" ] || { echo "I4H_VENV_ROOT cannot be /" >&2; exit 2; }
  mkdir -p "$VENV_ROOT"
fi

usage() {
  cat <<EOF
usage: ./setup.sh [all|links|clean]

With no argument, syncs everything.

  all         sync every component
  links       restore project .venv links to existing I4H_VENV_ROOT environments
  clean       delete caches, venvs, and third_party checkouts

Set I4H_VENV_ROOT to an absolute directory to store component environments
outside the checkout. The project .venv paths become compatibility symlinks.
Set I4H_SETUP_PROJECTS to a space-separated project subset for a specialized
container; ordinary host setup should leave it unset.
Set I4H_UV_SYNC_ARGS only for controlled automation that needs extra uv sync
flags.
EOF
}

venv_path() {
  printf '%s/%s\n' "$VENV_ROOT" "$1"
}

link_external_venv() {
  local project="$1"
  local require_existing="${2:-0}"
  local link="$project/.venv"
  local target
  target="$(venv_path "$project")"

  mkdir -p "$(dirname "$target")"
  if [ -L "$link" ]; then
    if [ "$(readlink "$link")" != "$target" ]; then
      rm -f "$link"
    fi
  elif [ -e "$link" ]; then
    echo "refusing to replace the existing environment at $link" >&2
    echo "remove it or run without I4H_VENV_ROOT" >&2
    return 1
  fi

  if [ "$require_existing" -eq 1 ] && [ ! -d "$target" ]; then
    echo "external environment is missing: $target" >&2
    return 1
  fi
  [ -L "$link" ] || ln -s "$target" "$link"
}

# Asked of the project rather than kept as a list here: a new project that
# points a source at third_party/ gets the checkouts without touching setup.sh.
need_third_party() {
  grep -q "third_party/" "$1/pyproject.toml" 2>/dev/null
}

[ $# -eq 0 ] && set -- all
[ "$#" -eq 1 ] || { usage >&2; exit 2; }
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
  exit 0
fi

if [ "$1" = "clean" ]; then
  count=0
  while IFS= read -r -d '' path; do
    rm -rf "$path"; count=$((count + 1))
  done < <(find . -name .venv -prune -o -name third_party -prune -o \
    \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name '*.egg-info' \) \
    -print0)
  find . -name .venv -prune -o -name '*.pyc' -print0 | xargs -0r rm -f
  echo "clean: removed $count cache directories"

  count=0
  while IFS= read -r -d '' path; do
    rm -rf "$path"; count=$((count + 1))
  done < <(find . -name third_party -prune -o -name .venv -print0)
  echo "clean: removed $count venvs"

  if [ -n "$VENV_ROOT" ]; then
    count=0
    for project in "${ordered[@]}"; do
      target="$(venv_path "$project")"
      if [ -d "$target" ]; then
        rm -rf "$target"
        count=$((count + 1))
      fi
    done
    echo "clean: removed $count external venvs from $VENV_ROOT"
  fi

  # Everything that is not ours: the keep-list is exactly what
  # third_party/.gitignore whitelists, so clean leaves the tracked files and
  # takes the rest — checkouts, stray downloads, anything.
  count=0
  while IFS= read -r -d '' path; do
    rm -rf "$path"; count=$((count + 1))
  done < <(find third_party -mindepth 1 -maxdepth 1 \
    ! -name .gitignore ! -name setup.sh ! -name '*.patch' -print0)
  echo "clean: removed $count third_party entries (kept .gitignore, setup.sh, $(ls third_party/*.patch 2>/dev/null | wc -l) patches)"
  echo "clean: run ./setup.sh to rebuild"
  exit 0
fi

if [ "$1" = "links" ]; then
  [ -n "$VENV_ROOT" ] || { echo "links requires I4H_VENV_ROOT" >&2; exit 2; }
  for project in "${ordered[@]}"; do
    link_external_venv "$project" 1
  done
  echo "linked: ${ordered[*]}"
  exit 0
fi

[ "$1" = "all" ] || { echo "setup.sh: unknown operation '$1'" >&2; usage >&2; exit 2; }

# Always re-run for targets that need it. Every step in that script self-guards,
# and a half-finished run (checkouts made, patches not applied) is invisible to
# any directory test — which is exactly how the openpi lerobot/pyav rewrites got
# skipped once the checkouts existed.
for t in "${ordered[@]}"; do
  need_third_party "$t" || continue
  echo "==> $t needs third_party checkouts; running third_party/setup.sh"
  if [ ! -x third_party/setup.sh ]; then
    echo "!! third_party/setup.sh is missing." >&2
    echo "   Isaac Sim, IsaacLab, IsaacLab-Arena and Isaac-GR00T checkouts are not vendored." >&2
    echo "   Restore third_party/setup.sh from this repository before syncing." >&2
    exit 1
  fi
  third_party/setup.sh "${I4H_THIRD_PARTY_TARGET:-all}"
  break
done

failed=()
uv_sync_args=()
if [ -n "${I4H_UV_SYNC_ARGS:-}" ]; then
  read -r -a uv_sync_args <<<"$I4H_UV_SYNC_ARGS"
fi
for project in "${ordered[@]}"; do
  # Only the projects that carry tests define a dev extra; asking for it
  # elsewhere is a hard error in uv, not a no-op.
  extra=()
  grep -q '^dev = ' "$project/pyproject.toml" 2>/dev/null && extra=(--extra dev)
  echo "==> uv sync --project $project ${extra[*]}"
  # -u VIRTUAL_ENV so an activated venv does not capture the sync.
  sync_env=(env -u VIRTUAL_ENV)
  if [ -n "$VENV_ROOT" ]; then
    link_external_venv "$project"
    sync_env+=("UV_PROJECT_ENVIRONMENT=$(venv_path "$project")")
  fi
  if "${sync_env[@]}" uv sync --project "$project" "${extra[@]}" "${uv_sync_args[@]}" 2>&1 | sed 's/^/    /'; then
    :
  else
    echo "    !! failed" >&2
    failed+=("$project")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
echo
echo "synced: ${ordered[*]}"
