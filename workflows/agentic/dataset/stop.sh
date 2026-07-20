#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/utils.sh
source "${SCRIPT_DIR}/../scripts/utils.sh"

status=0
agentic_stop_by_pattern "dataset conversions" "dataset" "i4h-agentic-hdf5-to-lerobot([[:space:]]|$)" "$@" || status=$?

agentic_stop_visualizers() {
  agentic_parse_stop_options "dataset visualizers" "$@"
  local label="dataset visualizers"
  local pattern="lerobot.scripts.visualize_dataset_html([[:space:]]|$)"
  if [[ -n "${AGENTIC_STOP_ENV}" ]]; then
    label="${label} env=${AGENTIC_STOP_ENV}"
    pattern="${pattern}.*--repo-id(=|[[:space:]]+)[^[:space:]]*/${AGENTIC_STOP_ENV}([[:space:]]|$)"
  fi
  agentic_stop_matching "${label}" "${AGENTIC_STOP_FORCE}" "${AGENTIC_STOP_TIMEOUT}" "${pattern}"
}

agentic_stop_visualizers "$@" || status=$?
exit "${status}"
