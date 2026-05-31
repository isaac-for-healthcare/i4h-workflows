#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPONENTS=(arena policy mimic dataset cosmos)

case "${1:-all}" in
  -h|--help)
    echo "usage: $(basename "$0") [arena|policy|mimic|dataset|cosmos|all] [stop args...]"
    exit 0
    ;;
  all|"")
    shift || true
    for component in "${COMPONENTS[@]}"; do
      "${SCRIPT_DIR}/${component}/stop.sh" "$@"
    done
    ;;
  arena|policy|mimic|dataset|cosmos)
    COMPONENT="$1"
    shift
    "${SCRIPT_DIR}/${COMPONENT}/stop.sh" "$@"
    ;;
  *)
    for component in "${COMPONENTS[@]}"; do
      "${SCRIPT_DIR}/${component}/stop.sh" "$@"
    done
    ;;
esac
