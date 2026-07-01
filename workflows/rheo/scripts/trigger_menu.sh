#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

TRIGGER_HOST="${TRIGGER_HOST:-localhost}"
TRIGGER_PORT="${TRIGGER_PORT:-8081}"
BASE_URL="http://${TRIGGER_HOST}:${TRIGGER_PORT}"

post() {
  local path="$1"
  echo
  echo ">>> POST ${BASE_URL}${path}"
  curl -sS -X POST "${BASE_URL}${path}"
  echo
}

get_status() {
  echo
  echo ">>> GET ${BASE_URL}/status"
  curl -sS "${BASE_URL}/status"
  echo
}

show_menu() {
  cat <<EOF

===========================================
  SiDi VLA Team - Rheo Robot Trigger Menu
  Server: ${BASE_URL}
===========================================
  1) tray_pick_and_place
  2) cart_push
  3) reset environment
  4) show status
  5) exit
===========================================
EOF
}

main() {
  while true; do
    show_menu
    read -r -p "Escolha uma opcao [1-5]: " choice
    case "${choice}" in
      1)
        post "/trigger?sequence=tray_pick_and_place"
        ;;
      2)
        post "/trigger?sequence=cart_push"
        ;;
      3)
        post "/reset"
        ;;
      4)
        get_status
        ;;
      5)
        echo "Saindo."
        exit 0
        ;;
      *)
        echo "Opcao invalida."
        ;;
    esac
  done
}

main "$@"
