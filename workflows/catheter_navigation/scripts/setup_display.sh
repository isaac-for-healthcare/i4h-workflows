#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Set up an X11 display + xauth cookie so the interactive catheter viewport
# (OpenCV/Qt window) can open, including from inside a Docker container
# (`xhost +local:docker`).
#
# IMPORTANT: `export DISPLAY` only persists in your shell if you *source* this
# script. If you run it normally (./scripts/setup_display.sh) the DISPLAY value
# is lost when the script exits, and the viewport falls back to :0 and fails
# with "could not connect to display :0".
#
# Recommended usage (note the leading `source`):
#   source ./scripts/setup_display.sh            # uses DISPLAY=:1 by default
#   DISPLAY_NUM=:0 source ./scripts/setup_display.sh
#
# Then launch the viewport in the same shell:
#   export PYTHONPATH="$PWD/scripts/simulation:$PYTHONPATH"
#   python -m fluorosim.examples.interactive_catheter_slang_viewport \
#     --ct-dir /tmp/fluoro_cache

# Detect whether we were sourced; only enable `set -e` when executed directly so
# a failed command can't kill the user's interactive shell.
_sourced=0
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    _sourced=1
fi
if [ "$_sourced" -eq 0 ]; then
    set -euo pipefail
fi

# Display to use (override with DISPLAY_NUM=:0 etc).
DISPLAY_NUM="${DISPLAY_NUM:-:1}"
export DISPLAY="$DISPLAY_NUM"

echo "[setup_display] Using DISPLAY=$DISPLAY"

# Ensure the xauth / xhost tools are present.
if ! command -v xauth >/dev/null 2>&1 || ! command -v xhost >/dev/null 2>&1; then
    echo "[setup_display] Installing x11-xserver-utils (needs sudo)..."
    sudo apt-get update
    sudo apt-get install -y x11-xserver-utils
fi

# Warn if there is no X server socket for this display.
if [ ! -e "/tmp/.X11-unix/X${DISPLAY#:}" ]; then
    echo "[setup_display] WARNING: no X server socket found at /tmp/.X11-unix/X${DISPLAY#:}." >&2
    echo "[setup_display]          Available: $(ls /tmp/.X11-unix 2>/dev/null || echo none)" >&2
fi

# Refresh the MIT-MAGIC-COOKIE for this display. The removes may fail if no
# cookie exists yet, so don't let that abort the script.
xauth remove "$DISPLAY" 2>/dev/null || true
xauth remove "$(hostname)/unix${DISPLAY}" 2>/dev/null || true
xauth add "$(hostname)/unix${DISPLAY}" MIT-MAGIC-COOKIE-1 "$(mcookie)"

# Allow local (non-network) connections, including containers, to use the display.
xhost +local:docker || true

echo "[setup_display] XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-<unset>}"
ls -l "${XAUTHORITY:-$HOME/.Xauthority}" || true

echo "[setup_display] Done. DISPLAY=$DISPLAY is ready."
if [ "$_sourced" -eq 0 ]; then
    echo "[setup_display] NOTE: you ran this in a subshell, so DISPLAY did NOT persist."
    echo "[setup_display]       In your shell, run:  export DISPLAY=$DISPLAY"
    echo "[setup_display]       (or re-run as:       source ${BASH_SOURCE[0]} )"
fi
