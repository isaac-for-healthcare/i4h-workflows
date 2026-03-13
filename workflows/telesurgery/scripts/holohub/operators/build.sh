#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e; cd "$(dirname "$0")"

# --- Config ---
LOG="build.log"; > "$LOG"
PYTHON_EXEC="${PYTHON_EXEC:-$(which python3)}"
EXT=$($PYTHON_EXEC -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
PY_TAG=$($PYTHON_EXEC -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')
OPTS="-G Ninja -DCMAKE_PREFIX_PATH=/opt/nvidia/holoscan -DCMAKE_MODULE_PATH=/opt/nvidia/holohub/cmake -DCMAKE_BUILD_TYPE=Release -Wno-dev -DPYTHON_EXECUTABLE=$PYTHON_EXEC"

echo "Building operators [$PY_TAG] (log: $LOG)"

build() {
    local NAME="$1"
    local DIR="$2"
    local CHECK="$3"
    local INSTALL_FUNC="$4"

    if [ -f "$CHECK" ] || [ ! -f "$DIR/CMakeLists.txt" ]; then
        echo "  $NAME: ✓"
        return
    fi

    echo "  $NAME: building..."
    local BUILD_DIR="$DIR/.build_$PY_TAG"
    mkdir -p "$DIR/lib/$PY_TAG" "$BUILD_DIR"

    {
        echo "=== $NAME ==="
        cmake -S "$DIR" -B "$BUILD_DIR" $OPTS
        ninja -C "$BUILD_DIR"
        $INSTALL_FUNC "$BUILD_DIR"
    } >> "$LOG" 2>&1 || { echo "  $NAME: FAILED"; exit 1; }

    echo "  $NAME: ✓"
}

install_streamlift() {
    local BD="$1"
    find "$BD" -name "libstreamlift*.so*" -exec cp {} "$DIR/lib/$PY_TAG/" \;
    for d in streamlift_downsampler streamlift_upsampler; do
        [ -d "/$d" ] && cp -r "/$d" "$DIR/"
    done
}

install_aja() {
    local BD="$1"
    find "$BD" -name "lib*aja_source*.so" -exec cp {} "$DIR/lib/$PY_TAG/" \;
    [ -d /aja_source ] && cp /aja_source/* "$DIR/"
}

install_nv_codec() {
    local BD="$1"
    find "$BD" -name "lib*nvidia_video_codec*.so" -exec cp {} "$DIR/lib/$PY_TAG/" \;
    find "$BD" -name "libnv_video_*.so" -exec cp {} "$DIR/lib/$PY_TAG/" \;
    for d in nv_video_encoder nv_video_decoder; do
        [ -d "/$d" ] && cp -r "/$d" "$DIR/"
    done
}

# --- Streamlift ---
DIR="streamlift"
mkdir -p "$DIR/streamlift_downsampler" "$DIR/streamlift_upsampler"
build "streamlift" "$DIR" "$DIR/streamlift_downsampler/_streamlift_downsampler$EXT" install_streamlift

# --- AJA Source ---
DIR="camera/aja_source"
build "aja_source" "$DIR" "$DIR/_aja_source$EXT" install_aja

# --- Nvidia Video Codec ---
DIR="nvidia_video_codec"
mkdir -p "$DIR/nv_video_encoder" "$DIR/nv_video_decoder"
build "nvidia_video_codec" "$DIR" "$DIR/nv_video_encoder/_nv_video_encoder$EXT" install_nv_codec

echo "✓ All ready [$PY_TAG]"
