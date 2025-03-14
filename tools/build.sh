#!/bin/bash
set -eu

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

holoscan_dir=$SCRIPT_DIR/../workflows/robotic_ultrasound/scripts/holoscan_apps/

pushd $holoscan_dir
cmake -B build -S . && cmake --build build
popd
