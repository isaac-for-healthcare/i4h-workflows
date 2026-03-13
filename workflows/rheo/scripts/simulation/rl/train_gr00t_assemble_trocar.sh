#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# GR00T RL Post-training / Evaluation on Assemble-Trocar task
# Usage: bash train_gr00t_assemble_trocar.sh [MODE] [OPTIONS] [HYDRA_OVERRIDES...]
#
# Modes:
#   train (default)  - Run RL post-training
#   eval             - Run evaluation on trained checkpoint
#
# Examples:
#   # Training (requires --model_path)
#   bash train_gr00t_assemble_trocar.sh train --model_path /models/your_gr00t_ckpt
#
#   # Training with custom env scale
#   bash train_gr00t_assemble_trocar.sh train --model_path /models/your_gr00t_ckpt \
#       env.train.total_num_envs=32 env.eval.total_num_envs=4
#
#   # Evaluation of base model (no RL checkpoint)
#   bash train_gr00t_assemble_trocar.sh eval --model_path /models/your_gr00t_ckpt
#
#   # Evaluation of RL-trained checkpoint
#   bash train_gr00t_assemble_trocar.sh eval \
#       --model_path /path/to/rl_ckpt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="/workspaces"
RLINF_PATH="${WORKSPACE_ROOT}/third_party/RLinf"
CONFIG_PATH="${WORKSPACE_ROOT}/workflows/rheo/scripts/simulation/rl/rlinf_ext/config"
CONFIG_NAME="isaaclab_ppo_gr00t_assemble_trocar"

# Default mode
MODE="train"

# Parse mode (first argument if it's train/eval)
if [[ "$1" == "train" || "$1" == "eval" ]]; then
    MODE="$1"
    shift
fi

# Parse arguments
MODEL_PATH=""
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [MODE] [OPTIONS] [HYDRA_OVERRIDES...]"
            echo ""
            echo "Modes:"
            echo "  train (default)    Run RL post-training"
            echo "  eval               Run evaluation on trained checkpoint"
            echo ""
            echo "Options:"
            echo "  --model_path PATH  Path to GR00T base model checkpoint or RL training checkpoint (required)"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Hydra Overrides (examples):"
            echo "  env.train.total_num_envs=32"
            echo "  env.eval.total_num_envs=4"
            echo "  runner.resume_dir=/path/to/checkpoint"
            echo "  runner.max_epochs=500"
            echo ""
            echo "Examples:"
            echo "  # Training"
            echo "  $0 train --model_path /models/gr00t_ckpt"
            echo ""
            echo "  # Evaluate base model"
            echo "  $0 eval --model_path /models/gr00t_ckpt"
            echo ""
            echo "  # Evaluate RL-trained checkpoint"
            echo "  $0 eval --model_path /path/to/rl_ckpt"
            exit 0
            ;;
        *)
            HYDRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

# Validate model path
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required"
    echo "Usage: $0 [train|eval] --model_path /path/to/gr00t_ckpt [OPTIONS...]"
    exit 1
fi

# Add model path to overrides
HYDRA_OVERRIDES+=("actor.model.model_path=${MODEL_PATH}")
HYDRA_OVERRIDES+=("rollout.model.model_path=${MODEL_PATH}")

# Set environment
export EMBODIED_PATH="${RLINF_PATH}/examples/embodiment"
export PYTHONPATH="${RLINF_PATH}:${WORKSPACE_ROOT}/workflows/rheo/scripts:${WORKSPACE_ROOT}/workflows/rheo/scripts/simulation/rl:${PYTHONPATH}"
export RLINF_EXT_MODULE=rlinf_ext
# Isaac Sim environment (for omni imports)
export ISAAC_PATH="${ISAAC_PATH:-/isaac-sim}"
export CARB_APP_PATH="${CARB_APP_PATH:-${ISAAC_PATH}/kit}"
export EXP_PATH="${EXP_PATH:-${ISAAC_PATH}/apps}"
export PYTHONPATH="${ISAAC_PATH}/exts:${ISAAC_PATH}/extscore:${ISAAC_PATH}/extscache:${PYTHONPATH}"
export PYTHONPATH="${ISAAC_PATH}/kit/kernel/py:${ISAAC_PATH}/kit/exts:${ISAAC_PATH}/kit/extscore:${PYTHONPATH}"
export LD_LIBRARY_PATH="${ISAAC_PATH}/kit:${ISAAC_PATH}/kit/kernel/plugins:${LD_LIBRARY_PATH}"

# Mode-specific setup
if [[ "$MODE" == "eval" ]]; then
    SCRIPT_FILE="${RLINF_PATH}/examples/embodiment/eval_embodied_agent.py"
    MODE_DISPLAY="Evaluation"
else
    SCRIPT_FILE="${RLINF_PATH}/examples/embodiment/train_embodied_agent.py"
    MODE_DISPLAY="Training"
fi

# Setup logging directory with timestamp (same location as experiment results)
TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
LOG_DIR="${SCRIPT_DIR}/results/gr00t_assemble_trocar/${MODE}_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/${MODE}.log"
mkdir -p "${LOG_DIR}"

# Add log_path to hydra overrides
HYDRA_OVERRIDES+=("runner.logger.log_path=${LOG_DIR}")

echo "========================================"
echo "GR00T RL ${MODE_DISPLAY}: Assemble-Trocar"
echo "========================================"
echo "Mode: ${MODE}"
echo "Model Path: ${MODEL_PATH}"
echo "Config: ${CONFIG_NAME}"
echo "Log Dir: ${LOG_DIR}"
echo "Overrides: ${HYDRA_OVERRIDES[*]}"
echo "========================================"

/isaac-sim/python.sh "${SCRIPT_FILE}" \
    --config-path "${CONFIG_PATH}" \
    --config-name "${CONFIG_NAME}" \
    "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================"
echo "Log saved to: ${LOG_FILE}"
echo "========================================"
