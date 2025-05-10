# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# if sigma > 2.0000e-03: # skip 1 setp
# if sigma > 2.1618e-02: # skip 5 steps
# if sigma > 1.8636e-01: # skip 10 steps
# if sigma > 9.6542e-01: # skip 15 steps
# if sigma > 1.2866e+00: # skip 16 steps x
# if sigma > 1.6954e+00: # skip 17 steps x
# if sigma > 2.2107e+00: # skip 18 steps x
# if sigma > 3.6538e+00: # skip 20 steps x

# Function to run a command on a specific GPU
# For multi-GPU setup using torchrun
sigma_max=80
sigma_threshold=1.2866e+00
foreground_label="3,4"
num_input_frames=1
seed=1
controlnet_specs=./environments/cosmos_transfer1/config/inference_cosmos_transfer1_two_views.json
source_data_dir=/healthcareeng_monai/I4H/2025-04-30-21-25-Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0-NEWBATCH
output_data_dir=/healthcareeng_monai/I4H/2025-04-30-21-25-Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0-NEWBATCH-debug

save_name_offset=0
export CHECKPOINT_DIR="/workspace/code/cosmos-transfer1/checkpoints"
export PROJECT_ROOT="/workspace/code/i4h-workflows"
# Set PYTHONPATH with absolute paths
export PYTHONPATH="$PROJECT_ROOT/third_party/cosmos-transfer1:$PROJECT_ROOT/workflows/robotic_ultrasound/scripts:$PROJECT_ROOT/workflows/robotic_ultrasound/scripts/simulation"
export DEBUG_GENERATION="0"

# Current working directory for reference
CWD="$(pwd)"
echo "Current directory: $CWD"
echo "PYTHONPATH: $PYTHONPATH"

# Use absolute path to transfer.py
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$PYTHONPATH torchrun \
    --nnodes=1 --node_rank=0 \
    --nproc_per_node=1 \
    environments/cosmos_transfer1/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --source_data_dir $source_data_dir \
    --output_data_dir $output_data_dir \
    --controlnet_specs $controlnet_specs \
    --save_name_offset $save_name_offset \
    --offload_text_encoder_model \
    --height 224 \
    --width 224 \
    --fps 30 \
    --foreground_label $foreground_label \
    --sigma_threshold $sigma_threshold \
    --sigma_max $sigma_max \
    --num_gpus 1 \
    --num_input_frames $num_input_frames \
    --seed $seed
