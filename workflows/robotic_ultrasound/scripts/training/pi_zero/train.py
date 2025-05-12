# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os

from openpi import train
from openpi.training.config import DataConfigFactory

from policy_runner.pi0_policy.config import get_config
from policy_runner.pi0_policy.utils import compute_normalization_stats


def ensure_norm_stats_exist(config):
    """Ensure normalization statistics exist, computing them if necessary."""
    data_config = config.data
    if isinstance(data_config, DataConfigFactory):
        data_config = data_config.create(config.assets_dirs, config.model)

    output_path = config.assets_dirs / data_config.repo_id
    stats_file = output_path / "norm_stats.json"

    if not os.path.exists(stats_file):
        print(f"Normalization statistics not found at {stats_file}. Computing...")
        compute_normalization_stats(config)
    else:
        print(f"Normalization statistics found at {stats_file}. Skipping computation.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a PI-Zero model")
    parser.add_argument(
        "--config", type=str, default="robotic_ultrasound", help="Configuration name to use for training"
    )
    parser.add_argument(
        "--exp_name", type=str, required=True, help="Name of the experiment for logging and checkpointing"
    )
    parser.add_argument("--repo_id", type=str, default="i4h/robotic_ultrasound", help="Repository ID for the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Get configuration using the provided config name
    config = get_config(name=args.config, repo_id=args.repo_id, exp_name=args.exp_name)
    # Ensure we have normalization stats
    ensure_norm_stats_exist(config)
    # Begin training
    train.main(config)
