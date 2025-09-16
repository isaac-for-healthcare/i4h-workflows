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

"""
Python module serving as a SO-ARM Starter extension for Isaac Lab.

This extension provides environments and tasks for SO-ARM Starter simulation,
including tool handling, tray organization, and assistance workflows.
"""

import gymnasium as gym

# Register the environment with full module path
gym.register(
    id="Isaac-SOARM101-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "so_arm_starter_ext.tasks.so_arm_starter.approach.config." "soarm101.so_arm_env_cfg:SOARMStarterEnvCfg"
        ),
    },
)

print("✅ Isaac-SOARM101-v0 environment registered successfully!")
