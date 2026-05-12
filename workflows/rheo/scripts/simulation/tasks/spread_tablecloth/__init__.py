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

import gymnasium as gym

from . import (
    g1_spread_tablecloth_env_cfg,
    g1_spread_tablecloth_teleop_env_cfg,
    h2_spread_tablecloth_env_cfg,
    h2_spread_tablecloth_teleop_env_cfg,
)

gym.register(
    id="Isaac-Spread-Tablecloth-G129-Inspire-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g1_spread_tablecloth_env_cfg.G1SpreadTableclothEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Spread-Tablecloth-H2-Sharpa-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": h2_spread_tablecloth_env_cfg.H2SpreadTableclothEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Spread-Tablecloth-G129-Inspire-Teleop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g1_spread_tablecloth_teleop_env_cfg.G1SpreadTableclothTeleopEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Spread-Tablecloth-H2-Sharpa-Teleop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": h2_spread_tablecloth_teleop_env_cfg.H2SpreadTableclothTeleopEnvCfg,
    },
    disable_env_checker=True,
)

