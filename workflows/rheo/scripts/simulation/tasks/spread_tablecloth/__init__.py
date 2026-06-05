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

from .g1_spread_tablecloth_env_cfg import G1SpreadTableclothEnvCfg
from .g1_spread_tablecloth_teleop_env_cfg import G1SpreadTableclothTeleopEnvCfg
from .h2_spread_tablecloth_env_cfg import H2SpreadTableclothEnvCfg
from .h2_spread_tablecloth_teleop_env_cfg import H2SpreadTableclothTeleopEnvCfg

_ENTRY = "isaaclab.envs:ManagerBasedRLEnv"
for task_id, cfg_cls in [
    ("Isaac-Spread-Tablecloth-G129-Inspire-Joint", G1SpreadTableclothEnvCfg),
    ("Isaac-Spread-Tablecloth-H2-Sharpa-Joint", H2SpreadTableclothEnvCfg),
    ("Isaac-Spread-Tablecloth-G129-Inspire-Teleop", G1SpreadTableclothTeleopEnvCfg),
    ("Isaac-Spread-Tablecloth-H2-Sharpa-Teleop", H2SpreadTableclothTeleopEnvCfg),
]:
    gym.register(id=task_id, entry_point=_ENTRY, kwargs={"env_cfg_entry_point": cfg_cls}, disable_env_checker=True)
