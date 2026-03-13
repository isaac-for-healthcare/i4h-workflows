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

from . import g1_assemble_trocar_env_cfg, g1_assemble_trocar_teleop_env_cfg

gym.register(
    id="Isaac-Assemble-Trocar-G129-Dex3-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g1_assemble_trocar_env_cfg.G1AssembleTrocarEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Assemble-Trocar-G129-Dex3-Joint-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g1_assemble_trocar_env_cfg.G1AssembleTrocarEvalEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Assemble-Trocar-G129-Dex3-Teleop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g1_assemble_trocar_teleop_env_cfg.G1AssembleTrocarTeleopEnvCfg,
    },
    disable_env_checker=True,
)
