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

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from ..teleop import ik_rel_env_cfg
from . import agents, franka_manager_rl_env_cfg
from ....pick.config.g1 import pick_g1_env_cfg

##
# Inverse Kinematics - Absolute Pose Control
##
gym.register(
    id="Isaac-Reach-Torso-FrankaUsRs-IK-RL-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_manager_rl_env_cfg.FrankaModRGBDIkRlEnvCfg},
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##
gym.register(
    id="Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.ModFrankaUltrasoundTeleopEnv,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
