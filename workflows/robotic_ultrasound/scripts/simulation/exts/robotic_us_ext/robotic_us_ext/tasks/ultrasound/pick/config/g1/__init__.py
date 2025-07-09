# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import pick_g1_env_cfg


gym.register(
    id="Isaac-PickReach-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pick_g1_env_cfg.PickReachG1EnvCfg,
    },
    disable_env_checker=True,
)
