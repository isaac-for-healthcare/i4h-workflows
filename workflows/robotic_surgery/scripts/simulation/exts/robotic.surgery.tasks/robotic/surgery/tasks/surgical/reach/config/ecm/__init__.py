# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Reach-ECM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.ECMReachEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ECMReachPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Reach-ECM-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.ECMReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ECMReachPPORunnerCfg,
    },
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Reach-ECM-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.ECMReachEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ECMReachPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-ECM-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.ECMReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ECMReachPPORunnerCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Reach-ECM-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.ECMReachEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ECMReachPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-ECM-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.ECMReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ECMReachPPORunnerCfg,
    },
    disable_env_checker=True,
)
