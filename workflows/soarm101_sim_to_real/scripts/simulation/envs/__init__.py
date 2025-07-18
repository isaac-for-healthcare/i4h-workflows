"""
SO-ARM 101 Gymnasium Environment Registration
"""

import gymnasium as gym
from gymnasium.envs.registration import register

# Register the SO-ARM 101 environments
register(
    id="SoArm101-v0",
    entry_point="simulation.envs.soarm101_env:SoArm101Env",
    max_episode_steps=1000,
    kwargs={
        "num_envs": 1,
        "device": "cuda",
        "render_mode": "human",
        "sim_steps_per_action": 30, # More steps for smoother movement
        "action_scale": 1,         # Larger movements by default
    }
)

register(
    id="SoArm101Reach-v0", 
    entry_point="simulation.envs.soarm101_env:SoArm101ReachEnv",
    max_episode_steps=500,
    kwargs={
        "num_envs": 1,
        "device": "cuda",
        "render_mode": "human",
        "sim_steps_per_action": 15,  # More steps for smoother movement
        "action_scale": 1,         # Larger movements by default
    }
)

# Register vectorized version
register(
    id="SoArm101Vec-v0",
    entry_point="simulation.envs.soarm101_env:SoArm101VecEnv", 
    max_episode_steps=1000,
    kwargs={
        "num_envs": 16,
        "device": "cuda",
        "render_mode": None,
        "sim_steps_per_action": 10,  # Fewer steps for performance with 16 envs
        "action_scale": 1.5,         # Larger movements by default
    }
)

# Register teleoperation environment
register(
    id="SoArm101Teleop-v0",
    entry_point="simulation.envs.soarm101_teleop_env:SoArm101TeleopEnv",
    max_episode_steps=10000,  # Long episodes for continuous teleoperation
    kwargs={
        "host": "localhost",
        "port": 8888,
        "auto_connect": True,
        "device": "cuda",
        "render_mode": "human",
        "sim_steps_per_action": 25,  # Responsive for real robot (was 25)
        "action_scale": 1.0,         # Direct mapping for real robot
    }
) 