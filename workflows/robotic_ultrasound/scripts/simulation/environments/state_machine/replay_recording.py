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

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates a single-arm manipulator.")
parser.add_argument(
    "hdf5_path", type=str, help="Path to a .hdf5 file with recorded data or a directory of .hdf5 files."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default="Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0", help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# isort: off
import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Import extensions to set up environment tasks
from robotic_us_ext import tasks  # noqa: F401
from simulation.environments.state_machine.utils import reset_scene_to_initial_state, validate_hdf5_path

# isort: on


def main():
    """Main function."""
    # Check if the provided path contains valid HDF5 files
    total_episodes = validate_hdf5_path(args_cli.hdf5_path)
    print(f"total_episodes: {total_episodes}")
    if total_episodes == 0:
        raise AssertionError(
            "Unsupported file format. Please use a directory containing .hdf5 files or a single .hdf5 file."
        )

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Use inference mode for all simulation state updates
    with torch.inference_mode():
        # reset environment at start
        obs = env.reset()

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        episode_idx = 0
        action_idx = 0
        torso_obs_key = "observations/torso_obs"
        joint_state_key = "abs_joint_pos"
        joint_vel_key = "observations/joint_vel"
        if "Rel" in args_cli.task:
            action_key = "action"
        else:
            action_key = "abs_action"

        # simulate physics
        while simulation_app.is_running():
            actions = reset_scene_to_initial_state(
                env,
                args_cli.hdf5_path,
                episode_idx,
                action_key,
                torso_obs_key,
                joint_state_key,
                joint_vel_key,
            )

            for episode_idx in range(total_episodes):
                print(f"\nepisode_idx: {episode_idx}")

                # get from the second action since the first joint state is not recorded
                for action_step, action in enumerate(actions[1:]):
                    # Get next action from recorded data
                    current_action = torch.tensor(action, device=args_cli.device)
                    current_action = current_action.unsqueeze(0)

                    print(f"Step {action_step}/{len(actions)} | Episode {episode_idx + 1}/{total_episodes}")
                    print(f"Action: {current_action}")

                    # Step environment with recorded action
                    obs, rew, terminated, truncated, info_ = env.step(current_action)

                    # Update counters
                    action_idx += 1

                env.reset()
                # Check if next episode exists before trying to load it
                if episode_idx + 1 >= total_episodes:
                    print(f"Completed all episodes ({total_episodes})")
                    break

                actions = reset_scene_to_initial_state(
                    env,
                    args_cli.hdf5_path,
                    episode_idx + 1,
                    action_key,
                    torso_obs_key,
                    joint_state_key,
                    joint_vel_key,
                )

            # Check if we've reached the end of available episodes
            if episode_idx >= total_episodes - 1 or actions is None:
                print(f"Reached the end of available episodes ({episode_idx + 1}/{total_episodes})")
                break

    # close the environment and data file
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
