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
# python sim_with_dds_eval.py --data_path=../data/hdf5/heldout-test50/ --use_rel --enable_camera
# python playback_recorded.py --data_path=../../data/hdf5/heldout-test50/ --use_rel --enable_camera
# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates a single-arm manipulator.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default="Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0", help="Name of the task.")
parser.add_argument("--use_rel", action="store_true", default=False, help="Use relative actions.")
parser.add_argument("--data_path", type=str, required=True, help="Path to either .hdf5 file containing recorded data.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os

import gymnasium as gym
import h5py
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
# Import extensions to set up environment tasks
from robotic_us_ext import tasks  # noqa: F401


def get_hdf5_episode_data(root, action_key: str):
    """Get episode data from HDF5 format."""
    base_path = "data/demo_0"
    try:
        actions = root[f"{base_path}/{action_key}"][()]
    except KeyError as e:
        print(f"Error loading data using key: {base_path}/{action_key}")
        print(f"Available keys: {list(root[base_path].keys())}")
        raise e
    return actions


def get_observation_episode_data(data_path: str, episode_idx: int, key: str):
    """Get episode data from HDF5 format for a given key."""
    with h5py.File(os.path.join(data_path, f"data_{episode_idx}.hdf5"), "r") as f:
        data = get_hdf5_episode_data(f, key)
        return data


def get_episode_data(data_path: str, episode_idx: int):
    """Get episode data from HDF5 format."""
    # Load initial episode data
    try:
        action_key = "action" if args_cli.use_rel else "abs_action"
        root = h5py.File(os.path.join(data_path, f"data_{episode_idx}.hdf5"), "r")
        # total_episodes are the number of files starting with data__ in the directory
        total_episodes = len([k for k in os.listdir(data_path) if k.startswith("data_")])
        actions = get_hdf5_episode_data(root, action_key)
        return actions, total_episodes
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None


def reset_organ_to_position(env, object_position):
    """Reset the organ position."""
    organs = env.unwrapped.scene._rigid_objects["organs"]
    root_state = organs.data.default_root_state.clone()
    root_state[:, :7] = torch.tensor(object_position[0, :7], device="cuda:0")
    # write to the sim
    organs.write_root_state_to_sim(root_state)


def reset_robot_to_position(env, robot_initial_joint_state, joint_vel=None):
    """Reset the robot to the initial joint state."""
    robot = env.unwrapped.scene["robot"]
    joint_pos = torch.tensor(robot_initial_joint_state[0], device="cuda:0").unsqueeze(0)
    if joint_vel is not None:
        joint_vel = torch.tensor(joint_vel[0], device="cuda:0")
    else:
        print("No joint velocity provided, setting to zero")
        joint_vel = torch.zeros_like(joint_pos)
    # set joint positions
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def main():
    """Main function."""
    # Check for HDF5 format by inspecting files in the directory
    if os.path.isdir(args_cli.data_path):
        files = os.listdir(args_cli.data_path)
        if not any(file.endswith(".hdf5") for file in files):
            raise AssertionError("Unsupported file format. Please use a directory containing .hdf5 files.")
    else:
        raise AssertionError("Unsupported file format. Please use a directory containing .hdf5 files.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment at start
    obs = env.reset()

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    episode_idx = 0
    action_idx = 0

    # simulate physics
    while simulation_app.is_running():
        torso_obs_key = "observations/torso_obs"
        actions, total_episodes = get_episode_data(args_cli.data_path, episode_idx)

        # Check if we've reached the end of available episodes
        if episode_idx >= total_episodes - 1 or actions is None:
            print(f"Reached the end of available episodes ({episode_idx + 1}/{total_episodes})")
            break

        object_position = get_observation_episode_data(args_cli.data_path, episode_idx, torso_obs_key)
        reset_organ_to_position(env, object_position)

        # reset robot to initial joint state
        joint_state_key = "abs_joint_pos"
        robot_initial_joint_state = get_observation_episode_data(args_cli.data_path, episode_idx, joint_state_key)
        try:
            joint_vel_key = "observations/joint_vel"
            joint_vel = get_observation_episode_data(args_cli.data_path, episode_idx, joint_vel_key)
        except KeyError:
            print("No joint velocity provided, setting to zero")
            joint_vel = None
        reset_robot_to_position(env, robot_initial_joint_state, joint_vel=joint_vel)

        with torch.inference_mode():
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

                actions, _ = get_episode_data(args_cli.data_path, episode_idx + 1)
                object_position = get_observation_episode_data(args_cli.data_path, episode_idx + 1, torso_obs_key)
                reset_organ_to_position(env, object_position)
                robot_initial_joint_state = get_observation_episode_data(
                    args_cli.data_path, episode_idx + 1, joint_state_key
                )
                try:
                    joint_vel = get_observation_episode_data(args_cli.data_path, episode_idx + 1, joint_vel_key)
                except KeyError:
                    print("No joint velocity provided, setting to zero")
                    joint_vel = None
                reset_robot_to_position(env, robot_initial_joint_state, joint_vel=joint_vel)

    # close the environment and data file
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()