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

import argparse
import collections

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher
from simulation.environments.state_machine.utils import (
    RobotPositions,
    RobotQuaternions,
    capture_camera_images,
    compute_relative_action,
    get_joint_states,
    get_robot_obs,
)
from simulation.utils.common import resolve_checkpoint_path

# add argparse arguments
parser = argparse.ArgumentParser(description="This script evaluate the pi0 model in a single-arm manipulator.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="nvidia/Liver_Scan_Pi0_Cosmos_Rel",
    help="checkpoint path. Default to use policy checkpoint in the latest assets.",
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="i4h/sim_liver_scan",
    help="the LeRobot repo id for the dataset norm.",
)

# append AppLauncher cli argr
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.ckpt_path = resolve_checkpoint_path(args_cli.ckpt_path)

# launch omniverse app
app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app
reset_flag = False

# isort: off
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# This is to avoid boto3 conflict with app launcher
from policy_runner.pi0.runners import PI0PolicyRunner

# Import extensions to set up environment tasks
from robotic_us_ext import tasks  # noqa: F401
# isort: on


def get_reset_action(env, use_rel: bool = True):
    """Get the reset action."""
    reset_pos = torch.tensor(RobotPositions.SETUP, device=args_cli.device)
    reset_quat = torch.tensor(RobotQuaternions.DOWN, device=args_cli.device)
    reset_tensor = torch.cat([reset_pos, reset_quat], dim=-1)
    reset_tensor = reset_tensor.repeat(env.unwrapped.num_envs, 1)
    if not use_rel:
        return reset_tensor
    else:
        robot_obs = get_robot_obs(env)
        return compute_relative_action(reset_tensor, robot_obs)


def main():
    """Main function."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # modify configuration
    env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    obs = env.reset()

    # Recommended to use 40 steps to allow enough steps to reset the SETUP position of the robot
    reset_steps = 40
    max_timesteps = 250

    # allow environment to settle
    for _ in range(reset_steps):
        reset_tensor = get_reset_action(env)
        obs, rew, terminated, truncated, info_ = env.step(reset_tensor)

    policy_runner = PI0PolicyRunner(
        ckpt_path=args_cli.ckpt_path,
        repo_id=args_cli.repo_id,
        task_description="Perform a liver ultrasound.",
    )
    # Number of steps played before replanning
    replan_steps = 5

    # simulate environment
    while simulation_app.is_running():
        global reset_flag
        with torch.inference_mode():
            action_plan = collections.deque()

            for t in range(max_timesteps):
                if not action_plan:
                    # get images
                    rgb_images, _, _ = capture_camera_images(
                        env, ["room_camera", "wrist_camera"], device=env.unwrapped.device
                    )
                    room_img, wrist_img = rgb_images[0, 0, ...].cpu().numpy(), rgb_images[0, 1, ...].cpu().numpy()
                    action_chunk = policy_runner.infer(
                        room_img=room_img, wrist_img=wrist_img, current_state=get_joint_states(env)[0]
                    )
                    action_plan.extend(action_chunk[:replan_steps])

                action = action_plan.popleft()

                action = action.astype(np.float32)

                # convert to torch
                action = torch.tensor(action, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)

                # step the environment
                obs, rew, terminated, truncated, info_ = env.step(action)

            env.reset()
            print("Resetting the environment.")
            for _ in range(reset_steps):
                reset_tensor = get_reset_action(env)
                obs, rew, terminated, truncated, info_ = env.step(reset_tensor)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
