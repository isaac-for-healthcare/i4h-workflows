# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates a single-arm manipulator.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--num_episodes", type=int, default=0, help="Number of episodes to collect. If 0, no data collection is performed."
)
parser.add_argument(
    "--camera_names",
    type=str,
    nargs="+",
    default=["room_camera", "wrist_camera"],
    help="List of camera names to capture from.",
)
parser.add_argument("--reset_steps", type=int, default=15, help="Number of steps to take during environment reset.")
parser.add_argument("--max_steps", type=int, default=350, help="Maximum number of steps before forcing a reset.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Import remaining modules."""
import gymnasium as gym  # noqa: F401, E402
import torch  # noqa: F401, E402
from data_collection import DataCollectionManager  # noqa: F401, E402
from meta_state_machine.ultrasound_state_machine import UltrasoundStateMachine  # noqa: F401, E402
from modules.force_module import ForceControlModule  # noqa: F401, E402
from modules.orientation_module import OrientationControlModule  # noqa: F401, E402
from modules.path_planning_module import PathPlanningModule  # noqa: F401, E402
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: F401, E402
from robotic_us_ext import tasks  # noqa: F401, E402
from utils import (  # noqa: F401, E402
    KeyboardHandler,
    RobotPositions,
    RobotQuaternions,
    UltrasoundState,
    capture_camera_images,
    compute_relative_action,
    get_robot_obs,
)


def main():
    """Main function."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    assert args_cli.num_envs == 1, "Number of environments must be 1 for this script"
    # Ensure no timeout
    env_cfg.terminations.time_out = None
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Setup data collection if requested
    data_collector = None
    if args_cli.num_episodes > 0:
        data_collector = DataCollectionManager(
            task_name=args_cli.task,
            num_episodes=args_cli.num_episodes,
            num_envs=args_cli.num_envs,
            device=args_cli.device,
        )
    # reset environment at start
    obs = env.reset()

    # Start the keyboard listener in a separate thread
    keyboard_handler = KeyboardHandler()
    keyboard_handler.start_keyboard_listener()

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Initialize control modules
    modules = {
        "force": ForceControlModule(device=args_cli.device, use_quaternion=True),
        "orientation": OrientationControlModule(device=args_cli.device, use_quaternion=True),
        "path_planning": PathPlanningModule(device=args_cli.device, use_quaternion=True),
    }

    # Initialize state machine
    state_machine = UltrasoundStateMachine(modules, device=args_cli.device)

    reset_pos = torch.tensor(RobotPositions.SETUP, device=args_cli.device)
    reset_quat = torch.tensor(RobotQuaternions.DOWN, device=args_cli.device)
    reset_tensor = torch.cat([reset_pos, reset_quat], dim=-1)
    reset_tensor = reset_tensor.repeat(env.unwrapped.num_envs, 1)

    count = 0
    while simulation_app.is_running() and (
        data_collector is None or data_collector.completed_episodes < args_cli.num_episodes
    ):
        with torch.inference_mode():
            # Reset the environment on 'r' key press
            # Handle reset conditions
            if (
                str(state_machine.sm_state.state) == str(UltrasoundState.DONE)
                or (count % args_cli.max_steps == 0 and count > 0)
                or keyboard_handler.reset_flag
            ):
                keyboard_handler.reset_flag = False

                if data_collector is not None:
                    if str(state_machine.sm_state.state) == str(UltrasoundState.DONE):
                        data_collector.on_episode_complete()
                    else:
                        print(f"State: {state_machine.sm_state.state}")
                        data_collector.on_episode_reset()

                print("-" * 80)
                print("[INFO]: Resetting environment...")
                count = 0
                env.reset()
                state_machine.reset()
                for _ in range(args_cli.reset_steps):
                    robot_obs = get_robot_obs(env)
                    rel_commands = compute_relative_action(reset_tensor, robot_obs)
                    obs, rew, terminated, truncated, info_ = env.step(rel_commands)

            robot_obs = get_robot_obs(env)

            # Compute combined action from all modules
            rel_commands, abs_commands = state_machine.compute_action(env, robot_obs[0])
            print(f"Step@{count} w/ action: {abs_commands}")
            print(f"State: {state_machine.sm_state.state}")
            # Step using relative commands
            obs, rew, terminated, truncated, info_ = env.step(rel_commands)

            # Record data if collecting
            if data_collector is not None:
                # Capture camera images if data collection is happening
                rgb_images, depth_images = capture_camera_images(env, args_cli.camera_names, device=args_cli.device)
                obs["rgb_images"] = rgb_images
                obs["depth_images"] = depth_images

                data_collector.record_step(
                    env,
                    obs,
                    rel_commands,
                    abs_commands,
                    robot_obs,
                    state_machine.sm_state.state.value,  # Add current state as string
                )
            # Update counter
            count += 1
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
