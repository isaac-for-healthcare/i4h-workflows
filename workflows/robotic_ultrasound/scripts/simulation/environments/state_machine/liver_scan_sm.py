# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os

from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
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
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output.")

# Add DDS-related arguments
parser.add_argument("--rti_license_file", type=str, help="the path of rti_license_file.")
parser.add_argument("--infer_domain_id", type=int, default=0, help="domain id to publish data for inference.")
parser.add_argument("--viz_domain_id", type=int, default=1, help="domain id to publish data for visualization.")
parser.add_argument(
    "--topic_in_room_camera",
    type=str,
    default="topic_room_camera_data_rgb",
    help="topic name to consume room camera rgb",
)
parser.add_argument(
    "--topic_in_wrist_camera",
    type=str,
    default="topic_wrist_camera_data_rgb",
    help="topic name to consume wrist camera rgb",
)

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
from meta_state_machine.ultrasound_state_machine import UltrasoundStateMachine  # noqa: F401, E402
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: F401, E402
from robotic_us_ext import tasks  # noqa: F401, E402
from simulation.environments.state_machine.data_collection import DataCollectionManager  # noqa: F401, E402
from simulation.environments.state_machine.modules.force_module import ForceControlModule  # noqa: F401, E402
from simulation.environments.state_machine.modules.orientation_module import (  # noqa: F401, E402
    OrientationControlModule,
)
from simulation.environments.state_machine.modules.path_planning_module import PathPlanningModule  # noqa: F401, E402
from simulation.environments.state_machine.utils import (  # noqa: F401, E402
    RobotPositions,
    RobotQuaternions,
    UltrasoundState,
    capture_camera_images,
    compute_relative_action,
    get_robot_obs,
)

# Add publisher classes before main()
pub_data = {
    "room_cam": None,
    "wrist_cam": None,
}
hz = 30


class RoomCamPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_room_camera, CameraInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.focal_len = 12.0
        output.height = 224
        output.width = 224
        output.data = pub_data["room_cam"].tobytes()
        return output


class WristCamPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_wrist_camera, CameraInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.height = 224
        output.width = 224
        output.data = pub_data["wrist_cam"].tobytes()
        return output


def main():
    """Main function."""

    # Add after environment creation
    if args_cli.rti_license_file is None or not os.path.isabs(args_cli.rti_license_file):
        raise ValueError("RTI license file must be an existing absolute path.")
    os.environ["RTI_LICENSE_FILE"] = args_cli.rti_license_file
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

    if args_cli.debug:
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

    # initialize publishers
    viz_r_cam_writer = RoomCamPublisher(args_cli.viz_domain_id)
    viz_w_cam_writer = WristCamPublisher(args_cli.viz_domain_id)

    count = 0
    while simulation_app.is_running() and (
        data_collector is None or data_collector.completed_episodes < args_cli.num_episodes
    ):
        with torch.inference_mode():
            # Handle reset conditions
            if str(state_machine.sm_state.state) == str(UltrasoundState.DONE) or (
                count % args_cli.max_steps == 0 and count > 0
            ):
                if data_collector is not None:
                    if str(state_machine.sm_state.state) == str(UltrasoundState.DONE):
                        data_collector.on_episode_complete()
                    else:
                        if args_cli.debug:
                            print(f"State: {state_machine.sm_state.state}")
                        data_collector.on_episode_reset()

                if args_cli.debug:
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
            if args_cli.debug:
                print(f"Step@{count} w/ action: {abs_commands}")
                print(f"State: {state_machine.sm_state.state}")
            # Step using relative commands
            obs, rew, terminated, truncated, info_ = env.step(rel_commands)

            # Capture camera images if data collection is happening
            rgb_images, depth_images = capture_camera_images(env, args_cli.camera_names, device=args_cli.device)
            if args_cli.debug:
                print(f"RGB images: {rgb_images.shape}")
                print(f"Depth images: {depth_images.shape}")

            obs["rgb_images"] = rgb_images
            obs["depth_images"] = depth_images
            # Publish camera data
            pub_data["room_cam"] = rgb_images[0, 0, ...].cpu().numpy()
            pub_data["wrist_cam"] = rgb_images[0, 1, ...].cpu().numpy()
            if args_cli.debug:
                print("Publishing camera data to DDS")
                print(f"Room cam: {pub_data['room_cam'].shape}, dtype: {pub_data['room_cam'].dtype}")
                print(f"Wrist cam: {pub_data['wrist_cam'].shape}, dtype: {pub_data['wrist_cam'].dtype}")
            viz_r_cam_writer.write(0.1, 1.0)
            viz_w_cam_writer.write(0.1, 1.0)


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
