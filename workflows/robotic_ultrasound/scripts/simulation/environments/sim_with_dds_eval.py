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
import os

import gymnasium as gym
import numpy as np
import torch
import h5py
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from dds.schemas.usp_info import UltraSoundProbeInfo
from dds.subscriber import SubscriberWithQueue
from isaaclab.app import AppLauncher
from simulation.environments.state_machine.utils import (
    RobotPositions,
    RobotQuaternions,
    capture_camera_images,
    compute_relative_action,
    compute_transform_sequence,
    get_joint_states,
    get_probe_pos_ori,
    get_robot_obs,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Run simulation in a single-arm manipulator, communication via DDS.")
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
    "--rti_license_file",
    type=str,
    default=os.getenv("RTI_LICENSE_FILE"),
    help="the path of rti_license_file. Default will use environment variables `RTI_LICENSE_FILE`",
)
parser.add_argument("--infer_domain_id", type=int, default=0, help="domain id to publish data for inference.")
parser.add_argument("--viz_domain_id", type=int, default=1, help="domain id to publish data for visualization.")
parser.add_argument(
    "--topic_in_room_camera",
    type=str,
    default="topic_room_camera_data_rgb",
    help="topic name to consume room camera rgb",
)
parser.add_argument(
    "--topic_in_room_camera_depth",
    type=str,
    default="topic_room_camera_data_depth",
    help="topic name to consume room camera depth",
)
parser.add_argument(
    "--topic_in_wrist_camera",
    type=str,
    default="topic_wrist_camera_data_rgb",
    help="topic name to consume wrist camera rgb",
)
parser.add_argument(
    "--topic_in_wrist_camera_depth",
    type=str,
    default="topic_wrist_camera_data_depth",
    help="topic name to consume wrist camera depth",
)
parser.add_argument(
    "--topic_in_franka_pos",
    type=str,
    default="topic_franka_info",
    help="topic name to consume franka pos",
)
parser.add_argument(
    "--topic_in_probe_pos",
    type=str,
    default="topic_ultrasound_info",
    help="topic name to consume probe pos",
)
parser.add_argument(
    "--topic_out",
    type=str,
    default="topic_franka_ctrl",
    help="topic name to publish generated franka actions",
)
parser.add_argument("--width", type=int, default=224, help="width of the image")
parser.add_argument("--height", type=int, default=224, help="height of the image")
parser.add_argument("--log_probe_pos", action="store_true", default=False, help="Log probe position.")
parser.add_argument(
    "--scale", type=float, default=1000.0, help="Scale factor to convert from omniverse to organ coordinate system."
)
parser.add_argument("--use_rel", action="store_true", default=False, help="Use relative actions.")
parser.add_argument("--data_path", type=str, required=True, help="Path to either .hdf5 file containing recorded data.")
parser.add_argument("--policy_name", type=str, required=True, default='', help="policy name to save the data.")
parser.add_argument("--action_dim", type=int, default=16, help="action dimension. 16 for groot, 50 for pi0.")



# append AppLauncher cli argruments
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
reset_flag = False

# isort: off
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Import extensions to set up environment tasks
from robotic_us_ext import tasks  # noqa: F401

# isort: on

pub_data = {
    "room_cam": None,
    "wrist_cam": None,
    "joint_pos": None,
    "probe_pos": None,
    "probe_ori": None,
    "room_cam_depth": None,
    "wrist_cam_depth": None,
}
hz = 30

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


class RoomCamPublisher(Publisher):
    def __init__(self, topic: str, domain_id: int, rgb: bool = True):
        super().__init__(topic, CameraInfo, 1 / hz, domain_id)
        self.rgb = rgb

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.focal_len = 12.0
        output.height = args_cli.height
        output.width = args_cli.width
        if self.rgb:
            output.data = pub_data["room_cam"].tobytes()
        else:
            output.data = pub_data["room_cam_depth"].tobytes()
        return output


class WristCamPublisher(Publisher):
    def __init__(self, topic: str, domain_id: int, rgb: bool = True):
        super().__init__(topic, CameraInfo, 1 / hz, domain_id)
        self.rgb = rgb

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.height = args_cli.height
        output.width = args_cli.width
        if self.rgb:
            output.data = pub_data["wrist_cam"].tobytes()
        else:
            output.data = pub_data["wrist_cam_depth"].tobytes()
        return output


class PosPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_franka_pos, FrankaInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = FrankaInfo()
        output.joints_state_positions = pub_data["joint_pos"].tolist()
        return output


class ProbePosPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_probe_pos, UltraSoundProbeInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = UltraSoundProbeInfo()
        output.position = pub_data["probe_pos"].tolist()
        output.orientation = pub_data["probe_ori"].tolist()
        return output


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
    env_cfg.scene.room_camera.width = args_cli.width
    env_cfg.scene.room_camera.height = args_cli.height
    env_cfg.scene.wrist_camera.width = args_cli.width
    env_cfg.scene.wrist_camera.height = args_cli.height
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    obs = env.reset()

    if args_cli.rti_license_file is not None:
        if not os.path.isabs(args_cli.rti_license_file):
            raise ValueError("RTI license file must be an existing absolute path.")
        os.environ["RTI_LICENSE_FILE"] = args_cli.rti_license_file

    # Recommended to use 40 steps to allow enough steps to reset the SETUP position of the robot

    infer_r_cam_writer = RoomCamPublisher(topic=args_cli.topic_in_room_camera, domain_id=args_cli.infer_domain_id)
    infer_w_cam_writer = WristCamPublisher(topic=args_cli.topic_in_wrist_camera, domain_id=args_cli.infer_domain_id)
    infer_pos_writer = PosPublisher(args_cli.infer_domain_id)
    viz_r_cam_writer = RoomCamPublisher(topic=args_cli.topic_in_room_camera, domain_id=args_cli.viz_domain_id)
    viz_w_cam_writer = WristCamPublisher(topic=args_cli.topic_in_wrist_camera, domain_id=args_cli.viz_domain_id)
    viz_r_cam_depth_writer = RoomCamPublisher(
        topic=args_cli.topic_in_room_camera_depth, domain_id=args_cli.viz_domain_id, rgb=False
    )
    viz_w_cam_depth_writer = WristCamPublisher(
        topic=args_cli.topic_in_wrist_camera_depth, domain_id=args_cli.viz_domain_id, rgb=False
    )
    viz_pos_writer = PosPublisher(args_cli.viz_domain_id)
    viz_probe_pos_writer = ProbePosPublisher(args_cli.viz_domain_id)
    infer_reader = SubscriberWithQueue(args_cli.infer_domain_id, args_cli.topic_out, FrankaCtrlInput, 1 / hz)
    infer_reader.start()
    import time
    time.sleep(10)
    # Number of steps played before replanning
    replan_steps = 5

    episode_idx = 0
    action_idx = 0
    max_timesteps = 250

    # simulate environment
    while simulation_app.is_running():
        global reset_flag
        
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
                action_plan = collections.deque()
                robot_obs = []
                for t in range(max_timesteps):
                    # get and publish the current images and joint positions
                    rgb_images, depth_images = capture_camera_images(
                        env, ["room_camera", "wrist_camera"], device=env.unwrapped.device
                    )
                    pub_data["room_cam"], pub_data["room_cam_depth"] = (
                        rgb_images[0, 0, ...].cpu().numpy(),
                        depth_images[0, 0, ...].cpu().numpy(),
                    )
                    pub_data["wrist_cam"], pub_data["wrist_cam_depth"] = (
                        rgb_images[0, 1, ...].cpu().numpy(),
                        depth_images[0, 1, ...].cpu().numpy(),
                    )
                    pub_data["joint_pos"] = get_joint_states(env)[0]

                    robot_obs.append(get_robot_obs(env))
                    # Get the pose of the mesh objects (mesh)
                    # The mesh objects are aligned with the organ (organ) in the US image view (us)
                    # The US is attached to the end-effector (ee), so we have the following computation logics:
                    # Each frame-to-frame transformation is available in the scene
                    # mesh -> organ -> ee -> us
                    quat_mesh_to_us, pos_mesh_to_us = compute_transform_sequence(env, ["mesh", "organ", "ee", "us"])
                    pub_data["probe_pos"], pub_data["probe_ori"] = get_probe_pos_ori(
                        quat_mesh_to_us, pos_mesh_to_us, scale=args_cli.scale, log=args_cli.log_probe_pos
                    )
                    viz_r_cam_writer.write()
                    viz_w_cam_writer.write()
                    viz_r_cam_depth_writer.write()
                    viz_w_cam_depth_writer.write()
                    viz_pos_writer.write()
                    viz_probe_pos_writer.write()
                    if not action_plan:
                        # publish the images and joint positions when run policy inference
                        infer_r_cam_writer.write()
                        infer_w_cam_writer.write()
                        infer_pos_writer.write()

                        ret = None
                        while ret is None:
                            ret = infer_reader.read_data()
                        o: FrankaCtrlInput = ret
                        action_chunk = np.array(o.joint_positions, dtype=np.float32).reshape(args_cli.action_dim, 6)
                        action_plan.extend(action_chunk[:replan_steps])

                    action = action_plan.popleft()

                    action = action.astype(np.float32)

                    # convert to torch
                    action = torch.tensor(action, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)

                    # step the environment
                    obs, rew, terminated, truncated, info_ = env.step(action)

                env.reset()
                robot_obs = torch.stack(robot_obs, dim=0)
                print(f"robot_obs shape: {robot_obs.shape}, saved to {args_cli.data_path}/robot_obs_{episode_idx}.npz")
                np.savez(
                    os.path.join(args_cli.data_path, f"{args_cli.policy_name}_robot_obs_{episode_idx}.npz"),
                    robot_obs=robot_obs.cpu().numpy()
                )

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

    infer_reader.stop()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
