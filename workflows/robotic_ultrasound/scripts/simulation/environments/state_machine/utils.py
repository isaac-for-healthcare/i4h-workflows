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

import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import h5py
import isaaclab.utils.math as math_utils  # noqa: F401
import numpy as np
import torch


# MARK: - State Machine Enums + Dataclasses
class UltrasoundState(Enum):
    """States for the ultrasound procedure."""

    SETUP = "setup"
    APPROACH = "approach"
    CONTACT = "contact"
    SCANNING = "scanning"
    DONE = "done"


@dataclass(frozen=True)
class PhantomScanPositions:
    """
    Defines scan positions in the phantom's local coordinate frame.
    All positions are relative to the phantom's origin.
    """

    SCAN_START: tuple[float, float, float] = (0.0030, 0.05, 0.2)
    SCAN_CONTACT: tuple[float, float, float] = (0.0030, 0.05, 0.0983)  # Starting position of scan
    SCAN_END: tuple[float, float, float] = (-0.0473, -0.0399, 0.0857)


@dataclass(frozen=True)
class RobotPositions:
    """Robot position configurations stored as torch tensors."""

    SETUP: tuple[float, float, float] = (0.3229, -0.0110, 0.3000)
    ORGAN_OFFSET: tuple[float, float, float] = (0.0040, -0.0519 + 0.08, 0.1510)
    TARGET_OFFSET: tuple[float, float, float] = (0.0040, -0.08, 0.00)


@dataclass(frozen=True)
class RobotQuaternions:
    """Robot quaternion configurations stored as torch tensors."""

    # Define Euler angles in degrees for DOWN orientation (90 degrees around Y)
    DOWN_EULER_DEG = (180.0, 0.0, 180)
    # Convert to radians and then to quaternion
    DOWN: tuple[float, float, float, float] = tuple(
        math_utils.quat_from_euler_xyz(
            torch.tensor(math.radians(DOWN_EULER_DEG[0])),  # X rotation (rad)
            torch.tensor(math.radians(DOWN_EULER_DEG[1])),  # Y rotation (rad)
            torch.tensor(math.radians(DOWN_EULER_DEG[2])),  # Z rotation (rad)
        ).tolist()
    )


@dataclass(frozen=True)
class OrganEulerAngles:
    """Organ Euler angle configurations stored as torch tensors."""

    DOWN: tuple[float, float, float] = (-np.pi / 2, -np.pi / 2, 0)


@dataclass(frozen=True)
class CoordinateTransform:
    """Coordinate transformation matrices stored as torch tensors.

    The coordinate transformation matrix performs the following axis mapping:
        - x-axis maps to negative x-axis
        - y-axis maps to negative z-axis
        - z-axis maps to negative y-axis
    """

    OMNIVERSE_TO_ORGAN: torch.Tensor = torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float64)


@dataclass
class SMState:
    """State machine state container."""

    state: UltrasoundState = UltrasoundState.SETUP
    robot_obs: torch.Tensor = None
    contact_normal_force: torch.Tensor = None

    def reset(self) -> None:
        """Reset the state machine state."""
        self.state = UltrasoundState.SETUP
        self.robot_obs = None
        self.contact_normal_force = None


# MARK: - Util Functions
def compute_relative_action(action: torch.Tensor, robot_obs: torch.Tensor, return_np: bool = False) -> torch.Tensor:
    """Compute the relative action from the robot observation."""
    pos_sm = action[:, :3]
    rot_sm = action[:, 3:]
    delta_pos, delta_angle = math_utils.compute_pose_error(
        robot_obs[0, :, :3], robot_obs[0, :, 3:], pos_sm, rot_sm, rot_error_type="axis_angle"
    )
    rel_action = torch.cat([delta_pos, delta_angle], dim=-1)
    if return_np:
        return rel_action.cpu().numpy()
    else:
        return rel_action


def capture_camera_images(env, cam_names, include_seg=False, device="cuda"):
    """
    Captures RGB and depth images from specified cameras

    Args:
        env: The environment containing the cameras
        cam_names (list): List of camera names to capture from
        include_seg (bool): Whether to include semantic segmentation images
        device (str): Device to use for tensor operations

    Returns:
        tuple: If include_seg is False:
            (stacked_rgbs, stacked_depths, None) - Tensors of shape (1, num_cams, H, W, 3),
            (1, num_cams, H, W), and None
        If include_seg is True:
            (stacked_rgbs, stacked_depths, stacked_segs) - Tensors of shape (1, num_cams, H, W, 3),
            (1, num_cams, H, W), and (1, num_cams, H, W)
    """
    depths, rgbs, segs = [], [], []
    for cam_name in cam_names:
        camera_data = env.unwrapped.scene[cam_name].data

        # Extract RGB and depth images
        rgb = camera_data.output["rgb"][..., :3].squeeze(0)
        depth = camera_data.output["distance_to_image_plane"].squeeze(0)
        if include_seg:
            seg = camera_data.output["semantic_segmentation"][..., :3].squeeze(0)
            segs.append(seg)

        # Append to lists
        rgbs.append(rgb)
        depths.append(depth)

    # Stack results
    stacked_rgbs = torch.stack(rgbs).unsqueeze(0)
    stacked_depths = torch.stack(depths).unsqueeze(0)
    if include_seg:
        stacked_segs = torch.stack(segs).unsqueeze(0)
        return stacked_rgbs, stacked_depths, stacked_segs
    return stacked_rgbs, stacked_depths, None


def get_robot_obs(env):
    """Get the robot observation from the environment."""
    robot_data = env.unwrapped.scene["ee_frame"].data
    robot_pos = robot_data.target_pos_w - env.unwrapped.scene.env_origins
    robot_quat = robot_data.target_quat_w
    robot_obs = torch.cat([robot_pos, robot_quat], dim=-1)
    return robot_obs


def get_joint_states(env):
    """Get the robot joint states from the environment."""
    robot_data = env.unwrapped.scene["robot"].data
    robot_joint_pos = robot_data.joint_pos
    return robot_joint_pos.cpu().numpy()


def scale_points(points: Sequence[float], scale: float = 1000.0) -> torch.Tensor:
    """
    Scale points from one unit to another (e.g., meters to millimeters).

    Args:
        points: Points to scale
        scale: Scaling factor (default: 1000.0 for meters to millimeters)

    Returns:
        Scaled points
    """
    points_tensor = torch.tensor(points, dtype=torch.float64)
    return points_tensor * scale


def compute_transform_sequence(env, sequence_order: list[str]):
    """
    Compute the transformation from the 1st frame to the last frame by following the sequence order of frames.
    This is useful for getting the transformation in the scene, when only frame-to-frame transformations are available.
    The definition of `frame` is the same as the `frame` in the `env.unwrapped.scene`,
    but it is not necessary to have the actual frame in the scene.
    For example, we can use the `us` frame as long as a transformation from `us` to another actual frame is available.


    Args:
        env: the environment object containing the transformations in the scene.
        sequence_order: the sequence order of frames to compute the transformation.
            For example, if we have frames `A`, `B`, `C`, and we want to compute the transformation from `A` to `C`,
            the sequence_order should be `[A, B, C]`.

    Returns:
        quat, pos: The quaternion and position from the 1st frame to the last frame.
    """

    # Create rotation component of the transform matrix
    def transform_name(start, end):
        return f"{start}_to_{end}_transform"

    if len(sequence_order) <= 1:
        raise ValueError(f"sequence_order must contain at least two frames: {sequence_order}")

    quat = None
    pos = None

    for i in range(len(sequence_order) - 1):
        start = sequence_order[i]
        end = sequence_order[i + 1]

        try:
            transform_obj = env.unwrapped.scene[transform_name(start, end)]
        except Exception as e:
            print(f"Error getting transform object {transform_name(start, end)}: {e}")
            raise e

        next_quat = transform_obj.data.target_quat_source[0]
        next_pos = transform_obj.data.target_pos_source[0]

        if quat is None and pos is None:
            quat = next_quat
            pos = next_pos
        else:
            # The order of updating pos and quat can't be switched be below
            pos = pos + math_utils.quat_apply(quat, next_pos)
            quat = math_utils.quat_mul(quat, next_quat)

    return quat, pos


def get_probe_pos_ori(quat_mesh_to_us, pos_mesh_to_us, scale: float = 1000.0, log=False):
    """
    Convert the probe position and orientation to match input formats of raysim
    Args:
        quat_mesh_to_us: Quaternion in [w, x, y, z] format that can map mesh obj to us image view
        pos_mesh_to_us: Position in meters that can map mesh obj to us image view
        scale: Scaling factor to convert from meters to millimeters (default: 1000.0)
        log: If True, print the transformed position and orientation values for debugging

    Returns:
        transformed_position: numpy array of shape (3,) in organ coordinate system (millimeters)
        transformed_orientation: numpy array of shape (3,) with Euler angles in organ system (degrees)
    """
    # Get probe data from the end effector frame
    # scale the position from m to mm
    pos = pos_mesh_to_us * scale
    if isinstance(pos, torch.Tensor):
        pos_np = pos.cpu().numpy().squeeze()
    else:
        pos_np = pos.squeeze()

    # convert the quat to euler angles
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_mesh_to_us)
    # stack the euler angles into roll pitch yaw tensor
    euler_angles = np.array([roll.squeeze().cpu(), pitch.squeeze().cpu(), yaw.squeeze().cpu()])

    # Optional logging for debugging
    if log:
        # print the results in euler angles in degrees
        print("pos:", pos_np)
        print("euler angles:", np.degrees(euler_angles))

    # Return position as numpy array and orientation as Euler angles
    return pos_np, euler_angles


def reset_organ_to_position(env, object_position, device="cuda:0"):
    """Reset the organ position."""
    organs = env.unwrapped.scene._rigid_objects["organs"]
    root_state = organs.data.default_root_state.clone()
    root_state[:, :7] = torch.tensor(object_position[0, :7], device=device)
    # write to the sim
    organs.write_root_state_to_sim(root_state)


def reset_robot_to_position(env, robot_initial_joint_state, joint_vel=None, device="cuda:0"):
    """Reset the robot to the initial joint state."""
    robot = env.unwrapped.scene["robot"]
    joint_pos = torch.tensor(robot_initial_joint_state[0], device=device).unsqueeze(0)
    if joint_vel is not None:
        joint_vel = torch.tensor(joint_vel[0], device=device)
    else:
        print("No joint velocity provided, setting to zero")
        joint_vel = torch.zeros_like(joint_pos)
    # set joint positions
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def validate_hdf5_path(path):
    """Validate that the path contains HDF5 files and return the number of episodes."""
    if os.path.isdir(path):
        # Check if directory contains HDF5 files that start with "data_"
        hdf5_files = [file for file in os.listdir(path) if file.startswith("data_") and file.endswith(".hdf5")]
        return len(hdf5_files)
    elif os.path.isfile(path) and path.endswith(".hdf5"):
        return 1
    return 0


def get_hdf5_episode_data(data_root, data_key: str):
    """Get episode data from an open HDF5 file object."""
    base_path = "data/demo_0"
    try:
        dataset = data_root[f"{base_path}/{data_key}"][()]
    except KeyError as e:
        print(f"Error loading data using key: {base_path}/{data_key}")
        print(f"Available keys: {list(data_root[base_path].keys())}")
        raise e
    return dataset


def _load_hdf5_dataset(data_path: str, episode_idx: int, key: str):
    """Helper function to open HDF5 file and retrieve data for a specific key."""
    if not data_path.endswith(".hdf5"):
        file_path = os.path.join(data_path, f"data_{episode_idx}.hdf5")
    else:
        # If data_path is a direct file path, episode_idx might not be relevant unless used for logging/consistency.
        # Assuming if a direct file path is given, it's the specific file we need.
        file_path = data_path

    try:
        with h5py.File(file_path, "r") as root:
            data = get_hdf5_episode_data(root, key)
            return data
    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from {file_path} with key '{key}': {str(e)}")
        return None


def reset_scene_to_initial_state(
    env,
    hdf5_path: str,
    episode_idx: int,
    action_key: str,
    torso_obs_key: str,
    joint_state_key: str,
    joint_vel_key: str,
):
    """Reset the scene to the initial state.

    Args:
        env: The environment object.
        hdf5_path: The path to the HDF5 file or directory.
        episode_idx: The index of the episode.
        action_key: The key to the action data.
        torso_obs_key: The key to the torso observation data.
        joint_state_key: The key to the joint state data.
        joint_vel_key: The key to the joint velocity data.
    """

    actions = _load_hdf5_dataset(hdf5_path, episode_idx, action_key)
    object_position = _load_hdf5_dataset(hdf5_path, episode_idx, torso_obs_key)
    reset_organ_to_position(env, object_position)
    robot_initial_joint_state = _load_hdf5_dataset(hdf5_path, episode_idx, joint_state_key)
    try:
        joint_vel = _load_hdf5_dataset(hdf5_path, episode_idx, joint_vel_key)
    except KeyError:
        print("No joint velocity provided, setting to zero")
        joint_vel = None
    reset_robot_to_position(env, robot_initial_joint_state, joint_vel=joint_vel)
    return actions
