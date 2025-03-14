import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import omni.isaac.lab.utils.math as math_utils
import onnxruntime as ort
import torch
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.utils.math import compute_pose_error, quat_from_euler_xyz


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

    SETUP: tuple[float, float, float] = (0.0178, -0.04, 0.6)
    ORGAN_OFFSET: tuple[float, float, float] = (0.0178, -0.04, 0.0968)
    TARGET_OFFSET: tuple[float, float, float] = (0.0476, 0.0469, 0.0850)


@dataclass(frozen=True)
class RobotQuaternions:
    """Robot quaternion configurations stored as torch tensors."""

    # Define Euler angles in degrees for DOWN orientation (90 degrees around Y)
    DOWN_EULER_DEG = (180.0, 0.0, 180)
    # Convert to radians and then to quaternion
    DOWN: tuple[float, float, float, float] = tuple(
        quat_from_euler_xyz(
            torch.tensor(math.radians(DOWN_EULER_DEG[0])),  # X rotation (rad)
            torch.tensor(math.radians(DOWN_EULER_DEG[1])),  # Y rotation (rad)
            torch.tensor(math.radians(DOWN_EULER_DEG[2])),  # Z rotation (rad)
        ).tolist()
    )


@dataclass(frozen=True)
class OrganEulerAngles:
    """Organ Euler angle configurations stored as torch tensors."""

    DOWN_EULER_DEG: tuple[float, float, float] = (-90, -90, 0)
    DOWN: tuple[float, float, float, float] = tuple(
        quat_from_euler_xyz(
            torch.tensor(math.radians(DOWN_EULER_DEG[0])),  # X rotation (rad)
            torch.tensor(math.radians(DOWN_EULER_DEG[1])),  # Y rotation (rad)
            torch.tensor(math.radians(DOWN_EULER_DEG[2])),  # Z rotation (rad)
        ).tolist()
    )


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
    delta_pos, delta_angle = compute_pose_error(
        robot_obs[0, :, :3], robot_obs[0, :, 3:], pos_sm, rot_sm, rot_error_type="axis_angle"
    )
    rel_action = torch.cat([delta_pos, delta_angle], dim=-1)
    if return_np:
        return rel_action.cpu().numpy()
    else:
        return rel_action


def capture_camera_images(env, cam_names, device="cuda"):
    """
    Captures RGB and depth images from specified cameras

    Args:
        env: The environment containing the cameras
        cam_names (list): List of camera names to capture from
        device (str): Device to use for tensor operations

    Returns:
        tuple: (stacked_rgbs, stacked_depths) - Tensors of shape (1, num_cams, H, W, 3) and (1, num_cams, H, W)
    """
    depths, rgbs = [], []
    for cam_name in cam_names:
        camera_data = env.unwrapped.scene[cam_name].data

        # Extract RGB and depth images
        rgb = camera_data.output["rgb"][..., :3].squeeze(0)
        depth = camera_data.output["distance_to_image_plane"].squeeze(0)

        # Append to lists
        rgbs.append(rgb)
        depths.append(depth)

    # Stack results
    stacked_rgbs = torch.stack(rgbs).unsqueeze(0)
    stacked_depths = torch.stack(depths).unsqueeze(0)

    return stacked_rgbs, stacked_depths


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


def compute_transform_matrix(
    ov_point: Sequence[float],
    nifti_point: Sequence[float],
    rotation_matrix: None | torch.Tensor = None,
):
    """
    Create a transform matrix to convert from Omniverse coordinates (meters) to NIFTI coordinates (millimeters)

    Args:
        ov_point: point in Omniverse coordinates (meters)
        nifti_point: point in NIFTI coordinates (millimeters)
        rotation_matrix: Optional rotation matrix to convert from Omniverse to NIFTI coordinates.
            If None, the default rotation matrix CoordinateTransform.OMNIVERSE_TO_ORGAN will be used.

    Returns:
        A 4x4 homogeneous transformation matrix that maps points from Omniverse to NIFTI coordinates.
    """
    # Create rotation component of the transform matrix
    R = CoordinateTransform.OMNIVERSE_TO_ORGAN if rotation_matrix is None else rotation_matrix

    # Convert input points to tensors
    ov_point = torch.tensor(ov_point, dtype=torch.float64).unsqueeze(-1)
    nifti_point = torch.tensor(nifti_point, dtype=torch.float64)

    # Calculate translation component
    t = nifti_point - (R @ ov_point).squeeze(-1)

    # Create full 4x4 transform_matrix matrix
    transform_matrix = torch.eye(4, dtype=torch.float64)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = t

    return transform_matrix


def ov_to_nifti_orientation(
    env,
    ov_quat,
    rotation_matrix: None | torch.Tensor = None,
    ov_down_quat: None | Sequence[float] = None,
    organ_down_quat: None | Sequence[float] = None,
):
    """
    Convert quaternion from Omniverse to organ coordinate system Euler angles

    Args:
        env: The simulation environment containing the robot data
        ov_quat: Quaternion in [w, x, y, z] format from Omniverse
        rotation_matrix: 3 homogeneous transformation matrix that maps positions
            from Isaac Sim coordinate system to organ coordinate system
        ov_down_quat: Quaternion in [w, x, y, z] format for Omniverse "down" direction,
            default is RobotQuaternions.DOWN
        organ_down_quat: Euler angles in [x, y, z] format for organ "down" direction,
            default is OrganEulerAngles.DOWN

    Returns:
        Euler angles in organ coordinate system [x, y, z] in radians
    """
    # Set default values if not provided
    ov_down_quat = RobotQuaternions.DOWN if ov_down_quat is None else ov_down_quat
    ov_down_quat = torch.tensor(ov_down_quat, dtype=torch.float64, device=env.unwrapped.device).unsqueeze(0)

    organ_down_quat = OrganEulerAngles.DOWN if organ_down_quat is None else organ_down_quat
    organ_down_quat = torch.tensor(organ_down_quat, dtype=torch.float64, device=env.unwrapped.device).unsqueeze(0)

    coord_transform = CoordinateTransform.OMNIVERSE_TO_ORGAN if rotation_matrix is None else rotation_matrix
    coord_transform = torch.tensor(coord_transform, dtype=torch.float64, device=env.unwrapped.device)


    _, delta_rot = compute_pose_error(
        torch.zeros(1, 3, device=env.unwrapped.device),
        ov_quat,
        torch.zeros(1, 3, device=env.unwrapped.device),
        ov_down_quat,
        rot_error_type="quat",
    )
    delta_rot = math_utils.matrix_from_quat(delta_rot)

    # Apply coordinate transformation to delta rotation
    delta_rot_ct = coord_transform @ delta_rot @ torch.inverse(coord_transform)

    # Apply coordinate transformation to organ down quaternion
    ee_from_organ_matrix = delta_rot_ct.inverse() @ math_utils.matrix_from_quat(organ_down_quat)

    # Convert to quaternion
    quat = math_utils.quat_from_matrix(ee_from_organ_matrix)

    quat = math_utils.normalize(quat)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    # stack the euler angles into roll pich yaw tensor
    euler_angles = np.array([roll.squeeze().cpu().numpy(), pitch.squeeze().cpu().numpy(), yaw.squeeze().cpu().numpy()])

    return euler_angles


def get_probe_pos_ori(env, transform_matrix, scale: float | None = 1000.0, log=False):
    """Get the probe position and orientation from the environment and transform to organ coordinate system.

    This function performs two separate transformations:
    1. Position transformation: Converts 3D position from Isaac Sim (meters) to organ coordinate system (millimeters)
        using a homogeneous transformation matrix
    2. Orientation transformation: Converts quaternion orientation from Isaac Sim to
        Euler angles in organ coordinate system

    Args:
        env: The simulation environment containing the probe data
        transform_matrix: 4x4 homogeneous transformation matrix that maps positions
            from Isaac Sim coordinate system to organ coordinate system
        scale: Scaling factor to convert from meters to millimeters (default: 1000.0)
            If None, no scaling is applied
        log: If True, print the transformed position and orientation values for debugging

    Returns:
        transformed_position: numpy array of shape (3,) in organ coordinate system (millimeters)
        transformed_orientation: numpy array of shape (3,) with Euler angles in organ system (degrees)
    """
    # Get probe data from the end effector frame
    probe_data = env.unwrapped.scene["ee_frame"].data

    # Get probe position and remove environment origins offset (raw tensor has shape (batch_size=1, num_envs=1, 3))
    probe_pos = probe_data.target_pos_w - env.unwrapped.scene.env_origins
    # Get probe orientation quaternion [w,x,y,z] (shape: batch_size=1, num_envs=1, 4)
    probe_quat = probe_data.target_quat_w.squeeze(0)

    # Remove batch dimensions to get a simple position vector of shape (3,)
    probe_pos_flat = probe_pos.squeeze(0).squeeze(0)

    if scale is not None:
        probe_pos_flat = scale_points(probe_pos_flat, scale=scale)

    # Convert to homogeneous coordinates by adding a 1 as the 4th component
    pos_homogeneous = torch.cat([probe_pos_flat, torch.tensor([1.0], device=probe_pos.device)], dim=-1)

    # Apply 4x4 transformation matrix to convert to organ coordinate system
    transformed_pos = transform_matrix.to(probe_pos.device) @ pos_homogeneous
    transformed_pos = transformed_pos[:3]

    transformed_ori = ov_to_nifti_orientation(env, probe_quat)

    # Optional logging for debugging
    if log:
        print(f"Raw position (OV world, mm): {probe_pos_flat}")
        print(f"Transformed position (CT world, mm): {transformed_pos}")
        print(f"Raw orientation (OV world, quat): {probe_quat}")
        print(f"Transformed orientation (CT world, Euler): {transformed_ori}")

    # Return position as numpy array and orientation as Euler angles
    return transformed_pos.cpu().numpy(), transformed_ori


def get_np_images(env):
    """Get numpy images from the environment."""
    third_person_img = convert_dict_to_backend(env.unwrapped.scene["room_camera"].data.output, backend="numpy")["rgb"]
    third_person_img = third_person_img[0, :, :, :3].astype(np.uint8)

    wrist_img1 = convert_dict_to_backend(env.unwrapped.scene["wrist_camera"].data.output, backend="numpy")["rgb"]
    wrist_img1 = wrist_img1[0, :, :, :3].astype(np.uint8)

    return third_person_img, wrist_img1


def load_onnx_model(model_path):
    """Load the ACT ONNX model."""
    providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    print(f"Using providers: {providers}")
    # Create an InferenceSession with GPU support
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"session using: {session.get_providers()}")
    return session
