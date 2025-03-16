import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import omni.isaac.lab.utils.math as math_utils  # noqa: F401, E402
import onnxruntime as ort
import torch
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.utils.math import compute_pose_error, quat_from_euler_xyz
from scipy.spatial.transform import Rotation


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


def compute_transform_chain(env, route: list[str]):
    """
    Compute the transformation from the origin frame to the target frame using the route of frames.

    Args:
        frame_name: The name of the frame.
        route: The route of frames to compute the transformation.

    Returns:
        quat, pos: The quaternion and position from the 1st frame to the last frame.
    """

    # Create rotation component of the transform matrix
    def transform_name(start, end):
        return f"{start}_to_{end}_transform"

    if len(route) <= 1:
        raise ValueError(f"Route must contain at least two frames: {route}")

    quat = None
    pos = None

    for i in range(len(route) - 1):
        start = route[i]
        end = route[i + 1]

        try:
            transform_obj = env.unwrapped.scene[transform_name(start, end)]
        except Exception as e:
            print(f"Error getting transform object {transform_name(start, end)}: {e}")
            raise e

        next_quat = transform_obj.data.target_quat_source[0]
        next_pos = transform_obj.data.target_pos_source[0]

        if quat is None and pos is None:
            print(f"quat and pos for {transform_name(start, end)}")
            quat = next_quat
            pos = next_pos
        else:
            print(f"multiply quat and pos with {transform_name(start, end)}")
            # The order of updating pos and quat can't be switched be below
            pos = pos + math_utils.quat_apply(quat, next_pos)
            quat = math_utils.quat_mul(quat, next_quat)

    return quat, pos


def ov_to_nifti_orientation(
    ov_quat,
    rotation_matrix: None | torch.Tensor = None,
    ov_down_quat: None | Sequence[float] = None,
    organ_down_quat: None | Sequence[float] = None,
):
    """
    Convert quaternion from Omniverse to organ coordinate system Euler angles

    Args:
        ov_quat: Quaternion in [w, x, y, z] format from Omniverse
        rotation_matrix: 3x3 rotation matrix that maps positions
            from Isaac Sim coordinate system to organ coordinate system
        ov_down_quat: Quaternion in [w, x, y, z] format for Omniverse "down" direction,
            default is RobotQuaternions.DOWN
        organ_down_quat: Euler angles in [x, y, z] format for organ "down" direction,
            default is OrganEulerAngles.DOWN

    Returns:
        Euler angles in organ coordinate system [x, y, z] in radians
    """
    # Set default values if not provided
    if ov_down_quat is None:
        ov_down_quat = RobotQuaternions.DOWN  # Omniverse "down" quaternion [w, x, y, z]

    if organ_down_quat is None:
        organ_down_quat = [-np.pi / 2, -np.pi / 2, 0]  # Organ "down" Euler angles [x, y, z]

    # set default coordinate system transformation if not provided
    if rotation_matrix is None:
        coord_transform = np.array(CoordinateTransform.OMNIVERSE_TO_ORGAN)
    else:
        coord_transform = rotation_matrix

    # Step 1: Convert Omniverse quaternion to rotation matrix
    ov_rot = Rotation.from_quat(ov_quat, scalar_first=True)

    # Step 2: Create reference orientations
    # Define "down" in Omniverse using the provided quaternion
    ov_down_rot = Rotation.from_quat(ov_down_quat, scalar_first=True)

    # Define "down" in organ coordinates using the provided Euler angles
    organ_down_rot = Rotation.from_euler("xyz", organ_down_quat, degrees=False)

    # Step 3: Apply coordinate system transformation to Omniverse rotation
    # First convert to matrix representation
    ov_matrix = ov_rot.as_matrix()
    # Transform the rotation matrix to organ coordinate system
    transformed_matrix = coord_transform @ ov_matrix @ coord_transform.T
    # Convert back to a rotation object
    transformed_rot = Rotation.from_matrix(transformed_matrix)

    # Step 4: Apply this same relative rotation to the organ "down" orientation
    # First transform the Omniverse down direction
    ov_down_matrix = ov_down_rot.as_matrix()
    transformed_down_matrix = coord_transform @ ov_down_matrix @ coord_transform.T
    transformed_down_rot = Rotation.from_matrix(transformed_down_matrix)

    # Combine the transformed down direction with the relative rotation
    final_rot = organ_down_rot * transformed_down_rot.inv() * transformed_rot

    # Step 5: Convert to Euler angles
    organ_euler = final_rot.as_euler("xyz", degrees=False)

    return organ_euler


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
    # stack the euler angles into roll pich yaw tensor
    euler_angles = np.array([roll.squeeze().cpu().numpy(), pitch.squeeze().cpu().numpy(), yaw.squeeze().cpu().numpy()])

    # Optional logging for debugging
    if log:
        # print the results in euler angles in degrees
        print("pos:", pos_np)
        print("euler angles:", np.degrees(euler_angles))

    # Return position as numpy array and orientation as Euler angles
    return pos_np, euler_angles


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
