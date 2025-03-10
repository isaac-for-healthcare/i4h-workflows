from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import onnxruntime as ort
import torch
from omni.isaac.lab.utils.math import compute_pose_error
from pynput import keyboard
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
class RobotPositions:
    """Robot position configurations stored as torch tensors."""

    SETUP: tuple[float, float, float] = (0.3229, -0.0110, 0.3000)
    ORGAN_OFFSET: tuple[float, float, float] = (0.0040, -0.0519 + 0.08, 0.1510)
    TARGET_OFFSET: tuple[float, float, float] = (0.0040, -0.08, 0.00)


@dataclass(frozen=True)
class RobotQuaternions:
    """Robot quaternion configurations stored as torch tensors."""

    DOWN: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)


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
            If None, the default rotation matrix will be used.
            The default rotation matrix performs the following axis mapping:
            - x-axis remains as x-axis
            - y-axis maps to negative z-axis
            - z-axis maps to negative y-axis

    Returns:
        A 4x4 homogeneous transformation matrix that maps points from Omniverse to NIFTI coordinates.
    """
    # Create rotation component of the transform matrix
    if rotation_matrix is None:
        R = torch.tensor([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float64)
    else:
        R = rotation_matrix

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
    ov_quat, ov_down_quat: None | Sequence[float] = None, organ_down_quat: None | Sequence[float] = None
):
    """
    Convert quaternion from Omniverse to organ coordinate system Euler angles

    Args:
        ov_quat: Quaternion in [w, x, y, z] format from Omniverse
        ov_down_quat: Quaternion in [w, x, y, z] format for Omniverse "down", default is [0, 1, 0, 0]
        organ_down_quat: Euler angles in [x, y, z] format for organ "down", default is [0, 0, -90]

    Returns:
        Euler angles in organ coordinate system [x, y, z] in degrees
    """
    if ov_down_quat is None:
        ov_down_quat = [0, 1, 0, 0]
    if organ_down_quat is None:
        organ_down_quat = [0, 0, -90]

    # Step 1: Convert Omniverse quaternion to rotation matrix
    # Reformat from [w,x,y,z] to [x,y,z,w] for scipy
    ov_quat_scipy = [ov_quat[1], ov_quat[2], ov_quat[3], ov_quat[0]]
    ov_rot = Rotation.from_quat(ov_quat_scipy)

    # Step 2: Create reference orientations
    # Define "down" in Omniverse using a quaternion [0,1,0,0]
    ov_down_quat = [ov_down_quat[1], ov_down_quat[2], ov_down_quat[3], ov_down_quat[0]]  # [x,y,z,w] format for scipy
    ov_down_rot = Rotation.from_quat(ov_down_quat)

    # Define "down" in organ coordinates using Euler angles [0,0,-90]
    organ_down_rot = Rotation.from_euler("xyz", organ_down_quat, degrees=True)

    # Step 3: Define the relative rotation needed for the reference
    # This calculates how to rotate from Omniverse "down" to organ "down"
    reference_mapping = organ_down_rot * ov_down_rot.inv()

    # Step 4: Apply the same relative rotation to the current orientation
    # This preserves the relative angle from "down" in both systems
    result_rot = reference_mapping * ov_rot

    # Step 5: Convert to Euler angles
    organ_euler = result_rot.as_euler("xyz", degrees=True)

    return organ_euler


def get_probe_pos_ori(env, transform_matrix, scale: float = 1000.0, log=False):
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
        log: If True, print the transformed position and orientation values for debugging

    Returns:
        transformed_position: numpy array of shape (3,) in organ coordinate system (millimeters)
        transformed_orientation: numpy array of shape (3,) with Euler angles in organ system (degrees)
    """
    # Get probe data from the end effector frame
    probe_data = env.unwrapped.scene["ee_frame"].data

    # Get probe position and remove environment origins offset
    # Raw tensor has shape (batch_size=1, num_envs=1, 3)
    probe_pos = probe_data.target_pos_w - env.unwrapped.scene.env_origins

    # Remove batch dimensions to get a simple position vector of shape (3,)
    probe_pos_flat = probe_pos.squeeze(0).squeeze(0)

    # Scale position from meters to millimeters
    probe_pos_flat = scale_points(probe_pos_flat, scale=scale)

    # Convert to homogeneous coordinates by adding a 1 as the 4th component
    # This allows for the application of the 4x4 transformation matrix
    pos_homogeneous = torch.cat([probe_pos_flat, torch.tensor([1.0], device=probe_pos.device)], dim=-1)

    # Apply 4x4 transformation matrix to convert to organ coordinate system
    # This handles both rotation and translation in one operation
    transformed_pos = transform_matrix.to(probe_pos.device) @ pos_homogeneous

    # Extract only the spatial coordinates (x, y, z), discarding homogeneous component
    transformed_pos = transformed_pos[:3]

    # Get probe orientation as quaternion [w, x, y, z] and remove batch dimensions
    # Raw tensor has shape (batch_size=1, num_envs=1, 4)
    probe_quat = probe_data.target_quat_w.squeeze(0).squeeze(0)  # Shape (4,)

    transformed_ori = ov_to_nifti_orientation(probe_quat.cpu().numpy())

    # Optional logging for debugging
    if log:
        print(f"Raw position (Isaac Sim, meters): {probe_pos_flat}")
        print(f"Transformed position (organ, mm): {transformed_pos}")
        print(f"Raw orientation (Isaac Sim, quat): {probe_quat}")
        print(f"Transformed orientation (organ, Euler): {transformed_ori}")

    # Return position as numpy array and orientation as Euler angles
    return transformed_pos.cpu().numpy(), transformed_ori


def load_onnx_model(model_path):
    """Load the ACT ONNX model."""
    providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    print(f"Using providers: {providers}")
    # Create an InferenceSession with GPU support
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"session using: {session.get_providers()}")
    return session


class KeyboardHandler:
    def __init__(self):
        self.reset_flag = False
        self.listener = None

    def on_press(self, key):
        """Callback function to handle key press events."""
        try:
            if key.char == "r":
                self.reset_flag = True
        except AttributeError:
            pass

    def start_keyboard_listener(self):
        """Start a separate thread to listen for keyboard events."""
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def stop_keyboard_listener(self):
        """Stop the keyboard listener."""
        if self.listener:
            self.listener.stop()

    def is_reset_requested(self):
        """Check if a reset has been requested."""
        if self.reset_flag:
            self.reset_flag = False  # Reset the flag after checking
            return True
        return False
