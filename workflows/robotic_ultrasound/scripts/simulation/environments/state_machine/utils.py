from dataclasses import dataclass
from enum import Enum

import onnxruntime as ort
import torch
from omni.isaac.lab.utils.math import compute_pose_error, quat_from_euler_xyz
from pynput import keyboard
import math


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
    # Define Euler angles in degrees for DOWN orientation (90 degrees around Y)
    DOWN_EULER_DEG = (180.0, 0.0, 180.0)
    # Convert to radians and then to quaternion
    DOWN: tuple[float, float, float, float] = tuple(
        quat_from_euler_xyz(
            torch.tensor(math.radians(DOWN_EULER_DEG[0])),  # X rotation (rad)
            torch.tensor(math.radians(DOWN_EULER_DEG[1])),  # Y rotation (rad)
            torch.tensor(math.radians(DOWN_EULER_DEG[2])),  # Z rotation (rad)
        ).tolist()
    )



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


def write_joint_positions(env, joint_positions):
    """Set the joint positions of the robot directly.

    Args:
        env (gymnasium Env): The Gymnasium environment containing the robot.
        joint_positions (list or numpy array): The desired joint positions. Should be a list or array of length equal to the number of joints.

    Returns:
        None
    """
    robot = env.unwrapped.scene["robot"]
    # Set the joint positions in the robot's data
    joint_vel = torch.zeros_like(joint_positions)
    robot.write_joint_state_to_sim(joint_positions, joint_vel)


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
