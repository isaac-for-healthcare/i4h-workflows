# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Keyboard to 23D G1 WBC action adapter."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import carb
import numpy as np
import omni
import torch
from scipy.spatial.transform import Rotation


class ControlMode(Enum):
    """Control modes for the keyboard adapter."""

    RIGHT_HAND = "right_hand"
    LEFT_HAND = "left_hand"
    BOTH_HANDS = "both_hands"
    BASE_NAV = "base_nav"
    TORSO = "torso"
    HEIGHT = "height"


@dataclass
class KeyboardTo23DConfig:
    """Configuration for keyboard to 23D adapter."""

    pos_sensitivity: float = 0.01
    rot_sensitivity: float = 0.05
    vel_sensitivity: float = 0.05
    height_sensitivity: float = 0.01
    default_base_height: float = 0.75
    default_left_hand_pos: list[float] | None = None
    default_right_hand_pos: list[float] | None = None
    max_velocity: float = 0.4
    min_base_height: float = 0.65
    max_base_height: float = 0.85
    max_torso_angle: float = 0.3
    hand_pos_x_min: float = -0.1
    hand_pos_x_max: float = 0.5
    hand_pos_y_min: float = -0.5
    hand_pos_y_max: float = 0.5
    hand_pos_z_min: float = -0.3  # relative to pelvis frame
    hand_pos_z_max: float = 0.8

    def __post_init__(self):
        if self.default_left_hand_pos is None:
            self.default_left_hand_pos = [0.15, 0.15, 0.2]
        if self.default_right_hand_pos is None:
            self.default_right_hand_pos = [0.15, -0.15, 0.2]


class KeyboardTo23DAdapter:
    """Adapter that converts keyboard input to 23D G1 WBC actions."""

    def __init__(self, cfg: KeyboardTo23DConfig, sim_device: str = "cpu"):
        self.cfg = cfg
        self._sim_device = sim_device
        self.mode = ControlMode.BOTH_HANDS
        self.paused = False
        self._state = {
            "left_hand": 0.0,
            "right_hand": 0.0,
            "left_wrist_pos": np.array(cfg.default_left_hand_pos, dtype=np.float32),
            "left_wrist_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "right_wrist_pos": np.array(cfg.default_right_hand_pos, dtype=np.float32),
            "right_wrist_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "navigate_cmd": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "base_height": cfg.default_base_height,
            "torso_rpy": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
        self.locked_dims = set()
        self._additional_callbacks: dict[str, Callable] = {}
        self._setup_keyboard()
        self._print_instructions()

    def _setup_keyboard(self):
        import weakref

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def __del__(self):
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)

    def reset(self):
        self._state = {
            "left_hand": 0.0,
            "right_hand": 0.0,
            "left_wrist_pos": np.array(self.cfg.default_left_hand_pos, dtype=np.float32),
            "left_wrist_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "right_wrist_pos": np.array(self.cfg.default_right_hand_pos, dtype=np.float32),
            "right_wrist_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "navigate_cmd": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "base_height": self.cfg.default_base_height,
            "torso_rpy": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
        self.locked_dims.clear()
        self.paused = False
        print("[Adapter] Reset to default state")
        self._print_status()

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        action = np.zeros(23, dtype=np.float32)
        action[0] = self._state["left_hand"]
        action[1] = self._state["right_hand"]
        action[2:5] = self._state["left_wrist_pos"]
        action[5:9] = self._state["left_wrist_quat"]
        action[9:12] = self._state["right_wrist_pos"]
        action[12:16] = self._state["right_wrist_quat"]
        action[16:19] = self._state["navigate_cmd"] if not self.paused else np.zeros(3, dtype=np.float32)
        action[19] = self._state["base_height"]
        action[20:23] = self._state["torso_rpy"]
        if self.paused:
            action[16:19] = np.array([0.0, 0.0, 0.0])
        return torch.tensor(action, dtype=torch.float32, device=self._sim_device)

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name
            if key_name == "KEY_0":
                self.mode = ControlMode.BOTH_HANDS
                print("[Mode] Switched to BOTH HANDS")
                self._print_status()
            elif key_name == "KEY_1":
                self.mode = ControlMode.RIGHT_HAND
                print("[Mode] Switched to RIGHT HAND")
                self._print_status()
            elif key_name == "KEY_2":
                self.mode = ControlMode.LEFT_HAND
                print("[Mode] Switched to LEFT HAND")
                self._print_status()
            elif key_name == "KEY_3":
                self.mode = ControlMode.BASE_NAV
                print("[Mode] Switched to BASE NAVIGATION")
                self._print_status()
            elif key_name == "KEY_4":
                self.mode = ControlMode.TORSO
                print("[Mode] Switched to TORSO ORIENTATION")
                self._print_status()
            elif key_name == "KEY_5":
                self.mode = ControlMode.HEIGHT
                print("[Mode] Switched to HEIGHT CONTROL")
                self._print_status()
            elif key_name == "R":
                self.reset()
            elif key_name == "L":
                mode_key = self.mode.value
                if mode_key in self.locked_dims:
                    self.locked_dims.remove(mode_key)
                    print(f"[Lock] Unlocked {self.mode.value}")
                else:
                    self.locked_dims.add(mode_key)
                    print(f"[Lock] Locked {self.mode.value}")
            elif key_name == "SPACE":
                self.paused = not self.paused
                print(f"[Pause] {'PAUSED' if self.paused else 'RESUMED'}")
            elif self.mode.value not in self.locked_dims and not self.paused:
                self._process_mode_input(key_name)
            if key_name in self._additional_callbacks:
                self._additional_callbacks[key_name]()
        return True

    def _process_mode_input(self, key_name: str):
        if self.mode == ControlMode.BOTH_HANDS:
            self._process_both_hands_input(key_name)
        elif self.mode == ControlMode.RIGHT_HAND:
            self._process_hand_input(key_name, hand="right")
        elif self.mode == ControlMode.LEFT_HAND:
            self._process_hand_input(key_name, hand="left")
        elif self.mode == ControlMode.BASE_NAV:
            self._process_base_nav_input(key_name)
        elif self.mode == ControlMode.TORSO:
            self._process_torso_input(key_name)
        elif self.mode == ControlMode.HEIGHT:
            self._process_height_input(key_name)

    def _process_both_hands_input(self, key_name: str):
        if key_name == "Q":
            self._state["left_wrist_pos"][2] = min(
                self._state["left_wrist_pos"][2] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_z_max,
            )
            self._state["right_wrist_pos"][2] = min(
                self._state["right_wrist_pos"][2] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_z_max,
            )
        elif key_name == "E":
            self._state["left_wrist_pos"][2] = max(
                self._state["left_wrist_pos"][2] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_z_min,
            )
            self._state["right_wrist_pos"][2] = max(
                self._state["right_wrist_pos"][2] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_z_min,
            )
        elif key_name == "A":
            self._state["left_wrist_pos"][1] = min(
                self._state["left_wrist_pos"][1] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_y_max,
            )
            self._state["right_wrist_pos"][1] = max(
                self._state["right_wrist_pos"][1] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_y_min,
            )
        elif key_name == "D":
            self._state["left_wrist_pos"][1] = max(
                self._state["left_wrist_pos"][1] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_y_min,
            )
            self._state["right_wrist_pos"][1] = min(
                self._state["right_wrist_pos"][1] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_y_max,
            )
        elif key_name == "W":
            self._state["left_wrist_pos"][0] = min(
                self._state["left_wrist_pos"][0] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_x_max,
            )
            self._state["right_wrist_pos"][0] = min(
                self._state["right_wrist_pos"][0] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_x_max,
            )
        elif key_name == "S":
            self._state["left_wrist_pos"][0] = max(
                self._state["left_wrist_pos"][0] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_x_min,
            )
            self._state["right_wrist_pos"][0] = max(
                self._state["right_wrist_pos"][0] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_x_min,
            )
        elif key_name == "Z":
            self._apply_symmetric_roll(self.cfg.rot_sensitivity)
        elif key_name == "X":
            self._apply_symmetric_roll(-self.cfg.rot_sensitivity)
        elif key_name == "T":
            self._apply_pitch(self.cfg.rot_sensitivity)
        elif key_name == "G":
            self._apply_pitch(-self.cfg.rot_sensitivity)
        elif key_name == "K":
            self._state["left_hand"] = 1.0
            self._state["right_hand"] = 1.0
            print("[Both Hands] Grippers: CLOSED (grip)")
        elif key_name == "J":
            self._state["left_hand"] = 0.0
            self._state["right_hand"] = 0.0
            print("[Both Hands] Grippers: OPEN (release)")

    def _apply_symmetric_roll(self, delta_roll: float):
        delta_rpy_left = np.array([delta_roll, 0.0, 0.0])
        delta_rpy_right = np.array([-delta_roll, 0.0, 0.0])
        self._apply_quaternion_delta("left_wrist_quat", delta_rpy_left)
        self._apply_quaternion_delta("right_wrist_quat", delta_rpy_right)
        print(f"[Both Hands] Symmetric Roll (ΔL={delta_roll:.3f}, ΔR={-delta_roll:.3f})")

    def _apply_pitch(self, delta_pitch: float):
        delta_rpy = np.array([0.0, delta_pitch, 0.0])
        self._apply_quaternion_delta("left_wrist_quat", delta_rpy)
        self._apply_quaternion_delta("right_wrist_quat", delta_rpy)
        print(f"[Both Hands] Pitch: {delta_pitch:+.3f} rad")

    def _apply_quaternion_delta(self, key: str, delta_rpy: np.ndarray):
        delta_quat = Rotation.from_euler("xyz", delta_rpy).as_quat()
        delta_quat_wxyz = np.array([delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]])
        current_quat = self._state[key]
        new_quat = self._quaternion_multiply(current_quat, delta_quat_wxyz)
        norm = np.linalg.norm(new_quat)
        if norm > 1e-6:
            self._state[key] = new_quat / norm
        else:
            self._state[key] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            print(f"[Warning] {key} quaternion reset to identity due to normalization issue")

    def _process_hand_input(self, key_name: str, hand: str):
        pos_key = f"{hand}_wrist_pos"
        quat_key = f"{hand}_wrist_quat"
        gripper_key = f"{hand}_hand"
        if key_name == "W":
            self._state[pos_key][0] = min(self._state[pos_key][0] + self.cfg.pos_sensitivity, self.cfg.hand_pos_x_max)
        elif key_name == "S":
            self._state[pos_key][0] = max(self._state[pos_key][0] - self.cfg.pos_sensitivity, self.cfg.hand_pos_x_min)
        elif key_name == "A":
            self._state[pos_key][1] = min(self._state[pos_key][1] + self.cfg.pos_sensitivity, self.cfg.hand_pos_y_max)
        elif key_name == "D":
            self._state[pos_key][1] = max(self._state[pos_key][1] - self.cfg.pos_sensitivity, self.cfg.hand_pos_y_min)
        elif key_name == "Q":
            self._state[pos_key][2] = min(self._state[pos_key][2] + self.cfg.pos_sensitivity, self.cfg.hand_pos_z_max)
        elif key_name == "E":
            self._state[pos_key][2] = max(self._state[pos_key][2] - self.cfg.pos_sensitivity, self.cfg.hand_pos_z_min)
        elif key_name in ["Z", "X", "T", "G", "C", "V"]:
            delta_rpy = np.zeros(3)
            if key_name == "Z":
                delta_rpy[0] = self.cfg.rot_sensitivity
            elif key_name == "X":
                delta_rpy[0] = -self.cfg.rot_sensitivity
            elif key_name == "T":
                delta_rpy[1] = self.cfg.rot_sensitivity
            elif key_name == "G":
                delta_rpy[1] = -self.cfg.rot_sensitivity
            elif key_name == "C":
                delta_rpy[2] = self.cfg.rot_sensitivity
            elif key_name == "V":
                delta_rpy[2] = -self.cfg.rot_sensitivity
            delta_quat = Rotation.from_euler("xyz", delta_rpy).as_quat()
            delta_quat_wxyz = np.array([delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]])
            current_quat = self._state[quat_key]
            new_quat = self._quaternion_multiply(current_quat, delta_quat_wxyz)
            norm = np.linalg.norm(new_quat)
            if norm > 1e-6:
                self._state[quat_key] = new_quat / norm
            else:
                self._state[quat_key] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                print(f"[Warning] {hand.capitalize()} hand quaternion reset to identity")
        elif key_name == "K":
            self._state[gripper_key] = 1.0
            print(f"[{hand.capitalize()} Hand] Gripper: CLOSED (grip)")
        elif key_name == "J":
            self._state[gripper_key] = 0.0
            print(f"[{hand.capitalize()} Hand] Gripper: OPEN (release)")

    def _process_base_nav_input(self, key_name: str):
        if key_name == "W":
            self._state["navigate_cmd"][0] = min(
                self._state["navigate_cmd"][0] + self.cfg.vel_sensitivity,
                self.cfg.max_velocity,
            )
        elif key_name == "S":
            self._state["navigate_cmd"][0] = max(
                self._state["navigate_cmd"][0] - self.cfg.vel_sensitivity,
                -self.cfg.max_velocity,
            )
        elif key_name == "A":
            self._state["navigate_cmd"][1] = min(
                self._state["navigate_cmd"][1] + self.cfg.vel_sensitivity,
                self.cfg.max_velocity,
            )
        elif key_name == "D":
            self._state["navigate_cmd"][1] = max(
                self._state["navigate_cmd"][1] - self.cfg.vel_sensitivity,
                -self.cfg.max_velocity,
            )
        elif key_name == "Q":
            self._state["navigate_cmd"][2] = min(self._state["navigate_cmd"][2] + self.cfg.vel_sensitivity, 1.0)
        elif key_name == "E":
            self._state["navigate_cmd"][2] = max(self._state["navigate_cmd"][2] - self.cfg.vel_sensitivity, -1.0)
        elif key_name == "X":
            self._state["navigate_cmd"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            print("[Navigation] STOP - All velocities zeroed")

    def _process_torso_input(self, key_name: str):
        if key_name == "Z":
            self._state["torso_rpy"][0] = min(
                self._state["torso_rpy"][0] + self.cfg.rot_sensitivity,
                self.cfg.max_torso_angle,
            )
        elif key_name == "X":
            self._state["torso_rpy"][0] = max(
                self._state["torso_rpy"][0] - self.cfg.rot_sensitivity,
                -self.cfg.max_torso_angle,
            )
        elif key_name == "T":
            self._state["torso_rpy"][1] = min(
                self._state["torso_rpy"][1] + self.cfg.rot_sensitivity,
                self.cfg.max_torso_angle,
            )
        elif key_name == "G":
            self._state["torso_rpy"][1] = max(
                self._state["torso_rpy"][1] - self.cfg.rot_sensitivity,
                -self.cfg.max_torso_angle,
            )
        elif key_name == "C":
            self._state["torso_rpy"][2] = min(
                self._state["torso_rpy"][2] + self.cfg.rot_sensitivity,
                self.cfg.max_torso_angle,
            )
        elif key_name == "V":
            self._state["torso_rpy"][2] = max(
                self._state["torso_rpy"][2] - self.cfg.rot_sensitivity,
                -self.cfg.max_torso_angle,
            )

    def _process_height_input(self, key_name: str):
        if key_name == "W":
            self._state["base_height"] = min(
                self._state["base_height"] + self.cfg.height_sensitivity,
                self.cfg.max_base_height,
            )
        elif key_name == "S":
            self._state["base_height"] = max(
                self._state["base_height"] - self.cfg.height_sensitivity,
                self.cfg.min_base_height,
            )

    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def _print_instructions(self):
        print("=" * 60)
        print("Keyboard to 23D Adapter - Controls")
        print("=" * 60)
        print("MODE SWITCHING:")
        print("  [0] Both Hands  [1] Right Hand  [2] Left Hand")
        print("  [3] Base Nav    [4] Torso       [5] Height")
        print("")
        print("BOTH HANDS MODE (Synchronized):")
        print("  Q/E: Both Up/Down   A/D: Apart/Together")
        print("  W/S: Both Forward/Back   Z/X: Symmetric Roll")
        print("  T/G: Both Pitch   K: Close Both Grippers   J: Open Both Grippers")
        print("")
        print("HAND MODE (Right/Left):")
        print("  W/S: Forward/Back   A/D: Left/Right   Q/E: Up/Down")
        print("  Z/X: Roll   T/G: Pitch   C/V: Yaw")
        print("  K: Close Gripper   J: Open Gripper")
        print("")
        print("BASE NAVIGATION MODE:")
        print("  W/S: Forward/Back   A/D: Strafe   Q/E: Rotate   X: STOP")
        print("")
        print("TORSO MODE:")
        print("  Z/X: Roll   T/G: Pitch   C/V: Yaw")
        print("")
        print("HEIGHT MODE:")
        print("  W/S: Raise/Lower")
        print("")
        print("SPECIAL KEYS:")
        print("  [R] Reset all   [L] Lock mode   [Space] Pause")
        print("")
        print("WORKSPACE LIMITS:")
        print(f"  Hand X: [{self.cfg.hand_pos_x_min:.2f}, {self.cfg.hand_pos_x_max:.2f}] m")
        print(f"  Hand Y: [{self.cfg.hand_pos_y_min:.2f}, {self.cfg.hand_pos_y_max:.2f}] m")
        print(f"  Hand Z: [{self.cfg.hand_pos_z_min:.2f}, {self.cfg.hand_pos_z_max:.2f}] m")
        print(f"  Base velocity: ±{self.cfg.max_velocity:.2f} m/s")
        print(f"  Base height: [{self.cfg.min_base_height:.2f}, {self.cfg.max_base_height:.2f}] m")
        print("=" * 60)

    def _print_status(self):
        lock_icon = "🔒" if self.mode.value in self.locked_dims else ""
        print("─" * 60)
        print(f"Current Mode: {self.mode.value.upper()} {lock_icon}")
        print("─" * 60)
