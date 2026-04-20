#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record HDF5 demos with Meta Quest controllers")
parser.add_argument("--task", type=str, default="Isaac-Spread-Tablecloth-G129-Inspire-Teleop", help="task name")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments")
parser.add_argument("--step_hz", type=int, default=30, help="environment stepping rate in Hz")
parser.add_argument("--num_demos", type=int, default=1, help="number of demos to record (0 = infinite)")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Import Pinocchio before AppLauncher for Pink IK teleoperation.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/rlinf/demo.hdf5",
    help="HDF5 file path for saved demos",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, extra_kit_args = parser.parse_known_args()
sys.argv += extra_kit_args

if args_cli.enable_pinocchio:
    with contextlib.suppress(Exception):
        import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import omni.ui as ui
import torch
import isaaclab.utils.math as PoseUtils
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.openxr.xr_cfg import XrCfg, remove_camera_configs
from isaaclab.devices.retargeter_base import RetargeterBase
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode
from isaaclab_mimic.ui.instruction_display import InstructionDisplay
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from simulation.tasks import spread_tablecloth  # noqa: F401


# ---------------------------------------------------------------------------
# Wrist retargeting offset: (0, -75, 90) euler → xyzw quaternion.
# Same offset used by the upstream G1TriHandUpperBodyMotionControllerGripperRetargeter
# for converting from OpenXR controller frame to G1 wrist frame.
# ---------------------------------------------------------------------------
_WRIST_OFFSET_QUAT = torch.tensor([-0.4619, 0.5358, 0.4619, 0.5358], dtype=torch.float32)

# Inspire hand grasp amplitudes (radians) – tune as needed.
_FINGER_PROXIMAL_CLOSE = 1.5
_FINGER_INTERMEDIATE_CLOSE = 1.5
_THUMB_YAW_CLOSE = 0.8
_THUMB_PITCH_CLOSE = 1.0
_THUMB_INTERMEDIATE_CLOSE = 0.8
_THUMB_DISTAL_CLOSE = 0.8

_DEFAULT_CONTROLLER_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class RateLimiter:
    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            self.last_time = time.time()


class KeyboardControls:
    """Keyboard controls inside Isaac Sim window.

    `B`: start recording
    `S`: save current demo
    `R`: reset current demo
    """

    def __init__(self):
        import carb.input
        import omni.appwindow

        self._carb_input = carb.input
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._start_pressed = False
        self._save_pressed = False
        self._reset_pressed = False
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            if event.input == self._carb_input.KeyboardInput.B:
                self._start_pressed = True
            elif event.input == self._carb_input.KeyboardInput.S:
                self._save_pressed = True
            elif event.input == self._carb_input.KeyboardInput.R:
                self._reset_pressed = True
        return True

    def consume_start(self) -> bool:
        if self._start_pressed:
            self._start_pressed = False
            return True
        return False

    def consume_save(self) -> bool:
        if self._save_pressed:
            self._save_pressed = False
            return True
        return False

    def consume_reset(self) -> bool:
        if self._reset_pressed:
            self._reset_pressed = False
            return True
        return False

    def close(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)


def setup_output_directories(dataset_file: str) -> tuple[str, str]:
    output_filepath = os.path.abspath(dataset_file)
    output_dir = os.path.dirname(output_filepath)
    output_file_name = os.path.basename(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir, output_file_name


@dataclass
class ControllerButtonState:
    left_primary_pressed: bool = False
    right_primary_pressed: bool = False
    right_secondary_pressed: bool = False


def _clone_xr_cfg(xr_cfg: XrCfg | None) -> XrCfg:
    if xr_cfg is None:
        return XrCfg()
    default_cfg = XrCfg()
    return XrCfg(
        anchor_pos=tuple(getattr(xr_cfg, "anchor_pos", default_cfg.anchor_pos)),
        anchor_rot=tuple(getattr(xr_cfg, "anchor_rot", default_cfg.anchor_rot)),
        anchor_prim_path=getattr(xr_cfg, "anchor_prim_path", default_cfg.anchor_prim_path),
        anchor_rotation_mode=getattr(xr_cfg, "anchor_rotation_mode", default_cfg.anchor_rotation_mode),
        anchor_rotation_smoothing_time=getattr(
            xr_cfg, "anchor_rotation_smoothing_time", default_cfg.anchor_rotation_smoothing_time
        ),
        anchor_rotation_custom_func=getattr(
            xr_cfg, "anchor_rotation_custom_func", default_cfg.anchor_rotation_custom_func
        ),
        near_plane=float(getattr(xr_cfg, "near_plane", default_cfg.near_plane)),
        fixed_anchor_height=bool(getattr(xr_cfg, "fixed_anchor_height", default_cfg.fixed_anchor_height)),
    )


def _get_controller_sample(raw_data: object, target: DeviceBase.TrackingTarget) -> np.ndarray | None:
    if not isinstance(raw_data, dict):
        return None
    sample = raw_data.get(target)
    if isinstance(sample, np.ndarray) and sample.size > 0:
        return sample
    return None


def _read_controller_input(controller_data: np.ndarray | None, input_index: int, default: float = 0.0) -> float:
    if controller_data is None:
        return default
    input_row_index = int(DeviceBase.MotionControllerDataRowIndex.INPUTS.value)
    if len(controller_data) <= input_row_index:
        return default
    inputs = controller_data[input_row_index]
    if len(inputs) <= input_index:
        return default
    return float(inputs[input_index])


def _read_controller_pose(controller_data: np.ndarray | None, default_pose: np.ndarray) -> np.ndarray:
    if controller_data is None:
        return default_pose.copy()
    pose_row_index = int(DeviceBase.MotionControllerDataRowIndex.POSE.value)
    if len(controller_data) <= pose_row_index:
        return default_pose.copy()
    pose = np.asarray(controller_data[pose_row_index], dtype=np.float32)
    if pose.shape != (7,):
        return default_pose.copy()
    return pose


def _read_controller_click(controller_data: np.ndarray | None, button_index: int) -> bool:
    return _read_controller_input(controller_data, button_index, 0.0) > 0.5


def _retarget_controller_pose(controller_pose: np.ndarray, sim_device: torch.device) -> torch.Tensor:
    """Convert OpenXR controller pose to G1 wrist target pose for PINK IK.

    Uses the same rotation offset as the upstream
    G1TriHandUpperBodyMotionControllerGripperRetargeter: (0, -75, 90) euler.
    Output format: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w].
    """
    wrist_pos = torch.tensor(controller_pose[:3], dtype=torch.float32).unsqueeze(0)
    wrist_quat = torch.tensor(controller_pose[3:], dtype=torch.float32).unsqueeze(0)

    openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
    offset_pose = PoseUtils.make_pose(
        torch.zeros_like(wrist_pos),
        PoseUtils.matrix_from_quat(_WRIST_OFFSET_QUAT.unsqueeze(0)),
    )
    result_pose = PoseUtils.pose_in_A_to_pose_in_B(offset_pose, openxr_pose)
    pos_out, rot_out = PoseUtils.unmake_pose(result_pose)
    quat_out = PoseUtils.quat_from_matrix(rot_out)
    return torch.cat([pos_out[0], quat_out[0]], dim=0).to(device=sim_device)


def _build_inspire_hand_action(
    left_trigger: float, right_trigger: float, device: torch.device
) -> torch.Tensor:
    """Build 24-D Inspire hand joint targets from two trigger values.

    Joint order matches INSPIRE_HAND_JOINT_NAMES in the env cfg.
    """
    lt = max(0.0, min(1.0, left_trigger))
    rt = max(0.0, min(1.0, right_trigger))

    fp = _FINGER_PROXIMAL_CLOSE
    fi = _FINGER_INTERMEDIATE_CLOSE
    ty = _THUMB_YAW_CLOSE
    tp = _THUMB_PITCH_CLOSE
    ti = _THUMB_INTERMEDIATE_CLOSE
    td = _THUMB_DISTAL_CLOSE

    return torch.tensor(
        [
            # Left proximal (index, middle, pinky, ring, thumb_yaw)
            lt * fp, lt * fp, lt * fp, lt * fp, lt * ty,
            # Right proximal
            rt * fp, rt * fp, rt * fp, rt * fp, rt * ty,
            # Left intermediate (index, middle, pinky, ring, thumb_pitch)
            lt * fi, lt * fi, lt * fi, lt * fi, lt * tp,
            # Right intermediate
            rt * fi, rt * fi, rt * fi, rt * fi, rt * tp,
            # Thumb intermediate/distal (L, R, L, R)
            lt * ti, rt * ti, lt * td, rt * td,
        ],
        dtype=torch.float32,
        device=device,
    )


class DirectOpenXRPinkIKDevice:
    """Read Quest controller poses directly and build PINK IK actions for G1 + Inspire hand.

    Action tensor layout (38-D):
        [left_wrist_pos(3), left_wrist_quat_xyzw(4),
         right_wrist_pos(3), right_wrist_quat_xyzw(4),
         inspire_hand_joints(24)]
    """

    def __init__(self, xr_cfg: XrCfg | None, sim_device: str):
        self._sim_device = torch.device(sim_device)
        self._button_state = ControllerButtonState()
        self._left_pose_seen = False
        self._right_pose_seen = False
        self._prev_left_pose = _DEFAULT_CONTROLLER_POSE.copy()
        self._prev_right_pose = _DEFAULT_CONTROLLER_POSE.copy()
        self._prev_left_trigger = 0.0
        self._prev_right_trigger = 0.0
        self._last_raw_data: dict | None = None
        self._logged_left_inputs = False
        self._logged_right_inputs = False

        import carb.settings

        settings = carb.settings.get_settings()
        settings.set("/persistent/xr/openxr/disableInputBindings", False)

        devices_cfg = DevicesCfg(
            devices={
                "direct_openxr": OpenXRDeviceCfg(
                    sim_device=sim_device,
                    xr_cfg=_clone_xr_cfg(xr_cfg),
                )
            }
        )
        self._device = create_teleop_device("direct_openxr", devices_cfg.devices, {})
        self._device._required_features.add(RetargeterBase.Requirement.MOTION_CONTROLLER)

        unbind_fn = getattr(self._device, "_unbind_all_buttons", None)
        if callable(unbind_fn):
            unbind_fn()

        self._diag_frame = 0

    def _log_available_inputs(self) -> None:
        xr_core = getattr(self._device, "_xr_core", None)
        if xr_core is None:
            return
        if not self._logged_left_inputs:
            left_device = xr_core.get_input_device("/user/hand/left")
            if left_device is not None:
                print(f"[XR] left input names: {sorted(str(n) for n in (left_device.get_input_names() or ()))}")
                self._logged_left_inputs = True
        if not self._logged_right_inputs:
            right_device = xr_core.get_input_device("/user/hand/right")
            if right_device is not None:
                print(f"[XR] right input names: {sorted(str(n) for n in (right_device.get_input_names() or ()))}")
                self._logged_right_inputs = True

    def reset(self) -> None:
        self._device.reset()
        self._button_state = ControllerButtonState()
        self._left_pose_seen = False
        self._right_pose_seen = False
        self._prev_left_pose = _DEFAULT_CONTROLLER_POSE.copy()
        self._prev_right_pose = _DEFAULT_CONTROLLER_POSE.copy()
        self._prev_left_trigger = 0.0
        self._prev_right_trigger = 0.0
        self._last_raw_data = None

    def close(self) -> None:
        unbind_fn = getattr(self._device, "_unbind_all_buttons", None)
        if callable(unbind_fn):
            unbind_fn()

    def controller_presence(self) -> tuple[bool, bool]:
        left_seen = _get_controller_sample(self._last_raw_data, DeviceBase.TrackingTarget.CONTROLLER_LEFT) is not None
        right_seen = _get_controller_sample(self._last_raw_data, DeviceBase.TrackingTarget.CONTROLLER_RIGHT) is not None
        return left_seen, right_seen

    def poll_buttons(
        self,
        on_start: Callable[[], None],
        on_success: Callable[[], None],
        on_reset: Callable[[], None],
    ) -> None:
        left_controller = _get_controller_sample(self._last_raw_data, DeviceBase.TrackingTarget.CONTROLLER_LEFT)
        right_controller = _get_controller_sample(self._last_raw_data, DeviceBase.TrackingTarget.CONTROLLER_RIGHT)

        button_0 = int(DeviceBase.MotionControllerInputIndex.BUTTON_0.value)
        button_1 = int(DeviceBase.MotionControllerInputIndex.BUTTON_1.value)

        left_primary = _read_controller_click(left_controller, button_0)
        right_primary = _read_controller_click(right_controller, button_0)
        right_secondary = _read_controller_click(right_controller, button_1)

        if left_primary and not self._button_state.left_primary_pressed:
            on_start()
        if right_primary and not self._button_state.right_primary_pressed:
            on_success()
        if right_secondary and not self._button_state.right_secondary_pressed:
            on_reset()

        self._button_state.left_primary_pressed = left_primary
        self._button_state.right_primary_pressed = right_primary
        self._button_state.right_secondary_pressed = right_secondary

    def advance(self) -> torch.Tensor | None:
        raw_data = self._device.advance()
        self._last_raw_data = raw_data if isinstance(raw_data, dict) else None
        self._log_available_inputs()

        left_controller = _get_controller_sample(raw_data, DeviceBase.TrackingTarget.CONTROLLER_LEFT)
        right_controller = _get_controller_sample(raw_data, DeviceBase.TrackingTarget.CONTROLLER_RIGHT)

        self._diag_frame += 1
        if self._diag_frame <= 5 or (self._diag_frame % 300 == 0):
            keys = list(raw_data.keys()) if isinstance(raw_data, dict) else type(raw_data).__name__
            lp = _read_controller_pose(left_controller, _DEFAULT_CONTROLLER_POSE) if left_controller is not None else None
            rp = _read_controller_pose(right_controller, _DEFAULT_CONTROLLER_POSE) if right_controller is not None else None
            lt = _read_controller_input(left_controller, int(DeviceBase.MotionControllerInputIndex.TRIGGER.value)) if left_controller is not None else None
            rt = _read_controller_input(right_controller, int(DeviceBase.MotionControllerInputIndex.TRIGGER.value)) if right_controller is not None else None
            print(f"[XR-diag] frame={self._diag_frame} keys={keys} L_pose={lp} R_pose={rp} L_trig={lt} R_trig={rt}")

        if left_controller is not None:
            self._prev_left_pose = _read_controller_pose(left_controller, self._prev_left_pose)
            self._prev_left_trigger = _read_controller_input(
                left_controller,
                int(DeviceBase.MotionControllerInputIndex.TRIGGER.value),
                self._prev_left_trigger,
            )
            self._left_pose_seen = True
        if right_controller is not None:
            self._prev_right_pose = _read_controller_pose(right_controller, self._prev_right_pose)
            self._prev_right_trigger = _read_controller_input(
                right_controller,
                int(DeviceBase.MotionControllerInputIndex.TRIGGER.value),
                self._prev_right_trigger,
            )
            self._right_pose_seen = True

        if not (self._left_pose_seen and self._right_pose_seen):
            return None

        left_wrist = _retarget_controller_pose(self._prev_left_pose, self._sim_device)
        right_wrist = _retarget_controller_pose(self._prev_right_pose, self._sim_device)
        hand_action = _build_inspire_hand_action(
            self._prev_left_trigger, self._prev_right_trigger, self._sim_device
        )
        return torch.cat([left_wrist, right_wrist, hand_action], dim=0)


def main() -> None:
    rate_limiter = None if args_cli.xr else RateLimiter(args_cli.step_hz)
    output_dir, output_file_name = setup_output_directories(args_cli.dataset_file)

    num_envs = int(getattr(args_cli, "num_envs", 1) or 1)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.seed = args_cli.seed

    if args_cli.xr:
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
            if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "camera_images"):
                env_cfg.observations.camera_images = None
        env_cfg.sim.render.antialiasing_mode = "DLSS"

        print("[XR] XR mode active — pass --/persistent/xr/... CLI args to tune rendering")

    if args_cli.enable_cameras and hasattr(env_cfg, "observations"):
        obs_cfg = env_cfg.observations
        if hasattr(obs_cfg, "camera_images") and obs_cfg.camera_images is not None:
            for name in ("front_camera", "left_wrist_camera", "right_wrist_camera"):
                if hasattr(obs_cfg.camera_images, name):
                    setattr(obs_cfg.policy, name, getattr(obs_cfg.camera_images, name))

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.observations.policy.concatenate_terms = False

    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = {}

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.seed(args_cli.seed)

    start_requested = False
    save_requested = False
    reset_requested = False
    teleop_active = not args_cli.xr
    recording_active = False
    current_demo_count = 0

    def request_start():
        nonlocal start_requested
        start_requested = True
        print("[XR] Start requested")

    def request_save():
        nonlocal save_requested
        save_requested = True
        print("[XR] Save requested")

    def request_reset():
        nonlocal reset_requested
        reset_requested = True
        print("[XR] Reset requested")

    keyboard_controls = KeyboardControls()

    use_isaac_teleop = hasattr(env_cfg, "isaac_teleop") and env_cfg.isaac_teleop is not None
    isaac_teleop_device = None

    if use_isaac_teleop:
        from isaaclab_teleop import create_isaac_teleop_device

        teleop_callbacks = {
            "START": request_start,
            "STOP": request_save,
            "RESET": request_reset,
        }
        isaac_teleop_device = create_isaac_teleop_device(
            env_cfg.isaac_teleop,
            sim_device=args_cli.device,
            callbacks=teleop_callbacks,
        )
        teleop_interface = None
        print("[TELEOP] Using IsaacTeleop pipeline (AVP hand tracking)")
    else:
        teleop_interface = DirectOpenXRPinkIKDevice(getattr(env_cfg, "xr", None), sim_device=args_cli.device)
        print("[TELEOP] Using DirectOpenXR pipeline (Quest controllers)")

    target = args_cli.num_demos if args_cli.num_demos > 0 else "\u221e"
    label_text = f"Ready. Left X or keyboard B to start demo 1/{target}"
    instruction_display = InstructionDisplay(xr=args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Recording Status")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    def _run_loop():
        nonlocal start_requested, save_requested, reset_requested
        nonlocal teleop_active, recording_active, current_demo_count

        last_no_action_log_time = 0.0
        first_action_logged = False
        zero_action_logged = False

        def reset_teleop():
            nonlocal first_action_logged, zero_action_logged
            if isaac_teleop_device is not None:
                isaac_teleop_device.reset()
            if teleop_interface is not None:
                teleop_interface.reset()
            first_action_logged = False
            zero_action_logged = False

        env.sim.reset()
        env.reset()
        reset_teleop()
        env.recorder_manager.reset()

        robot = env.unwrapped.scene["robot"]
        print(f"[DIAG] Robot joint names ({len(robot.data.joint_names)}): {robot.data.joint_names}")
        action_term = list(env.unwrapped.action_manager._terms.values())[0]
        print(f"[DIAG] Action dim={action_term.action_dim}, controlled_joints={getattr(action_term, '_controlled_joint_names', 'N/A')}")
        ik_ctrl = getattr(action_term, '_ik_controllers', None)
        if ik_ctrl:
            pink_cfg = ik_ctrl[0].pink_configuration
            print(f"[DIAG] Pinocchio model frames: {[f.name for f in pink_cfg.model.frames]}")

        print(f"Ready \u2014 target {target} demos | left X=start right A=save right B=reset | keyboard: B=start S=save R=reset")

        while simulation_app.is_running():
            with torch.inference_mode():
                if keyboard_controls.consume_start():
                    request_start()
                if keyboard_controls.consume_save():
                    request_save()
                if keyboard_controls.consume_reset():
                    request_reset()

                if isaac_teleop_device is not None:
                    action = isaac_teleop_device.advance()
                else:
                    action = teleop_interface.advance()
                    teleop_interface.poll_buttons(request_start, request_save, request_reset)

                if start_requested:
                    start_requested = False
                    env.reset()
                    reset_teleop()
                    env.recorder_manager.reset()
                    recording_active = True
                    teleop_active = True
                    action = None
                    label_text = f"Recording demo {current_demo_count + 1}/{target}"
                    instruction_display.show_demo(label_text)

                if teleop_active and action is None:
                    now = time.time()
                    if now - last_no_action_log_time > 2.0:
                        if teleop_interface is not None:
                            left_seen, right_seen = teleop_interface.controller_presence()
                            print(f"[XR] Waiting for teleop action: left_controller={left_seen} right_controller={right_seen}")
                        else:
                            print("[XR] Waiting for IsaacTeleop action (start AR and connect headset)")
                        last_no_action_log_time = now

                if teleop_active and action is not None and not first_action_logged:
                    action_abs_max = float(action.abs().max().item())
                    action_norm = float(action.norm().item())
                    print(
                        "[XR] First action received: "
                        f"shape={tuple(action.shape)} max_abs={action_abs_max:.4f} norm={action_norm:.4f}"
                    )
                    first_action_logged = True
                    if action_abs_max < 1e-5:
                        print("[XR] Action tensor is present but near zero. Move the controllers to test motion.")
                        zero_action_logged = True
                elif teleop_active and action is not None and not zero_action_logged:
                    action_abs_max = float(action.abs().max().item())
                    if action_abs_max < 1e-5:
                        print("[XR] Action tensor remains near zero. Controller pose/buttons may not be updating.")
                        zero_action_logged = True

                if teleop_active and action is not None:
                    expected_dim = env.action_space.shape[-1]
                    if action.shape[0] < expected_dim:
                        action = torch.cat(
                            [
                                action,
                                torch.zeros(expected_dim - action.shape[0], device=action.device, dtype=action.dtype),
                            ]
                        )
                    elif action.shape[0] > expected_dim:
                        action = action[:expected_dim]

                    env.step(action.repeat(env.num_envs, 1))
                else:
                    env.sim.render()

                if recording_active and save_requested:
                    save_requested = False
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])
                    current_demo_count += 1
                    recording_active = False
                    teleop_active = False
                    print(f"Demo {current_demo_count} saved")

                    if args_cli.num_demos > 0 and current_demo_count >= args_cli.num_demos:
                        break

                    env.reset()
                    reset_teleop()
                    env.recorder_manager.reset()
                    label_text = f"Ready. Left X or keyboard B to start demo {current_demo_count + 1}/{target}"
                    instruction_display.show_demo(label_text)
                else:
                    save_requested = False

                if reset_requested:
                    reset_requested = False
                    env.reset()
                    reset_teleop()
                    env.recorder_manager.reset()
                    recording_active = False
                    teleop_active = False
                    label_text = f"Ready. Left X or keyboard B to start demo {current_demo_count + 1}/{target}"
                    instruction_display.show_demo(label_text)

                if rate_limiter:
                    rate_limiter.sleep(env)

    try:
        if isaac_teleop_device is not None:
            with isaac_teleop_device:
                _run_loop()
        else:
            _run_loop()
    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted by user")
    finally:
        keyboard_controls.close()
        if teleop_interface is not None:
            teleop_interface.close()
        env.close()

    print(f"Done \u2014 {current_demo_count} demos \u2192 {os.path.abspath(args_cli.dataset_file)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
