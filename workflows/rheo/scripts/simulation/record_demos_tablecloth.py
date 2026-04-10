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
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record HDF5 demos with Meta Quest controllers")
parser.add_argument("--task", type=str, default="Isaac-Spread-Tablecloth-G129-Dex3-Teleop", help="task name")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="motion_controllers",
    help="Name of the teleop device key in env_cfg.teleop_devices.",
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments")
parser.add_argument("--step_hz", type=int, default=30, help="environment stepping rate in Hz")
parser.add_argument("--num_demos", type=int, default=1, help="number of demos to record (0 = infinite)")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Import Pinocchio before AppLauncher (required for PINK IK inside the WBC action).",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/rlinf/demo.hdf5",
    help="HDF5 file path for saved demos",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    with contextlib.suppress(Exception):
        import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.ui as ui
import torch
from isaaclab.devices.openxr.xr_cfg import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode
from isaaclab_mimic.ui.instruction_display import InstructionDisplay
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from simulation.tasks import spread_tablecloth  # noqa: F401
from teleop_devices.motion_controllers import MotionControllersTeleopDevice


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


def setup_output_directories(dataset_file: str) -> tuple[str, str]:
    output_filepath = os.path.abspath(dataset_file)
    output_dir = os.path.dirname(output_filepath)
    output_file_name = os.path.basename(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir, output_file_name


def bind_motion_controller_buttons(
    teleop_interface,
    on_start: Callable[[], None],
    on_success: Callable[[], None],
    on_reset: Callable[[], None],
) -> None:
    """Bind Quest controller buttons to recorder commands.

    Left X: start recording.
    Right A: save current demo and reset.
    Right B: reset without saving.
    """

    unbind_fn = getattr(teleop_interface, "_unbind_all_buttons", None)
    if callable(unbind_fn):
        unbind_fn()

    bind_fn = getattr(teleop_interface, "_bind_button_press", None)
    if not callable(bind_fn):
        return

    bind_fn("/user/hand/left", "x", "record_demo_start", lambda _event: on_start())
    bind_fn("/user/hand/right", "a", "record_demo_success", lambda _event: on_success())
    bind_fn("/user/hand/right", "b", "record_demo_reset", lambda _event: on_reset())


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

    def request_save():
        nonlocal save_requested
        save_requested = True

    def request_reset():
        nonlocal reset_requested
        reset_requested = True

    def stop_cb():
        nonlocal teleop_active
        teleop_active = False

    teleop_callbacks = {"R": request_reset, "START": request_start, "STOP": stop_cb, "RESET": request_reset}
    device_name = args_cli.teleop_device

    if hasattr(env_cfg, "teleop_devices") and device_name in env_cfg.teleop_devices.devices:
        teleop_interface = create_teleop_device(device_name, env_cfg.teleop_devices.devices, teleop_callbacks)
    else:
        _td = MotionControllersTeleopDevice(sim_device=args_cli.device)
        teleop_interface = create_teleop_device(device_name, _td.get_teleop_device_cfg(embodiment=None).devices, teleop_callbacks)

    bind_motion_controller_buttons(teleop_interface, request_start, request_save, request_reset)

    def reset_teleop():
        teleop_interface.reset()
        for r in getattr(teleop_interface, "_retargeters", []):
            if hasattr(r, "reset"):
                r.reset()

    target = args_cli.num_demos if args_cli.num_demos > 0 else "∞"
    label_text = f"Ready. Press left X to start demo 1/{target}"
    instruction_display = InstructionDisplay(xr=args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Recording Status")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    env.sim.reset()
    env.reset()
    reset_teleop()
    env.recorder_manager.reset()

    print(f"Ready — target {target} demos | X=start A=save+reset B=reset")

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if start_requested:
                    start_requested = False
                    env.reset()
                    reset_teleop()
                    env.recorder_manager.reset()
                    recording_active = True
                    teleop_active = True
                    label_text = f"Recording demo {current_demo_count + 1}/{target}"
                    instruction_display.show_demo(label_text)

                action = teleop_interface.advance()

                if teleop_active:
                    expected_dim = env.action_space.shape[-1]
                    if action.shape[0] < expected_dim:
                        action = torch.cat(
                            [
                                action,
                                torch.zeros(expected_dim - action.shape[0], device=action.device, dtype=action.dtype),
                            ]
                        )
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
                    label_text = f"Ready. Press left X to start demo {current_demo_count + 1}/{target}"
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
                    label_text = f"Ready. Press left X to start demo {current_demo_count + 1}/{target}"
                    instruction_display.show_demo(label_text)

                if rate_limiter:
                    rate_limiter.sleep(env)

    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted by user")

    env.close()
    print(f"Done — {current_demo_count} demos → {os.path.abspath(args_cli.dataset_file)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
