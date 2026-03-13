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

import contextlib
import os
import time
import weakref

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli
from simulation.register_and_patch import register_workflow_assets, register_workflow_cli
from teleop_devices.keyboard_23d_adapter import KeyboardTo23DAdapter, KeyboardTo23DConfig

register_workflow_cli()

parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--teleop_device",
    dest="_root_teleop_device",
    type=str,
    choices=["keyboard", "motion_controllers"],
    default=None,
    help="Teleop device: 'keyboard' = 23-D keyboard; 'motion_controllers' = XR/Meta Quest controllers.",
)
parser.add_argument("--dataset_file", type=str, required=True, help="File path to export recorded demos.")
parser.add_argument("--step_hz", type=int, default=50, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=1, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Steps with task success to auto-save a demo.",
)
parser.add_argument("--pos_sensitivity", type=float, default=0.1, help="[keyboard] Position sensitivity.")
parser.add_argument("--rot_sensitivity", type=float, default=0.1, help="[keyboard] Rotation sensitivity.")
parser.add_argument("--vel_sensitivity", type=float, default=0.1, help="[keyboard] Velocity sensitivity.")
parser.add_argument("--height_sensitivity", type=float, default=0.01, help="[keyboard] Height sensitivity.")
parser.add_argument(
    "--enable_pinocchio", action="store_true", default=False, help="Enable Pinocchio (needed for XR/WBC)."
)

add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    with contextlib.suppress(Exception):
        import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.log
import omni.ui as ui
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode
from isaaclab_mimic.ui.instruction_display import InstructionDisplay
from teleop_devices.motion_controllers import MotionControllersTeleopDevice

register_workflow_assets()


class KeyboardControls:
    """In-sim keyboard controls via carb.input (B=start, S=save, R=reset).

    Works regardless of whether XR button binding succeeds — the same approach
    used by record_demos_meta_quest.py for trocar data collection.
    """

    def __init__(self):
        import carb.input
        import omni.appwindow

        self._carb_input = carb.input
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._start_pressed = False
        self._success_pressed = False
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
                self._success_pressed = True
            elif event.input == self._carb_input.KeyboardInput.R:
                self._reset_pressed = True
        return True

    def consume_start(self) -> bool:
        if self._start_pressed:
            self._start_pressed = False
            return True
        return False

    def consume_success(self) -> bool:
        if self._success_pressed:
            self._success_pressed = False
            return True
        return False

    def consume_reset(self) -> bool:
        if self._reset_pressed:
            self._reset_pressed = False
            return True
        return False

    def close(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)


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


def setup_output_directories(dataset_file: str | None = None) -> tuple[str, str]:
    path = os.path.abspath(dataset_file or args_cli.dataset_file)
    output_dir = os.path.dirname(path)
    output_file_name = os.path.basename(path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir, output_file_name


def create_environment_config(output_dir: str, output_file_name: str):
    """Build env config via Arena builder."""
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    env_cfg.is_finite_horizon = False
    env_cfg.seed = args_cli.seed if getattr(args_cli, "seed", None) is not None else 42
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    success_term = None
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "task_success"):
            success_term = env_cfg.terminations.task_success
            env_cfg.terminations.task_success = None
        elif hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    if success_term is not None:
        print(f"[INFO] Success condition: {success_term.func.__name__}")
    else:
        print("[WARNING] No success condition — auto-save disabled, use keyboard S to save")
    env_cfg.observations.policy.concatenate_terms = False
    return env_cfg, env_name, success_term


def main() -> None:
    teleop_device = getattr(args_cli, "teleop_device", None)
    if teleop_device is None:
        teleop_device = getattr(args_cli, "_root_teleop_device", None)
    if teleop_device is None:
        teleop_device = "keyboard"
    args_cli.teleop_device = teleop_device

    use_motion_controller = args_cli.teleop_device == "motion_controllers"
    use_xr = getattr(args_cli, "xr", False)
    seed = args_cli.seed if getattr(args_cli, "seed", None) is not None else 42
    rate_limiter = None if use_xr else RateLimiter(args_cli.step_hz)
    output_dir, output_file_name = setup_output_directories()

    env_cfg, env_name, success_term = create_environment_config(output_dir, output_file_name)

    # if use_xr:
    #     env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        env.seed(seed)
        print(f"[INFO] Environment created: {env_name}")
        print(f"[INFO] Action space shape: {env.action_space.shape}")
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    if not use_motion_controller:
        action_dim = env.action_space.shape[-1] if len(env.action_space.shape) > 1 else env.action_space.shape[0]
        if action_dim != 23:
            omni.log.error(f"Action space is {action_dim}D, expected 23D")
            env.close()
            simulation_app.close()
            return

    should_reset = False
    teleop_active = not use_xr
    recording_active = not use_xr
    current_demo_count = 0
    success_step_count = 0
    target = args_cli.num_demos if args_cli.num_demos > 0 else "∞"

    def reset_cb():
        nonlocal should_reset
        should_reset = True

    def start_cb():
        nonlocal teleop_active
        teleop_active = True

    def stop_cb():
        nonlocal teleop_active
        teleop_active = False

    if use_motion_controller:
        teleop_callbacks = {"R": reset_cb, "START": start_cb, "STOP": stop_cb, "RESET": reset_cb}
        device_name = "motion_controllers"
        if hasattr(env_cfg, "teleop_devices") and device_name in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(device_name, env_cfg.teleop_devices.devices, teleop_callbacks)
        else:
            _td = MotionControllersTeleopDevice(sim_device=args_cli.device)
            teleop_interface = create_teleop_device(
                device_name, _td.get_teleop_device_cfg(embodiment=None).devices, teleop_callbacks
            )
    else:
        config = KeyboardTo23DConfig(
            pos_sensitivity=args_cli.pos_sensitivity,
            rot_sensitivity=args_cli.rot_sensitivity,
            vel_sensitivity=args_cli.vel_sensitivity,
            height_sensitivity=args_cli.height_sensitivity,
        )
        teleop_interface = KeyboardTo23DAdapter(cfg=config, sim_device=args_cli.device)
        teleop_interface.add_callback("ENTER", reset_cb)

    keyboard = KeyboardControls() if use_xr else None

    def reset_teleop():
        teleop_interface.reset()
        for r in getattr(teleop_interface, "_retargeters", []):
            if hasattr(r, "reset"):
                r.reset()

    label_text = f"Ready. Press B to start demo 1/{target}"
    instruction_display = InstructionDisplay(use_xr)

    if use_xr:
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env.sim.reset()
    env.reset()
    reset_teleop()
    env.recorder_manager.reset()

    if not use_xr:
        window = EmptyWindow(env, "Recording Status")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            instruction_display.set_labels(None, demo_label)

    print("\n" + "=" * 60)
    print(f"Recording started! Target: {target} successful demos")
    if use_xr:
        print("Keyboard shortcuts (Isaac Sim window):")
        print("  B = start/resume teleoperation + recording")
        print("  S = manually save current demo")
        print("  R = reset environment")
    else:
        print("Keyboard shortcuts:")
        print("  ENTER = reset environment")
    print("=" * 60 + "\n")

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if keyboard and keyboard.consume_start():
                    env.reset()
                    reset_teleop()
                    env.recorder_manager.reset()
                    recording_active = True
                    teleop_active = True
                    success_step_count = 0
                    label_text = f"Recording demo {current_demo_count + 1}/{target}"
                    instruction_display.show_demo(label_text)
                    print(f"[B] Teleoperation started — recording demo {current_demo_count + 1}")

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

                if keyboard and keyboard.consume_success() and recording_active:
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])
                    current_demo_count += 1
                    recording_active = False
                    teleop_active = False
                    print(f"[S] Demo {current_demo_count} saved (manual)")
                    if args_cli.num_demos > 0 and current_demo_count >= args_cli.num_demos:
                        break
                    env.reset()
                    reset_teleop()
                    env.recorder_manager.reset()
                    label_text = f"Ready. Press B to start demo {current_demo_count + 1}/{target}"
                    instruction_display.show_demo(label_text)

                if recording_active and success_term is not None:
                    is_success = bool(success_term.func(env, **success_term.params)[0])
                    if is_success:
                        success_step_count += 1
                        if success_step_count % 20 == 1:
                            print(f"SUCCESS: {success_step_count}/{args_cli.num_success_steps} steps")
                        if success_step_count >= args_cli.num_success_steps:
                            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                            env.recorder_manager.set_success_to_episodes(
                                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                            )
                            env.recorder_manager.export_episodes([0])
                            current_demo_count += 1
                            recording_active = False
                            teleop_active = False
                            print(f"Demo {current_demo_count} auto-saved (task success)")
                            if args_cli.num_demos > 0 and current_demo_count >= args_cli.num_demos:
                                print(f"\n[INFO] Target reached: {current_demo_count} demos")
                                break
                            env.reset()
                            reset_teleop()
                            env.recorder_manager.reset()
                            success_step_count = 0
                            label_text = f"Ready. Press B to start demo {current_demo_count + 1}/{target}"
                            instruction_display.show_demo(label_text)
                    else:
                        success_step_count = 0

                if should_reset or (keyboard and keyboard.consume_reset()):
                    env.reset()
                    reset_teleop()
                    env.recorder_manager.reset()
                    recording_active = False
                    teleop_active = False
                    should_reset = False
                    success_step_count = 0
                    label_text = f"Ready. Press B to start demo {current_demo_count + 1}/{target}"
                    instruction_display.show_demo(label_text)
                    print("[R] Environment reset")

                if env.sim.is_stopped():
                    break

                if rate_limiter:
                    rate_limiter.sleep(env)

    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted")

    if keyboard:
        keyboard.close()
    env.close()
    print(f"\n{'=' * 60}")
    print(f"Done — {current_demo_count} demos saved to {os.path.abspath(args_cli.dataset_file)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
