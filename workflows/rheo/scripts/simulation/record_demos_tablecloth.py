#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Script to record demonstrations for ``Isaac-Spread-Tablecloth-G129-Inspire-Teleop``.

This is a *thin* fork of IsaacLab's upstream ``scripts/tools/record_demos.py``
(``i4h-workflows/third_party/IsaacLab/scripts/tools/record_demos.py``).  The
control flow, callback wiring, and dataset export semantics are intentionally
identical to upstream so behavior alignment is easy to reason about.

Differences vs upstream
-----------------------
1. ``--task`` defaults to ``Isaac-Spread-Tablecloth-G129-Inspire-Teleop`` (you
   can still pass any other task explicitly).
2. ``--dataset_file`` defaults to ``./datasets/tablecloth/demo.hdf5``.
3. ``--enable_pinocchio`` imports ``pinocchio`` *before* ``AppLauncher`` boots
   kit, which is required for Pink IK retargeting (this task uses Pink IK).
4. Adds a ``sys.path`` shim + ``from simulation.tasks import spread_tablecloth``
   to register the spread-tablecloth gym task entrypoint.
5. Adds a small ``KeyboardControls`` fallback (B = start, S = manual save,
   R = reset) so the script can be driven without putting on the headset.

The IsaacTeleop input is hand-tracking only — same convention as upstream's
``Isaac-PickPlace-G1-InspireFTP-Abs-v0`` (one task ID = one IsaacTeleop
pipeline).  If you need a Quest motion-controller variant, register a separate
task ID with its own env_cfg (see ``locomanipulation_g1_env_cfg.py`` upstream
for the controller-side template).

Recording / saving semantics (matches upstream)
-----------------------------------------------
* START / STOP / RESET come from the **IsaacTeleop XR client UI** through the
  ``callbacks=`` dict on the device, mirrored into ``running_recording_instance``
  via ``poll_control_events()`` each frame.
* Demo saving is **automatic** on task success — when the env's
  ``terminations.success`` term reports ``num_success_steps`` consecutive
  successful steps, the episode is flushed via ``record_pre_reset`` →
  ``set_success_to_episodes`` → ``export_episodes``.
* There is **no** manual "save now" button.  If the task has no
  ``terminations.success`` term you'll see a warning at startup and demos will
  not auto-save.

Required arguments:
    --task                    Name of the task.

Optional arguments (most relevant):
    --dataset_file            HDF5 file path for saved demos.
    --num_demos               Number of demos to record (0 = infinite).
    --num_success_steps       Steps of consecutive task success to count as a demo.
    --cloudxr_env             CloudXR .env profile (cloudxrjs / avp / none / path).
    --auto_launch_cloudxr     Toggle CloudXR auto-launch (default on; use
                              ``--no-auto_launch_cloudxr`` to disable).
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib
import logging
import os
import sys
import time
import weakref
from collections.abc import Callable
from pathlib import Path

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for the spread-tablecloth task.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Spread-Tablecloth-G129-Inspire-Teleop",
    help="Name of the task.",
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
        " (Ignored when env_cfg has IsaacTeleop configured, which is the case for the spread-tablecloth task.)"
    ),
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/tablecloth/demo.hdf5",
    help="File path to export recorded demos.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help=(
        "Path to a CloudXR .env file, or a shorthand: 'cloudxrjs' (Quest/Pico, default) or 'avp' (Apple Vision Pro)."
        " Set to 'none' to disable CloudXR auto-launch entirely."
    ),
)
parser.add_argument(
    "--auto_launch_cloudxr",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Auto-launch the CloudXR runtime when --cloudxr_env is set. Use --no-auto_launch_cloudxr to disable.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Import pinocchio before AppLauncher (required for tasks that use Pink IK retargeting).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Validate required arguments
if args_cli.task is None:
    parser.error("--task is required")

app_launcher_args = vars(args_cli)

if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# Pinocchio must be imported *before* kit boots (linker order: kit pulls in
# a conflicting libstdc++ otherwise).  The Pink IK action term needs it.
if args_cli.enable_pinocchio:
    with contextlib.suppress(Exception):
        import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

import omni.ui as ui

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Register the spread-tablecloth gym task.  ``simulation.tasks`` lives one
# directory up from this file; add it to sys.path so the import works no
# matter where the script is invoked from.
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from simulation.tasks import spread_tablecloth  # noqa: E402, F401

logger = logging.getLogger(__name__)

_CLOUDXR_ENV_SHORTHANDS: dict[str, str] = {}


def _resolve_cloudxr_env(value: str | None) -> str | None:
    """Resolve ``--cloudxr_env`` shorthands to absolute ``.env`` file paths.

    Accepts ``"cloudxrjs"`` (Quest/Pico), ``"avp"`` (Apple Vision Pro),
    ``"none"`` / ``None`` (disable), or an arbitrary file path.
    """
    if value is None or value.strip() == "" or value.lower() == "none":
        return None
    if not _CLOUDXR_ENV_SHORTHANDS:
        from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV

        _CLOUDXR_ENV_SHORTHANDS["cloudxrjs"] = CLOUDXR_JS_ENV
        _CLOUDXR_ENV_SHORTHANDS["avp"] = CLOUDXR_AVP_ENV
    return _CLOUDXR_ENV_SHORTHANDS.get(value.lower(), value)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

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
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


# ---------------------------------------------------------------------------
# DIVERGENCE FROM UPSTREAM ``IsaacLab/scripts/tools/record_demos.py``
# ---------------------------------------------------------------------------
#
# This class is a thin carb-keyboard fallback so the operator can drive
# recording without putting on the headset.  Bindings:
#
#     B  →  start_recording_instance      (== Quest UI 'Start')
#     S  →  manual_save_recording_instance (force-export episode as success)
#     R  →  reset_recording_instance      (== Quest UI 'Reset')
#
# The 'S' binding is the most opinionated divergence: upstream relies
# entirely on ``terminations.success`` for auto-export and has no manual save
# entry point.  We keep both: success_term still auto-exports normally, and
# the keyboard 'S' lets you force-mark the current episode as successful for
# tasks where a programmatic success term is missing or unreliable.
# ---------------------------------------------------------------------------
class KeyboardControls:
    """Carb keyboard fallback for B / S / R."""

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

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)


def setup_output_directories() -> tuple[str, str]:
    """Set up output directory + filename stem from ``--dataset_file``."""
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None, bool]:
    """Parse + tweak env_cfg for demo recording.

    Returns ``(env_cfg, success_term, use_isaac_teleop)``.
    """
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    use_isaac_teleop = hasattr(env_cfg, "isaac_teleop") and env_cfg.isaac_teleop is not None

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    if use_isaac_teleop or args_cli.xr:
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
            # remove_camera_configs() drops camera entities from the scene but
            # leaves observation terms that still reference them; clear the
            # camera_images obs group so the manager doesn't raise on resolve.
            if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "camera_images"):
                env_cfg.observations.camera_images = None
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term, use_isaac_teleop


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)


def setup_teleop_device(callbacks: dict[str, Callable], use_isaac_teleop: bool = False) -> object:
    """Create the teleop device that drives ``env.step()``.

    For IsaacTeleop tasks this builds an ``isaacteleop`` device backed by the
    ``pipeline_builder`` hardcoded in ``env_cfg.isaac_teleop`` (one task =
    one pipeline, mirroring the upstream IsaacLab convention).  For native
    tasks this falls back to keyboard / spacemouse from
    ``env_cfg.teleop_devices``.
    """
    teleop_interface = None
    try:
        if use_isaac_teleop:
            from isaaclab_teleop import create_isaac_teleop_device

            teleop_interface = create_isaac_teleop_device(
                env_cfg.isaac_teleop,
                sim_device=args_cli.device,
                callbacks=callbacks,
                cloudxr_env_file=_resolve_cloudxr_env(args_cli.cloudxr_env),
                auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
            )

        elif hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Supported devices: keyboard, spacemouse, handtracking")
                exit(1)

            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        exit(1)

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        exit(1)

    return teleop_interface


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Auto-export the current episode when ``success_term`` holds for ``num_success_steps``."""
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(
    env: gym.Env,
    success_step_count: int,
    instruction_display: InstructionDisplay,
    label_text: str,
    teleop_interface: object | None = None,
) -> int:
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    if teleop_interface is not None and hasattr(teleop_interface, "reset"):
        teleop_interface.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
    use_isaac_teleop: bool = False,
) -> int:
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    # For IsaacTeleop or XR, default to inactive until START is triggered
    running_recording_instance = not (args_cli.xr or use_isaac_teleop)

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("Recording started")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")


    def manual_save_recording_instance():
        nonlocal should_reset_recording_instance
        if not running_recording_instance:
            print("Manual save ignored: no recording in progress (press B to start).")
            return
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
        )
        env.recorder_manager.export_episodes([0])
        print("Manual save: episode exported as successful.")
        # Mirror upstream's auto-success path: queue a reset but keep
        # ``running_recording_instance`` True so the next demo starts
        # recording immediately after the env resets.
        should_reset_recording_instance = True

    # For IsaacTeleop the primary control path is poll_control_events();
    # these callbacks are bridged automatically and also serve native
    # (keyboard / spacemouse) devices that look at the "R" key directly.
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    teleop_interface = setup_teleop_device(teleoperation_callbacks, use_isaac_teleop)
    keyboard_controls = KeyboardControls()

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)

    def inner_loop():
        nonlocal current_recorded_demo_count, success_step_count, should_reset_recording_instance
        nonlocal running_recording_instance, label_text

        env.sim.reset()
        env.reset()
        teleop_interface.reset()

        subtasks = {}
        stack_name = "IsaacTeleop" if use_isaac_teleop else "native"
        print(f"{stack_name} recording started.")

        if use_isaac_teleop:
            from isaaclab_teleop import poll_control_events

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                # Keyboard fallback (DIVERGENCE FROM UPSTREAM, see KeyboardControls).
                if keyboard_controls.consume_start():
                    start_recording_instance()
                if keyboard_controls.consume_save():
                    manual_save_recording_instance()
                if keyboard_controls.consume_reset():
                    reset_recording_instance()

                action = teleop_interface.advance()

                if use_isaac_teleop:
                    ctrl = poll_control_events(teleop_interface)
                    if ctrl.is_active is not None:
                        running_recording_instance = ctrl.is_active
                    if ctrl.should_reset:
                        should_reset_recording_instance = True

                if action is None:
                    env.sim.render()
                    continue

                actions = action.repeat(env.num_envs, 1)

                if running_recording_instance:
                    obv = env.step(actions)
                    if subtasks is not None:
                        if subtasks == {}:
                            subtasks = obv[0].get("subtask_terms")
                        elif subtasks:
                            show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
                else:
                    env.sim.render()

                success_step_count_new, success_reset_needed = process_success_condition(
                    env, success_term, success_step_count
                )
                success_step_count = success_step_count_new
                if success_reset_needed:
                    should_reset_recording_instance = True

                if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                    print(label_text)

                if (
                    args_cli.num_demos > 0
                    and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos
                ):
                    label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                    instruction_display.show_demo(label_text)
                    print(label_text)
                    target_time = time.time() + 0.8
                    while time.time() < target_time:
                        if rate_limiter:
                            rate_limiter.sleep(env)
                        else:
                            env.sim.render()
                    break

                if should_reset_recording_instance:
                    success_step_count = handle_reset(
                        env, success_step_count, instruction_display, label_text, teleop_interface
                    )
                    should_reset_recording_instance = False

                if env.sim.is_stopped():
                    break

                if rate_limiter:
                    rate_limiter.sleep(env)

    try:
        if use_isaac_teleop:
            with teleop_interface:
                inner_loop()
        else:
            inner_loop()
    finally:
        keyboard_controls.close()

    return current_recorded_demo_count


def main() -> None:
    output_dir, output_file_name = setup_output_directories()

    global env_cfg
    env_cfg, success_term, use_isaac_teleop = create_environment_config(output_dir, output_file_name)

    # IsaacTeleop / XR runs uncapped (rate is enforced by the OpenXR session).
    if args_cli.xr or use_isaac_teleop:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    env = create_environment(env_cfg)

    current_recorded_demo_count = run_simulation_loop(env, None, success_term, rate_limiter, use_isaac_teleop)

    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    # env.close() already closes the USD stage via sim.clear_instance().
    # Pump the event loop so the viewport processes closure, then close.
    simulation_app.update()
    simulation_app.close()
