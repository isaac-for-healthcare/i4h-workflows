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

"""Record spread-tablecloth demos via teleoperation."""

import argparse
import contextlib
import faulthandler
import os
import sys
import time
import weakref
from collections.abc import Callable
from pathlib import Path

faulthandler.enable(all_threads=True)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record demonstrations for the spread-tablecloth task.")
parser.add_argument("--task", type=str, default="Isaac-Spread-Tablecloth-G129-Inspire-Teleop")
parser.add_argument("--teleop_device", type=str, default="handtracking")
parser.add_argument("--dataset_file", type=str, default="./datasets/tablecloth/demo.hdf5")
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--num_demos", type=int, default=0, help="0 = infinite")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Consecutive env steps the success term must hold before the demo is auto-saved.",
)
parser.add_argument("--cloudxr_env", type=str, default="cloudxrjs", help="cloudxrjs / avp / none / path")
parser.add_argument("--auto_launch_cloudxr", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
parser.add_argument(
    "--physics_backend",
    type=str,
    default="newton",
    choices=["newton", "physx"],
    help="Physics backend preset: 'newton' (coupled MJWarp+VBD cloth) or 'physx'.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.task is None:
    parser.error("--task is required")

if "handtracking" in args_cli.teleop_device.lower():
    vars(args_cli)["xr"] = True

# Pinocchio must load before Kit (linker conflict with Kit's libstdc++).
if args_cli.enable_pinocchio:
    with contextlib.suppress(Exception):
        import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Opt-in Warp per-launch verification (crashes with use_cuda_graph=True; see 8e8f6860).
if os.environ.get("DEBUG_WARP_VERIFY", "0").lower() in ("1", "true", "yes"):
    import warp as wp  # noqa: E402

    wp.config.verify_cuda = True

# --- Imports below need Kit running ---

import carb.input
import gymnasium as gym
import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
import omni.appwindow
import omni.ui as ui
import torch
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization
from isaaclab.utils import configclass
from isaaclab_mimic.ui.instruction_display import InstructionDisplay
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV, create_isaac_teleop_device, poll_control_events

_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from simulation.tasks import spread_tablecloth  # noqa: E402, F401
from simulation.tasks.spread_tablecloth.cloth_physics import select_physics_backend  # noqa: E402

_CLOUDXR_ENV_SHORTHANDS = {"cloudxrjs": CLOUDXR_JS_ENV, "avp": CLOUDXR_AVP_ENV}


def _resolve_cloudxr_env(value: str | None) -> str | None:
    if value is None or value.strip() == "" or value.lower() == "none":
        return None
    return _CLOUDXR_ENV_SHORTHANDS.get(value.lower(), value)


class RateLimiter:
    def __init__(self, hz: int):
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)
        self.last_time = time.time()

    def sleep(self, env: gym.Env):
        deadline = self.last_time + self.sleep_duration
        while time.time() < deadline:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = max(self.last_time + self.sleep_duration, time.time())


class KeyboardControls:
    """B = start, S = save, R = reset."""

    def __init__(self):
        self._ci = carb.input
        self._input = carb.input.acquire_input_interface()
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._flags = {"start": False, "save": False, "reset": False}
        self._sub = self._input.subscribe_to_keyboard_events(
            self._kb,
            lambda ev, *a, obj=weakref.proxy(self): obj._on_key(ev, *a),
        )

    def _on_key(self, event, *args, **kwargs) -> bool:
        if event.type == self._ci.KeyboardEventType.KEY_PRESS:
            key_map = {
                self._ci.KeyboardInput.B: "start",
                self._ci.KeyboardInput.S: "save",
                self._ci.KeyboardInput.R: "reset",
            }
            name = key_map.get(event.input)
            if name:
                self._flags[name] = True
        return True

    def consume(self, name: str) -> bool:
        if self._flags.get(name):
            self._flags[name] = False
            return True
        return False

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._input.unsubscribe_to_keyboard_events(self._kb, self._sub)


class PreStepCameraObservationsRecorder(RecorderTerm):
    """Records the camera_images obs group every pre-step."""

    def record_pre_step(self):
        cam_obs = self._env.obs_buf.get("camera_images")
        if cam_obs is None:
            return None, None
        return "obs_camera", cam_obs


@configclass
class _CameraRecorderTermCfg(RecorderTermCfg):
    class_type: type = PreStepCameraObservationsRecorder


@configclass
class ActionStateCameraRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Default recorder + camera observations."""

    record_pre_step_camera_observations: _CameraRecorderTermCfg = _CameraRecorderTermCfg()


def _setup_output_dir() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, output_name


def _create_env_cfg(output_dir: str, output_name: str):
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]

    # Apply physics backend preset onto env_cfg.sim.physics before gym.make.
    selected_backend = select_physics_backend(env_cfg, args_cli.physics_backend)
    print(f"[INFO]: Physics backend = {selected_backend}", flush=True)

    use_isaac_teleop = hasattr(env_cfg, "isaac_teleop") and env_cfg.isaac_teleop is not None

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    if use_isaac_teleop or args_cli.xr:
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
            if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "camera_images"):
                env_cfg.observations.camera_images = None

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    has_cameras = (
        args_cli.enable_cameras
        and hasattr(env_cfg, "observations")
        and hasattr(env_cfg.observations, "camera_images")
        and env_cfg.observations.camera_images is not None
    )
    if has_cameras:
        env_cfg.recorders: ActionStateCameraRecorderManagerCfg = ActionStateCameraRecorderManagerCfg()
    else:
        env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term, use_isaac_teleop


def _create_teleop(callbacks: dict[str, Callable], use_isaac_teleop: bool) -> object:
    if use_isaac_teleop:
        return create_isaac_teleop_device(
            env_cfg.isaac_teleop,
            sim_device=args_cli.device,
            callbacks=callbacks,
            cloudxr_env_file=_resolve_cloudxr_env(args_cli.cloudxr_env),
            auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
        )

    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        return create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)

    if args_cli.teleop_device.lower() == "keyboard":
        dev = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    elif args_cli.teleop_device.lower() == "spacemouse":
        dev = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    else:
        raise RuntimeError(f"Unsupported teleop device: {args_cli.teleop_device}")
    for k, cb in callbacks.items():
        dev.add_callback(k, cb)
    return dev


def _check_success(env, success_term, count: int) -> tuple[int, bool]:
    if success_term is None:
        return count, False
    if bool(success_term.func(env, **success_term.params)[0]):
        count += 1
        return count, count >= args_cli.num_success_steps
    return 0, False


def run(env, success_term, rate_limiter, use_isaac_teleop: bool) -> int:
    demo_count = 0
    success_steps = 0
    teleop_active = False
    recording = False
    should_reset = False
    kb_override = False
    target = args_cli.num_demos or "inf"

    def _noop_cb():
        pass

    teleop = _create_teleop({"R": _noop_cb, "START": _noop_cb, "STOP": _noop_cb, "RESET": _noop_cb}, use_isaac_teleop)
    kb = KeyboardControls()

    label = f"Ready. Press B to start demo 1/{target}"
    display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        win = EmptyWindow(env, "Instruction")
        with win.ui_window_elements["main_vstack"]:
            demo_lbl = ui.Label(label)
            display.set_labels(ui.Label(""), demo_lbl)

    def handle_reset():
        nonlocal success_steps, should_reset, teleop_active, recording, kb_override
        env.reset()
        teleop.reset()
        env.recorder_manager.reset()
        env.sim.step(render=True)
        success_steps = 0
        teleop_active = False
        recording = False
        should_reset = False
        kb_override = False

    def do_begin():
        nonlocal teleop_active, recording, label, kb_override
        handle_reset()
        teleop_active = recording = True
        kb_override = True
        label = f"Recording demo {demo_count + 1}/{target}"
        display.show_demo(label)
        print(f"[B] Recording demo {demo_count + 1}")

    def do_save():
        nonlocal demo_count, label
        if not recording:
            return
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
        env.recorder_manager.export_episodes([0])
        demo_count += 1
        print(f"[S] Demo {demo_count} saved")
        handle_reset()
        label = f"Ready. Press B to start demo {demo_count + 1}/{target}"
        display.show_demo(label)

    def do_discard():
        nonlocal label
        handle_reset()
        label = f"Ready. Press B to start demo {demo_count + 1}/{target}"
        display.show_demo(label)
        print("[R] Reset")

    def loop():
        nonlocal should_reset, teleop_active, success_steps, kb_override

        handle_reset()
        print(f"\nReady — target: {target} demos  |  B=start  S=save  R=reset\n")

        try:
            with torch.inference_mode():
                while simulation_app.is_running():
                    kb_begin = kb.consume("start")
                    kb_save = kb.consume("save")
                    kb_reset = kb.consume("reset")

                    action = teleop.advance()

                    if use_isaac_teleop and not kb_override:
                        ctrl = poll_control_events(teleop)
                        if ctrl.is_active is not None:
                            teleop_active = ctrl.is_active
                        if ctrl.should_reset:
                            should_reset = True

                    if kb_save:
                        do_save()
                        if args_cli.num_demos > 0 and demo_count >= args_cli.num_demos:
                            break
                        continue

                    if kb_begin:
                        do_begin()
                        continue

                    if kb_reset or should_reset:
                        do_discard()
                        continue

                    if action is None:
                        env.sim.render()
                        continue

                    if teleop_active:
                        act = action.repeat(env.num_envs, 1)
                        env.step(act)
                    else:
                        env.sim.render()

                    if recording:
                        success_steps, triggered = _check_success(env, success_term, success_steps)
                        if triggered:
                            do_save()
                            if args_cli.num_demos > 0 and demo_count >= args_cli.num_demos:
                                break

                    if env.sim.is_stopped():
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)

        except KeyboardInterrupt:
            print("\nInterrupted")

    try:
        if use_isaac_teleop:
            with teleop:
                loop()
        else:
            loop()
    finally:
        kb.close()

    return demo_count


def main():
    output_dir, output_name = _setup_output_dir()

    global env_cfg
    env_cfg, success_term, use_isaac_teleop = _create_env_cfg(output_dir, output_name)

    if args_cli.xr or use_isaac_teleop:
        rate_limiter = None
        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    n = run(env, success_term, rate_limiter, use_isaac_teleop)
    env.close()
    print(f"Done — {n} demos saved to {os.path.abspath(args_cli.dataset_file)}")


if __name__ == "__main__":
    main()
    simulation_app.update()
    simulation_app.close()
