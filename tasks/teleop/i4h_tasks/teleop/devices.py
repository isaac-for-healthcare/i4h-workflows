# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input devices a human drives the robot with.

Not tasks — a device is a source of commands; `drive.py` is the node that
consumes one. Split out so adding a device does not touch the task."""

from __future__ import annotations

import logging
import math
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from i4h_common.bus.base import Latest
from i4h_common.bus.messages import RobotCommand
from i4h_engine.task import TickContext

logger = logging.getLogger("i4h_tasks.teleop")


def keyboard_event_input_name(event: Any) -> str:
    """Return a Kit keyboard input name across carb API representations."""
    value = event.input
    name = str(getattr(value, "name", value)).rsplit(".", maxsplit=1)[-1]
    return name.removeprefix("KEY_").upper()


class InputDevice(ABC):
    """Source of joint targets driven by a human."""

    @abstractmethod
    def read(self, ctx: TickContext) -> np.ndarray | None:
        """Latest command, ``(num_envs, dof)``, or ``None`` if nothing new."""

    def open(self, ctx: TickContext) -> None:
        """Connect / calibrate. Blocking is fine here; never in :meth:`read`."""

    def close(self) -> None:
        """Release the hardware."""

    @property
    def done(self) -> bool:
        """True when the operator signalled end-of-demo."""
        return False


class BusDevice(InputDevice):
    """Reads :class:`~i4h_common.bus.RobotCommand` off the bus.

    Covers every device that already runs out-of-process — the VR client, a
    remote pendant, a browser UI — without a driver here for each one.
    """

    def __init__(self, key: str) -> None:
        self.key = key
        self._latest: Latest[RobotCommand] | None = None

    def open(self, ctx: TickContext) -> None:
        if ctx.bus is None:
            raise RuntimeError(f"teleop device needs a bus for {self.key}; run through run.sh")
        self._latest = Latest(ctx.bus, self.key, RobotCommand)

    def read(self, ctx: TickContext) -> np.ndarray | None:
        command = self._latest.take() if self._latest else None
        if command is None or not command.joint_positions:
            return None
        return np.tile(np.asarray(command.joint_positions, dtype=np.float32), (ctx.num_envs, 1))

    def close(self) -> None:
        if self._latest is not None:
            self._latest.close()
            self._latest = None


class KeyboardDevice(InputDevice):
    """Keyboard control for SO-ARM joints or relative Cartesian motion.

    IsaacLab's keyboard emits a six-value Cartesian delta plus an optional
    gripper command. Relative Cartesian scenes consume the delta directly.
    The joint-controlled SO-ARM maps the first five values to its arm joints
    and the binary command to its jaw.
    """

    def __init__(self, *, sensitivity: float = 1.0, step_rad: float = 0.02) -> None:
        self.sensitivity = sensitivity
        self.step_rad = step_rad
        self._impl: Any = None
        self._targets: np.ndarray | None = None
        self._cartesian = False
        self._done = False

    def open(self, ctx: TickContext) -> None:
        self._cartesian = ctx.act.action_space == "ee_pose"
        if self._cartesian:
            if ctx.act.dof != 6:
                raise RuntimeError(
                    "keyboard teleop supports relative six-value Cartesian scenes; "
                    f"this scene exposes {ctx.act.dof} values"
                )
            self._targets = None
        else:
            self._targets = np.array(ctx.scene.joints().pos, dtype=np.float32, copy=True)
            if self._targets.shape[-1] != ctx.act.dof:
                raise RuntimeError(
                    f"keyboard teleop needs {ctx.act.dof} joint targets, but the scene reports "
                    f"{self._targets.shape[-1]} measured joints"
                )
        try:
            from isaaclab.devices import Se3Keyboard  # noqa: PLC0415
            from isaaclab.devices import Se3KeyboardCfg

            self._impl = Se3Keyboard(
                Se3KeyboardCfg(
                    pos_sensitivity=0.05 * self.sensitivity,
                    rot_sensitivity=0.15 * self.sensitivity,
                    gripper_term=not self._cartesian,
                )
            )
            logger.info("keyboard teleop ready")
        except Exception:  # noqa: BLE001 - no Kit means no keyboard; hold pose instead of dying
            logger.warning("no Isaac keyboard device available; teleop will hold pose", exc_info=True)
            self._impl = None

    def read(self, ctx: TickContext) -> np.ndarray | None:
        if self._impl is None:
            return None
        command = self._impl.advance()
        detach = getattr(command, "detach", None)
        if callable(detach):
            command = detach().cpu().numpy()
        values = np.asarray(command, dtype=np.float32).reshape(-1)
        if self._cartesian:
            return np.tile(values[:6], (ctx.num_envs, 1))

        assert self._targets is not None
        arm_width = self._targets.shape[-1] - 1
        self._targets[:, :arm_width] += values[:arm_width] * self.step_rad
        if values.size > 6:
            self._targets[:, -1] = 0.35 if values[6] > 0 else -0.16
        return self._targets

    @property
    def done(self) -> bool:
        return self._done

    def close(self) -> None:
        self._impl = None


class CatheterKeyboardDevice(InputDevice):
    """Reference-style catheter keys plus four named C-arm projections."""

    def __init__(
        self,
        *,
        insertion_speed_mps: float = 0.016,
        rotation_rate_radps: float = 1.5,
        orbit_rate_radps: float = 0.45,
        key_hold_ttl_s: float = 0.20,
    ) -> None:
        self.insertion_speed_mps = float(insertion_speed_mps)
        self.rotation_rate_radps = float(rotation_rate_radps)
        self.orbit_rate_radps = float(orbit_rate_radps)
        self.key_hold_ttl_s = max(0.05, float(key_hold_ttl_s))
        self._input: Any = None
        self._keyboard: Any = None
        self._keyboard_sub: Any = None
        self._pressed: set[str] = set()
        self._released_at: dict[str, float] = {}
        self._orbit_target_rad: float | None = None
        self._reset_requested = False

    def open(self, ctx: TickContext) -> None:
        if ctx.act.action_space != "catheter_carm_velocity" or ctx.act.dof != 3:
            raise RuntimeError(
                "catheter_keyboard requires the three-value catheter_carm_velocity action space; "
                f"got {ctx.act.action_space!r} with {ctx.act.dof} values"
            )
        try:
            import carb
            import omni.appwindow

            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
            )
            logger.info("catheter keyboard ready: W/S insert, A/D rotate, 1-4 C-arm views, Q/E fine orbit, R reset")
        except Exception:
            logger.warning("no Isaac keyboard device available; catheter command will remain zero", exc_info=True)
            self.close()

    def read(self, ctx: TickContext) -> np.ndarray | None:
        if self._keyboard_sub is None:
            return None
        if self._reset_requested:
            self._reset_requested = False
            self._pressed.clear()
            self._released_at.clear()
            self._orbit_target_rad = None
            ctx.request_scene_reset()
            return np.zeros((ctx.num_envs, 3), dtype=np.float32)
        insertion_speed_mps = float(ctx.controls.get("catheter_insertion_speed_mps", self.insertion_speed_mps))
        forward = float(self._active("W")) - float(self._active("S"))
        rotation = float(self._active("D")) - float(self._active("A"))
        orbit = float(self._active("Q")) - float(self._active("E"))
        if orbit:
            self._orbit_target_rad = None
        elif self._orbit_target_rad is not None:
            joints = ctx.scene.joints("robot")
            current = float(joints.pos[0, joints.index_of("carm_orbit_rad")])
            error = self._orbit_target_rad - current
            if abs(error) < math.radians(0.25):
                self._orbit_target_rad = None
            else:
                orbit = float(np.clip(3.0 * error / self.orbit_rate_radps, -1.0, 1.0))
        command = np.array(
            [
                forward * insertion_speed_mps,
                rotation * self.rotation_rate_radps,
                orbit * self.orbit_rate_radps,
            ],
            dtype=np.float32,
        )
        return np.tile(command, (ctx.num_envs, 1))

    def close(self) -> None:
        if self._input is not None and self._keyboard is not None and self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._pressed.clear()
        self._released_at.clear()
        self._reset_requested = False

    def _active(self, key: str) -> bool:
        if key in self._pressed:
            return True
        released = self._released_at.get(key)
        return released is not None and time.monotonic() - released <= self.key_hold_ttl_s

    def _on_keyboard_event(self, event: Any, *_args: Any) -> bool:
        import carb

        key = keyboard_event_input_name(event)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key == "L":
                self._pressed.clear()
                self._released_at.clear()
                self._orbit_target_rad = None
            elif key == "R":
                self._reset_requested = True
            elif key in {"W", "S", "A", "D", "Q", "E"}:
                self._pressed.add(key)
                self._released_at.pop(key, None)
            elif key in {"1", "2", "3", "4"}:
                projection_key = key
                self._orbit_target_rad = {
                    "1": 0.0,
                    "2": math.radians(45.0),
                    "3": math.radians(90.0),
                    "4": math.radians(-30.0),
                }[projection_key]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE and key in self._pressed:
            self._pressed.remove(key)
            self._released_at[key] = time.monotonic()
        return True


class G1Keyboard23DDevice(InputDevice):
    """Keyboard controller for the G1 whole-body 23-value command space."""

    def __init__(self, *, sensitivity: float = 1.0, base_height: float = 0.75) -> None:
        self.sensitivity = sensitivity
        self.base_height = base_height
        self._impl: Any = None

    def open(self, ctx: TickContext) -> None:
        if ctx.act.dof != 23:
            raise RuntimeError(
                f"keyboard_23d requires the G1 23D WBC action space, but this scene exposes {ctx.act.dof} values"
            )
        from i4h_tasks.teleop.keyboard_23d import (  # noqa: PLC0415
            KeyboardTo23DAdapter,
            KeyboardTo23DConfig,
        )

        self._impl = KeyboardTo23DAdapter(
            KeyboardTo23DConfig(
                pos_sensitivity=0.01 * self.sensitivity,
                rot_sensitivity=0.05 * self.sensitivity,
                default_base_height=self.base_height,
            )
        )

    def read(self, ctx: TickContext) -> np.ndarray | None:
        if self._impl is None:
            return None
        return np.tile(self._impl.advance(), (ctx.num_envs, 1))

    def close(self) -> None:
        if self._impl is not None:
            self._impl.close()
            self._impl = None


class SoArmLeaderDevice(InputDevice):
    """SO-ARM 101 leader arm over serial: the follower mirrors the leader.

    Joint values arrive in the leader's own units and are mapped into sim
    radians using ``arena/i4h_arena/embodiments/manifest/so101.yaml`` — the
    calibration lives with the robot descriptor, not in this driver, because
    the dataset converter needs the same numbers.
    """

    def __init__(self, *, port: str = "/dev/ttyACM1", robot: str = "so101", recalibrate: bool = False) -> None:
        self.port = port
        self.robot_name = robot
        self.recalibrate = recalibrate
        self._serial: Any = None
        self._config: Any = None
        self._home: np.ndarray | None = None

    def open(self, ctx: TickContext) -> None:
        from i4h_common.config import get_robot_config  # noqa: PLC0415

        self._config = get_robot_config(self.robot_name)
        self._home = np.asarray(self._config.home_joint_pos_rad, dtype=np.float32)
        try:
            import serial  # noqa: PLC0415

            self._serial = serial.Serial(self.port, baudrate=1_000_000, timeout=0.0)
            logger.info("SO-ARM leader open on %s", self.port)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"cannot open SO-ARM leader on {self.port}: {exc}. Check the cable, or pass --teleop-port."
            ) from exc

    def read(self, ctx: TickContext) -> np.ndarray | None:
        if self._serial is None or self._home is None:
            return None
        line = self._serial.readline()
        if not line:
            return None  # non-blocking: nothing this tick is normal
        try:
            values = np.asarray([float(v) for v in line.decode().strip().split(",")], dtype=np.float32)
        except (UnicodeDecodeError, ValueError):
            return None
        if values.size != self._home.size:
            return None
        return np.tile(self._to_sim(values), (ctx.num_envs, 1))

    def _to_sim(self, leader_deg: np.ndarray) -> np.ndarray:
        """Leader degrees → sim radians, using the robot descriptor's calibration."""
        extra = self._config.extra.get("so101_leader", {}) if self._config else {}
        signs = extra.get("sim_signs_by_joint", {})
        radians = np.deg2rad(leader_deg)
        for index, joint in enumerate(self._config.joint_names):  # type: ignore[union-attr]
            radians[index] *= float(signs.get(joint, 1.0))
        wrist_baseline = float(extra.get("wrist_flex_baseline_deg", 0.0))
        if wrist_baseline and "wrist_flex" in self._config.joint_names:  # type: ignore[union-attr]
            radians[self._config.joint_index("wrist_flex")] -= np.deg2rad(wrist_baseline)  # type: ignore[union-attr]
        return (radians + self._home).astype(np.float32)  # type: ignore[operator]

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None


def make_device(name: str, **kwargs: Any) -> InputDevice:
    """Resolve a device by name."""
    builders: dict[str, Callable[..., InputDevice]] = {
        "catheter_keyboard": CatheterKeyboardDevice,
        "keyboard": KeyboardDevice,
        "keyboard_23d": G1Keyboard23DDevice,
        "so101_leader": SoArmLeaderDevice,
        "vr": lambda **kw: BusDevice(kw.pop("key", "i4h/teleop/robot/command")),
        "bus": lambda **kw: BusDevice(kw.pop("key", "i4h/teleop/robot/command")),
    }
    builder = builders.get(name)
    if builder is None:
        raise KeyError(f"unknown teleop device {name!r}; known: {sorted(builders)}")
    import inspect  # noqa: PLC0415

    if not isinstance(builder, type):
        return builder(**kwargs)
    accepted = set(inspect.signature(builder).parameters)
    return builder(**{k: v for k, v in kwargs.items() if k in accepted})
