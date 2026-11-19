# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keyboard teleoperation of the endoluminal catheter and C-arm as an Isaac Lab device.

The scene declares it under ``teleop_devices`` as ``catheter_keyboard``, so Isaac Lab's
``record_demos.py --teleop_device catheter_keyboard`` drives the four velocity
channels: insertion, axial rotation, tip bend and C-arm orbit.

``W/S`` insert and retract, ``A/D`` rotate, ``Z/C`` bend the tip either way, ``Q/E``
orbit the C-arm, ``1``-``4`` servo the C-arm to the AP, LAO-45, lateral and RAO-30
views, ``R`` resets, and ``L`` clears a stuck key. The insertion speed follows the
fluoroscopy panel's velocity slider when the panel is open.
"""

from __future__ import annotations

import logging
import math
import os
import time
import weakref
from collections.abc import Callable

import numpy as np
import torch
from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.utils import configclass

from i4h_arena.ui import live

logger = logging.getLogger("i4h_arena.devices.catheter")

# Keys whose command lasts as long as they are held.
MOTION_KEYS = frozenset({"W", "S", "A", "D", "Q", "E", "Z", "C"})
# Named C-arm projections, in orbit radians.
PROJECTION_KEYS = {"1": 0.0, "2": math.radians(45.0), "3": math.radians(90.0), "4": math.radians(-30.0)}


@configclass
class CatheterKeyboardCfg(DeviceCfg):
    """Rates for the four channels.

    They are set for how slowly these scenes run rather than for how fast a catheter
    moves: insertion is a velocity in simulation time, but the operator holds the key
    in wall time, and a scene rendering a real DRR every frame steps at about two
    hertz. ``insertion_speed_mps`` is bounded by the rod: feeding faster than the
    shaft can shed length buckles it into the vessel wall.
    """

    insertion_speed_mps: float = 0.009
    rotation_rate_radps: float = 1.5
    orbit_rate_radps: float = 0.45
    tip_bend_rate_radps: float = 1.5
    key_hold_ttl_s: float = 0.50
    """How long a key event counts as held, when Kit cannot be asked whether the key is down."""
    class_type: type[DeviceBase] | str = "{DIR}.catheter:CatheterKeyboard"


class CatheterKeyboard(DeviceBase):
    """Held keys as catheter and C-arm velocities, read from the device rather than inferred from events."""

    def __init__(self, cfg: CatheterKeyboardCfg, retargeters: list | None = None):
        super().__init__(retargeters)
        import carb  # noqa: PLC0415
        import omni.appwindow  # noqa: PLC0415

        self.cfg = cfg
        self._callbacks: dict[str, Callable] = {}
        self._held_since: dict[str, float] = {}
        self._orbit_target_rad: float | None = None
        self._key_log = os.environ.get("I4H_TELEOP_KEY_LOG", "").strip().lower() not in ("", "0", "false", "no", "off")
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._buttons = {
            key: getattr(carb.input.KeyboardInput, key) for key in MOTION_KEYS if hasattr(carb.input.KeyboardInput, key)
        }
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard, lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args)
        )

    def __del__(self):
        if getattr(self, "_subscription", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None

    def __str__(self) -> str:
        return "Catheter keyboard: W/S insert, A/D rotate, Z/C tip bend, Q/E orbit, 1-4 C-arm views, R reset"

    def reset(self) -> None:
        self._held_since.clear()
        self._orbit_target_rad = None

    def add_callback(self, key: str, func: Callable) -> None:
        self._callbacks[key] = func

    def advance(self) -> torch.Tensor:
        forward = float(self._active("W")) - float(self._active("S"))
        rotation = float(self._active("D")) - float(self._active("A"))
        orbit = float(self._active("Q")) - float(self._active("E"))
        tip_bend = float(self._active("C")) - float(self._active("Z"))
        if orbit:
            self._orbit_target_rad = None
        elif self._orbit_target_rad is not None:
            orbit = self._servo_orbit()
        insertion = live.CONTROLS.get("catheter_insertion_speed_mps", self.cfg.insertion_speed_mps)
        command = [
            forward * insertion,
            rotation * self.cfg.rotation_rate_radps,
            tip_bend * self.cfg.tip_bend_rate_radps,
            orbit * self.cfg.orbit_rate_radps,
        ]
        if self._key_log and any(command):
            print(f"[teleop command] {[round(value, 4) for value in command]}", flush=True)
        return torch.tensor(command, dtype=torch.float32, device=self.cfg.sim_device)

    def _servo_orbit(self) -> float:
        """Orbit command toward the selected view, proportional to the remaining angle."""
        env = live.current_env()
        if env is None:
            self._orbit_target_rad = None
            return 0.0
        current = float(env.action_manager.get_term("carm_orbit").joint_state().pos[0, 0])
        error = self._orbit_target_rad - current
        if abs(error) < math.radians(0.25):
            self._orbit_target_rad = None
            return 0.0
        return float(np.clip(3.0 * error / self.cfg.orbit_rate_radps, -1.0, 1.0))

    def _active(self, key: str) -> bool:
        """Whether ``key`` is down, asked of the device before it is inferred from events.

        Kit drops releases and, once something else holds the keyboard, stops sending
        presses and repeats, so polling the physical key is what stays right.
        """
        button = self._buttons.get(key)
        if button is not None:
            try:
                return bool(self._input.get_keyboard_value(self._keyboard, button))
            except Exception:  # noqa: BLE001 - fall back to events for the rest of the session
                logger.warning("keyboard state polling unavailable; falling back to key events", exc_info=True)
                self._buttons.clear()
        held = self._held_since.get(key)
        return held is not None and time.monotonic() - held <= self.cfg.key_hold_ttl_s

    def _on_keyboard_event(self, event, *args) -> bool:
        import carb  # noqa: PLC0415

        # Across carb versions the input is an enum or its name, and digits arrive as KEY_1.
        key = str(getattr(event.input, "name", event.input)).rsplit(".", maxsplit=1)[-1].removeprefix("KEY_").upper()
        if self._key_log:
            print(f"[teleop keys] {event.type} {key} held={sorted(self._held_since)}", flush=True)
        types = carb.input.KeyboardEventType
        if key in MOTION_KEYS and event.type in (types.KEY_PRESS, types.KEY_REPEAT):
            self._held_since[key] = time.monotonic()
        elif event.type == types.KEY_PRESS:
            if key == "L":
                self.reset()
            elif key == "R":
                self.reset()
                if "R" in self._callbacks:
                    self._callbacks["R"]()
            elif key in PROJECTION_KEYS:
                self._orbit_target_rad = PROJECTION_KEYS[key]
        return True
