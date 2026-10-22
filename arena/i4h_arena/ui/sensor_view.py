# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A running environment's camera-compatible sensors, as its operator panels read them.

The panels redraw on every Kit update, more often than the environment steps, so a
frame is read once and cached until :meth:`SensorView.invalidate`, which
:mod:`i4h_arena.ui.live` calls whenever the environment's step counter advances.
Display controls reach the sensor itself and invalidate the cache, so the next
redraw shows their effect.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from i4h_common.types import CameraFrame

logger = logging.getLogger("i4h_arena.ui.sensor_view")


def _np(value: Any) -> np.ndarray:
    """Detach a torch tensor (or an Isaac Lab ``ProxyArray`` around one) to numpy."""
    if isinstance(getattr(type(value), "torch", None), property):
        value = value.torch
    detach = getattr(value, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    return np.asarray(value)


class SensorView:
    """Read env 0's frames from the named sensors and drive their display controls."""

    def __init__(self, env: Any, sensors: tuple[str, ...]) -> None:
        self._env = env
        self._sensors = tuple(sensors)
        self._cache: dict[str, CameraFrame | None] = {}

    def invalidate(self) -> None:
        self._cache.clear()

    def _sensor(self, name: str) -> Any:
        return self._env.unwrapped.scene[name]

    def camera(self, name: str, *, output: str = "rgb") -> CameraFrame | None:
        """Env 0's ``output`` of sensor ``name`` as 8-bit RGB, or ``None`` when it has none."""
        if name not in self._sensors:
            return None
        key = f"{name}:{output}"
        if key not in self._cache:
            try:
                frame = _np(self._sensor(name).data.output[output])[0][..., :3].astype(np.uint8)
            except (KeyError, IndexError, AttributeError, TypeError):
                logger.debug("sensor %s has no %s output", name, output, exc_info=True)
                self._cache[key] = None
                return None
            self._cache[key] = CameraFrame(
                name=name, height=int(frame.shape[0]), width=int(frame.shape[1]), data=frame.tobytes()
            )
        return self._cache[key]

    def set_sensor_brightness(self, name: str, brightness: float) -> float | None:
        setter = getattr(self._sensor(name), "set_dsa_brightness", None)
        return float(setter(brightness)) if callable(setter) else None

    def set_sensor_appearance(self, name: str, appearance: str) -> str | None:
        """Select a display look, such as fluoroscopy or X-ray."""
        setter = getattr(self._sensor(name), "set_display_appearance", None)
        selected = setter(appearance) if callable(setter) else None
        if selected is None:
            return None
        self.invalidate()
        return str(selected)

    def set_sensor_display_control(self, name: str, control: str, value: float) -> float | None:
        """Adjust the display mapping, such as the window width."""
        setter = getattr(self._sensor(name), "set_display_control", None)
        selected = setter(control, value) if callable(setter) else None
        if selected is None:
            return None
        self.invalidate()
        return float(selected)

    def recalibrate_sensor_display(self, name: str) -> bool:
        """Re-fit the display mapping to the next frame."""
        recalibrate = getattr(self._sensor(name), "recalibrate_display", None)
        recalibrated = bool(recalibrate()) if callable(recalibrate) else False
        if recalibrated:
            self.invalidate()
        return recalibrated

    def select_sensor_projection(self, name: str, angle_rad: float) -> float | None:
        """Select a named projection, such as a C-arm view."""
        select = getattr(self._sensor(name), "select_projection", None)
        if not callable(select):
            return None
        selected = float(select(angle_rad))
        self.invalidate()
        return selected
