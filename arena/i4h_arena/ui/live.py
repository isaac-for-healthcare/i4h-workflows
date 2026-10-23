# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The running environment's operator UI: its sensor windows and their shared controls.

A scene with sensor views (the endoluminal fluoroscopy panel) adds :func:`attach` as
a startup event, so the panel opens in any Kit session with a window, whichever
Arena or Isaac Lab script built the environment. The panel refreshes itself on Kit's
update loop rather than waiting for a runner to push frames. Teleoperation devices,
which Isaac Lab builds without the environment, read the live environment and the
panel's controls (such as the insertion velocity slider) from here.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any

logger = logging.getLogger("i4h_arena.ui.live")

#: Operator settings shared by the panel's sliders and the teleoperation devices.
CONTROLS: dict[str, float] = {}

_env: weakref.ReferenceType | None = None
_windows: list[Any] = []
_view: Any = None
_subscription: Any = None
_last_episode_step: int | None = None
_last_env_step: int | None = None


def current_env() -> Any | None:
    """The environment :func:`attach` registered, or ``None`` before one starts."""
    return _env() if _env is not None else None


def attach(env: Any, env_ids: Any, scene_name: str) -> None:
    """Startup event: register ``env`` and, with a Kit window, open its scene's sensor views."""
    global _env, _view, _subscription, _last_episode_step
    _env = weakref.ref(env)
    if not scene_name or not env.sim.has_gui:
        return
    from i4h_arena.environments import live_scene  # noqa: PLC0415

    scene = live_scene(scene_name)
    if scene is None or not scene.default_sensor_views():
        return
    import omni.kit.app  # noqa: PLC0415

    from i4h_arena.ui.sensor_image_window import SensorImageWindow  # noqa: PLC0415

    view = _view = scene.make_sensor_view(env)
    for name in scene.default_sensor_views():
        _windows.append(
            SensorImageWindow(
                name=name,
                title=scene.sensor_view_titles().get(name),
                view=view,
                outputs=scene.sensor_view_outputs().get(name, (("RGB", "rgb"),)),
                keyboard_toggles=scene.sensor_view_keyboard_toggles().get(name, {}),
                projection_presets=scene.sensor_view_projection_presets().get(name, ()),
                projection_default_index=scene.sensor_view_projection_defaults().get(name, 0),
                appearances=scene.sensor_view_appearances().get(name, ()),
                display_controls=scene.sensor_view_display_controls().get(name, ()),
                sliders=scene.sensor_view_sliders().get(name, ()),
                readout=scene.sensor_view_readouts(env).get(name),
                controls=CONTROLS,
            )
        )
    _last_episode_step = None
    _subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(_refresh)


def _refresh(_event: Any) -> None:
    """Show the latest frame; restore the panel's scene controls when env 0 starts a new episode."""
    global _last_episode_step, _last_env_step
    env = current_env()
    if env is None:
        return
    try:
        if _view is not None and env.common_step_counter != _last_env_step:
            # The view caches what it reads within a step; a new step makes it stale.
            _last_env_step = env.common_step_counter
            _view.invalidate()
        step = int(env.episode_length_buf[0])
        reset = _last_episode_step is not None and step < _last_episode_step
        _last_episode_step = step
        for window in _windows:
            if reset:
                window.on_scene_reset()
            window.update()
    except Exception:
        logger.exception("sensor window refresh failed; the panel stops updating")
        close()


def close() -> None:
    global _subscription
    _subscription = None
    for window in _windows:
        window.close()
    _windows.clear()
