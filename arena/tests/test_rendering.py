# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RTX settings layering and the Scene's composed Arena ``env_cfg_callback``."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from i4h_arena import rendering, simulation
from i4h_arena.scenes.base import Scene


def test_scene_defaults_keep_ambient_light_on():
    assert rendering.SCENE_DEFAULTS["ambient_light_intensity"] == 1.0
    assert rendering.QUALITY["ambient_light_intensity"] == 1.0
    assert rendering.PERFORMANCE["ambient_light_intensity"] == 1.0


def test_scene_defaults_undo_isaaclab_3_rendering_experience():
    # Isaac Lab 3.0 GA's rendering kit turned these on; i4h scenes render with them off.
    for field in (
        "enable_translucency",
        "enable_reflections",
        "enable_global_illumination",
        "enable_ambient_occlusion",
        "enable_cached_raytracing",
    ):
        assert rendering.SCENE_DEFAULTS[field] is False
    assert rendering.SCENE_DEFAULTS["subpixel_mode"] == 0
    assert rendering.SCENE_DEFAULTS["dlss_mode"] == 0
    assert all(key.startswith("/rtx/") for key in rendering.SCENE_DEFAULTS["carb_settings"])


def test_merge_rtx_settings_later_layers_win_and_carb_settings_merge_by_key():
    merged = rendering.merge_rtx_settings(
        {"ambient_light_intensity": 1.0, "carb_settings": {"a": 1, "b": 1}},
        None,
        {"enable_translucency": True, "carb_settings": {"b": 2}},
        {"ambient_light_intensity": 0.5},
    )

    assert merged == {
        "ambient_light_intensity": 0.5,
        "enable_translucency": True,
        "carb_settings": {"a": 1, "b": 2},
    }


def test_merge_rtx_settings_does_not_mutate_presets():
    before = dict(rendering.QUALITY["carb_settings"])
    rendering.merge_rtx_settings(rendering.QUALITY, {"carb_settings": {"/rtx/extra": True}})

    assert rendering.QUALITY["carb_settings"] == before


class _Owner:
    def __init__(self, calls, name, rtx_settings=None):
        self._calls = calls
        self._name = name
        self.rtx_settings = rtx_settings

    def tune_env_cfg(self, env_cfg):
        self._calls.append(self._name)
        return env_cfg


class _Scene(Scene):
    name = "test_scene"

    def __init__(self, arena_env, extra=None):
        super().__init__(spec=None, args=argparse.Namespace())
        self._arena_env = arena_env
        self._extra = extra or {}

    def build(self):
        return self._arena_env

    def scene_rtx_settings(self):
        return self._extra


def test_arena_environment_runs_tuners_then_rtx_settings(monkeypatch):
    calls = []
    applied = {}
    monkeypatch.setattr(rendering, "apply_to_cameras", lambda cfg, s: applied.setdefault("cameras", s))
    monkeypatch.setattr(rendering, "add_startup_event", lambda cfg, s: applied.setdefault("startup", s))
    monkeypatch.setattr(simulation, "apply_simulation_defaults", lambda cfg: None)

    def previous(env_cfg):
        calls.append("previous")
        return env_cfg

    arena_env = SimpleNamespace(
        task=_Owner(calls, "task", {"enable_translucency": True}),
        embodiment=_Owner(calls, "embodiment", {"antialiasing_mode": "DLAA"}),
        env_cfg_callback=previous,
    )
    env_cfg = object()

    out = _Scene(arena_env, {"carb_settings": {"/rtx/extra": True}}).arena_environment()

    assert out.env_cfg_callback(env_cfg) is env_cfg
    assert calls == ["previous", "task", "embodiment"]
    expected = {
        **rendering.SCENE_DEFAULTS,
        "enable_translucency": True,
        "antialiasing_mode": "DLAA",
        "carb_settings": {**rendering.SCENE_DEFAULTS["carb_settings"], "/rtx/extra": True},
    }
    assert applied == {"cameras": expected, "startup": expected}


def test_arena_environment_defaults_to_scene_defaults(monkeypatch):
    applied = {}
    monkeypatch.setattr(rendering, "apply_to_cameras", lambda cfg, s: None)
    monkeypatch.setattr(rendering, "add_startup_event", lambda cfg, s: applied.setdefault("startup", s))
    monkeypatch.setattr(simulation, "apply_simulation_defaults", lambda cfg: None)
    arena_env = SimpleNamespace(task=object(), embodiment=object(), env_cfg_callback=None)

    _Scene(arena_env).arena_environment().env_cfg_callback(object())

    assert applied["startup"] == rendering.SCENE_DEFAULTS
