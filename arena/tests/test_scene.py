# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from i4h_arena import environments
from i4h_arena.scenes.base import Scene, scene_specs
from i4h_common.manifest import SceneSpec


class _BlankScene(Scene):
    name = "blank"

    def build(self) -> Any:
        raise NotImplementedError


def test_every_scene_is_an_arena_environment_named_after_it() -> None:
    for name in scene_specs():
        environment = getattr(environments, "".join(part.capitalize() for part in name.split("_")))
        assert environment.name == name


def test_a_zero_dof_scene_opens_no_sensor_panels() -> None:
    spec = SceneSpec(
        name="blank",
        impl="test_scene:_BlankScene",
        embodiment="none",
        action_space="joint_position",
        dof=0,
        cameras=(),
        objects=(),
        robots=(),
        gripper=False,
    )
    scene = _BlankScene(spec, SimpleNamespace())

    assert scene.default_sensor_views() == ()
    assert scene.make_sensor_view(SimpleNamespace(unwrapped=SimpleNamespace(scene={}))).camera("room") is None
