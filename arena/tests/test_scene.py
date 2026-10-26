# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from i4h_arena import environments
from i4h_arena.scenes.base import Scene, load_scene, scene_specs
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


@pytest.mark.parametrize(
    "name", ["g1_tablecloth", "h2_tablecloth", "endoluminal_navigation", "endoluminal_navigation_arm"]
)
def test_newton_only_scenes_default_to_newton_without_presets(name, monkeypatch) -> None:
    """Arena's and Isaac Lab's runners build a scene without ``configure_args``.

    Without ``--presets`` they resolve the environment's default backend, and these
    scenes' callbacks install a Newton solver, which Arena rejects under PhysX.
    """
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    # Arena's Scene imports PhysX USD schemas, which only exist inside Kit.
    arena_scene = ModuleType("isaaclab_arena.scene.scene")
    arena_scene.Scene = lambda assets=(): SimpleNamespace(assets=assets)
    monkeypatch.setitem(sys.modules, "isaaclab_arena.scene.scene", arena_scene)
    args = SimpleNamespace(num_envs=1, presets=None, no_cameras=True, device="cpu")

    assert load_scene(name, args, register_assets=False).build().default_physics_backend is PhysicsBackend.NEWTON
