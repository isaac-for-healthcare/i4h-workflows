# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

from i4h_arena.scenes.base import Scene, scene_specs
from i4h_common.manifest import SceneSpec


class _BlankScene(Scene):
    name = "blank"

    def build(self) -> Any:
        raise NotImplementedError


def test_zero_dof_scene_does_not_resolve_a_robot_home_pose() -> None:
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
    scene = _BlankScene(spec, SimpleNamespace(device="cpu"))
    env = SimpleNamespace(
        action_space=SimpleNamespace(shape=(0,)),
        unwrapped=SimpleNamespace(num_envs=1),
    )

    actuation = scene.make_actuation(env)

    assert actuation.dof == 0
    assert actuation.numpy().shape == (1, 0)


def test_only_the_g1_loco_manipulation_scenes_rebuild_physx_per_episode() -> None:
    """NVBug 6953999: opting in drops IsaacLab ``startup`` events, so it stays per scene."""
    opted_in = set()
    for name, spec in scene_specs().items():
        module, _, attr = spec.impl.partition(":")
        if getattr(importlib.import_module(module), attr).fresh_physics_per_episode:
            opted_in.add(name)

    assert Scene.fresh_physics_per_episode is False
    assert opted_in == {"g1_cart", "g1_tray"}
