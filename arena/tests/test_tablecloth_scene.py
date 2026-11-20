# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from i4h_arena.scenes.base import load_scene


def _args(**values) -> SimpleNamespace:
    return SimpleNamespace(**{"num_envs": 1, "presets": None, "no_cameras": True, **values})


@pytest.mark.parametrize("robot", ["g1", "h2"])
def test_cloth_runs_on_newton_with_pinocchio(robot):
    args = _args()
    scene = load_scene(f"{robot}_tablecloth", args, register_assets=False)
    scene.configure_args(args)
    assert args.enable_pinocchio and args.presets == "newton"


@pytest.mark.parametrize("robot", ["g1", "h2"])
def test_rejects_unsupported_cloth_backend_before_kit(robot):
    args = _args(presets="physx")
    scene = load_scene(f"{robot}_tablecloth", args, register_assets=False)
    with pytest.raises(ValueError, match="requires --presets newton"):
        scene.configure_args(args)


@pytest.mark.parametrize("robot", ["g1", "h2"])
def test_one_environment_per_session(robot):
    args = _args(num_envs=2)
    scene = load_scene(f"{robot}_tablecloth", args, register_assets=False)
    with pytest.raises(ValueError, match="one environment"):
        scene.configure_args(args)
