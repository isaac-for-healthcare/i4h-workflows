# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulation defaults kept from before Isaac Lab 3.0, and where scenes apply them."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from i4h_arena import rendering, simulation
from i4h_arena.scenes.base import Scene


def test_restores_the_pre_3_0_actuator_path_and_physx_forces():
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(
            use_newton_actuators=True, physics=SimpleNamespace(enable_external_forces_every_iteration=True)
        )
    )

    simulation.apply_simulation_defaults(env_cfg)

    assert env_cfg.sim.use_newton_actuators is False
    assert env_cfg.sim.physics.enable_external_forces_every_iteration is False


def test_leaves_a_newton_physics_cfg_without_the_physx_field_alone():
    physics = SimpleNamespace(num_substeps=4)
    env_cfg = SimpleNamespace(sim=SimpleNamespace(use_newton_actuators=True, physics=physics))

    simulation.apply_simulation_defaults(env_cfg)

    assert env_cfg.sim.use_newton_actuators is False
    assert vars(physics) == {"num_substeps": 4}


class _Scene(Scene):
    name = "test_scene"

    def __init__(self, arena_env):
        super().__init__(spec=None, args=argparse.Namespace())
        self._arena_env = arena_env

    def build(self):
        return self._arena_env


def test_arena_environment_applies_them_after_scene_tuning(monkeypatch):
    calls = []

    class Owner:
        rtx_settings = None

        def tune_env_cfg(self, env_cfg):
            calls.append("tune")
            return env_cfg

    monkeypatch.setattr(simulation, "apply_simulation_defaults", lambda cfg: calls.append("simulation"))
    monkeypatch.setattr(rendering, "apply_to_env_cfg", lambda cfg, settings: calls.append("rendering"))
    arena_env = SimpleNamespace(task=Owner(), embodiment=object(), env_cfg_callback=None)

    _Scene(arena_env).arena_environment().env_cfg_callback(object())

    assert calls == ["tune", "simulation", "rendering"]
