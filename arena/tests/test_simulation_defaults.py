# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulation defaults kept from before Isaac Lab 3.0, and where scenes apply them."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import pytest
from isaaclab.utils.string import ResolvableString

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


# -- a fresh PhysX scene for every episode (NVBug 6953999) ------------------


class _PhysxManager:
    """Stands in for ``isaaclab_physx.physics.PhysxManager``."""


class _NewtonManager:
    """Stands in for another backend's physics manager."""


def _kit_physx(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Install the Kit modules ``rebuild_physx_scene`` imports."""
    interface = SimpleNamespace(detach_stage=lambda: calls.append("detach_stage"))
    monkeypatch.setitem(sys.modules, "isaaclab_physx.physics", SimpleNamespace(PhysxManager=_PhysxManager))
    monkeypatch.setitem(sys.modules, "omni.physx", SimpleNamespace(get_physx_simulation_interface=lambda: interface))


def _physics_env(manager: type, calls: list[str]) -> SimpleNamespace:
    # SimulationContext keeps the configured manager class as an unresolved
    # "module:Class" string, not as the class.
    sim = SimpleNamespace(
        physics_manager=ResolvableString(f"{__name__}:{manager.__name__}"),
        stop=lambda: calls.append("stop"),
        play=lambda: calls.append("play"),
    )
    scene = SimpleNamespace(update=lambda dt: calls.append(f"update:{dt}"))
    return SimpleNamespace(unwrapped=SimpleNamespace(sim=sim, scene=scene, physics_dt=0.005))


def test_rebuild_detaches_the_stage_so_play_parses_a_new_physx_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _kit_physx(monkeypatch, calls)

    assert simulation.rebuild_physx_scene(_physics_env(_PhysxManager, calls))

    # Without the detach, play hangs on "Stage ... already attached".
    assert calls == ["stop", "detach_stage", "play", "update:0.005"]


def test_rebuild_leaves_another_physics_backend_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _kit_physx(monkeypatch, calls)

    assert not simulation.rebuild_physx_scene(_physics_env(_NewtonManager, calls))
    assert calls == []


def test_rebuild_is_a_no_op_when_kit_has_no_physx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "isaaclab_physx.physics", None)  # import raises ImportError

    assert not simulation.rebuild_physx_scene(_physics_env(_PhysxManager, calls))
    assert calls == []


@pytest.mark.parametrize("failing", ["detach_stage", "play"])
def test_a_failed_rebuild_names_the_bug_instead_of_failing_at_the_next_reset(
    monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    calls: list[str] = []
    _kit_physx(monkeypatch, calls)
    env = _physics_env(_PhysxManager, calls)

    def fail() -> None:
        calls.append(failing)
        raise RuntimeError(f"{failing} failed")

    if failing == "detach_stage":
        monkeypatch.setitem(
            sys.modules,
            "omni.physx",
            SimpleNamespace(get_physx_simulation_interface=lambda: SimpleNamespace(detach_stage=fail)),
        )
    else:
        env.unwrapped.sim.play = fail

    with pytest.raises(RuntimeError, match="NVBug 6953999") as raised:
        simulation.rebuild_physx_scene(env)

    assert str(raised.value.__cause__) == f"{failing} failed"
    # A play after a failed detach could hang on the still-attached stage.
    assert calls == (["stop", "detach_stage"] if failing == "detach_stage" else ["stop", "detach_stage", "play"])


@pytest.mark.parametrize(
    ("opted_in", "args", "installed"),
    [
        (True, {}, True),
        (False, {}, False),
        # An operator's reset happens mid-episode; rebuilding under a live session is not validated.
        (True, {"teleop_device": "keyboard"}, False),
        (True, {"num_demos": 0, "cloudxr_env": None}, False),
    ],
)
def test_the_environment_asks_for_a_fresh_physics_scene_only_for_scenes_that_opt_in(
    monkeypatch, opted_in, args, installed
):
    monkeypatch.setattr(rendering, "apply_to_env_cfg", lambda cfg, settings: None)

    class Owner:
        rtx_settings = None

    class OptedIn(_Scene):
        fresh_physics_per_episode = opted_in

    scene = OptedIn(SimpleNamespace(task=Owner(), embodiment=object(), env_cfg_callback=None))
    scene.args = argparse.Namespace(**args)
    env_cfg = SimpleNamespace(sim=SimpleNamespace(use_newton_actuators=True, physics=SimpleNamespace()))

    scene.arena_environment().env_cfg_callback(env_cfg)

    assert (getattr(env_cfg, "fresh_physics_per_episode", None) is simulation.rebuild_physx_scene) is installed
