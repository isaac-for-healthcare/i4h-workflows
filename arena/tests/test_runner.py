# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from i4h_arena import runner
from i4h_engine.task import TickContext


def test_runtime_class_has_an_explicit_name() -> None:
    assert runner.SimulationRunner.__name__ == "SimulationRunner"
    assert not hasattr(runner, "Runner")


def test_active_runner_requires_a_live_run() -> None:
    with pytest.raises(RuntimeError, match="launch a workflow.*--live"):
        runner.active_runner()


def test_render_scene_only_renders_and_pumps_ui_without_physics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    env = SimpleNamespace(unwrapped=SimpleNamespace(sim=SimpleNamespace(render=lambda: calls.append("render"))))
    app = SimpleNamespace(update=lambda: calls.append("update"))
    settings = SimpleNamespace(
        get=lambda path, default: True,
        set_bool=lambda path, value: calls.append(f"{path}={value}"),
    )
    monkeypatch.setattr(runner, "_kit_settings_manager", lambda: settings)

    runner._render_scene_only(env, app)

    assert calls == [
        "render",
        "/app/player/playSimulations=False",
        "update",
        "/app/player/playSimulations=True",
    ]


def test_render_scene_only_requires_sim_render() -> None:
    with pytest.raises(TypeError, match=r"env\.unwrapped\.sim\.render"):
        runner._render_scene_only(SimpleNamespace(unwrapped=SimpleNamespace()), SimpleNamespace())


def test_live_reset_rebuilds_actuation_and_restarts_recording() -> None:
    calls: list[str] = []
    replacement = SimpleNamespace(dof=3)
    simulation = object.__new__(runner.SimulationRunner)
    simulation.env = SimpleNamespace(reset=lambda: calls.append("env_reset"))
    simulation._view = SimpleNamespace(invalidate=lambda: calls.append("invalidate"))
    simulation.scene = SimpleNamespace(
        on_reset=lambda _env, _view: calls.append("scene_reset"),
        make_actuation=lambda _env, _view: replacement,
    )
    simulation.recorder = SimpleNamespace(
        restart_episode=lambda **kwargs: calls.append(f"record:{kwargs['node']}:{kwargs['task_id']}")
    )
    simulation._sensor_windows = (SimpleNamespace(on_scene_reset=lambda: calls.append("windows")),)
    engine = SimpleNamespace(
        active_nodes=("drive",),
        states={"drive": SimpleNamespace(node=SimpleNamespace(task_id="teleop/drive"))},
        replace_actuation=lambda _ctx, actuation: calls.append(f"actuation:{actuation is replacement}"),
    )
    ctx = TickContext(scene=SimpleNamespace(num_envs=1), act=SimpleNamespace(dof=3))

    simulation._reset_active_scene(engine, ctx)

    assert calls == [
        "env_reset",
        "invalidate",
        "scene_reset",
        "actuation:True",
        "record:drive:teleop/drive",
        "windows",
    ]


def _runner_with(mode: str, *, requested: int | None, resolved: int) -> runner.SimulationRunner:
    simulation = object.__new__(runner.SimulationRunner)
    simulation.workflow = SimpleNamespace(mode=mode)
    simulation._requested_max_steps = requested
    simulation.max_steps = resolved
    return simulation


def test_idle_never_takes_a_step_budget() -> None:
    """Render-only ticks take no physics step, so a cap would only limit looking."""
    assert _runner_with("idle", requested=600, resolved=600)._engine_max_steps() is None


def test_policy_modes_use_the_resolved_budget() -> None:
    assert _runner_with("policy", requested=None, resolved=600)._engine_max_steps() == 600


def test_teleop_stays_uncapped_when_no_cap_was_asked_for() -> None:
    """A demonstration is not cut off underneath the person giving it."""
    assert _runner_with("teleop", requested=None, resolved=600)._engine_max_steps() is None


def test_teleop_honours_an_explicit_cap() -> None:
    """Without this a failing attempt never ends, so it is never written out.

    Teleop runs ``until=success`` with no time limit. An attempt that will not
    succeed has no other way to terminate, and an episode that never terminates
    is never finalized -- which is what left ``--record-failures`` with nothing
    to record.
    """
    assert _runner_with("teleop", requested=600, resolved=600)._engine_max_steps() == 600
