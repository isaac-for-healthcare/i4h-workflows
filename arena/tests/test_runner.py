# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
    simulation.initial_state = None
    simulation._view = SimpleNamespace(invalidate=lambda: calls.append("invalidate"))
    simulation.scene = SimpleNamespace(
        on_reset=lambda _env, _view: calls.append("scene_reset"),
        make_actuation=lambda _env, _view: replacement,
    )
    simulation.recorder = SimpleNamespace(
        restart_episode=lambda **kwargs: calls.append(
            f"record:{kwargs['node']}:{kwargs['task_id']}:{kwargs['initial_state']}"
        )
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
        "record:drive:teleop/drive:None",
        "windows",
    ]


class _StatefulEnv:
    """An env with IsaacLab's scene-state API, recording how it was reset."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.num_envs = 2
        self.device = "cpu"
        self.scene = SimpleNamespace(get_state=self._get_state)

    @property
    def unwrapped(self) -> _StatefulEnv:
        return self

    def _get_state(self, is_relative: bool = False) -> dict:
        assert is_relative
        return {"rigid_object": {"organs": {"root_pose": torch.arange(14, dtype=torch.float32).reshape(2, 7)}}}

    def reset(self, seed: int | None = None) -> None:
        self.calls.append(("reset", seed))

    def reset_to(self, state: dict, env_ids: object, seed: int | None = None, is_relative: bool = False) -> None:
        self.calls.append(("reset_to", state, env_ids, seed, is_relative))


def _resetting_runner(*, initial_state: dict | None, recorder: object | None) -> runner.SimulationRunner:
    simulation = object.__new__(runner.SimulationRunner)
    simulation.env = _StatefulEnv()
    simulation._view = SimpleNamespace(invalidate=lambda: None)
    simulation.initial_state = initial_state
    simulation.recorder = recorder
    return simulation


def test_a_recording_run_captures_the_state_its_reset_drew() -> None:
    simulation = _resetting_runner(initial_state=None, recorder=object())

    state = simulation._reset_env(seed=7)

    assert simulation.env.calls == [("reset", 7)]
    np.testing.assert_array_equal(state["rigid_object"]["organs"]["root_pose"], np.arange(7).reshape(1, 7))


def test_a_run_without_a_recorder_captures_nothing() -> None:
    assert _resetting_runner(initial_state=None, recorder=None)._reset_env() is None


def test_replay_restores_the_recorded_state_on_every_env_instead_of_randomizing() -> None:
    """Without this the liver-scan phantom moves between recording and replay (NVBug 6928931)."""
    recorded = {"rigid_object": {"organs": {"root_pose": np.full((1, 7), 0.5, dtype=np.float32)}}}
    simulation = _resetting_runner(initial_state=recorded, recorder=None)

    simulation._reset_env(seed=3)

    ((kind, state, env_ids, seed, is_relative),) = simulation.env.calls
    assert (kind, env_ids, seed, is_relative) == ("reset_to", None, 3, True)
    pose = state["rigid_object"]["organs"]["root_pose"]
    assert isinstance(pose, torch.Tensor)
    np.testing.assert_array_equal(pose.numpy(), np.full((2, 7), 0.5))


def test_replay_refuses_a_state_recorded_from_another_scene() -> None:
    simulation = _resetting_runner(initial_state={"rigid_object": {"needle": {}}}, recorder=None)

    with pytest.raises(ValueError, match=r"no entry for \['rigid_object/organs'\]"):
        simulation._reset_env()


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
