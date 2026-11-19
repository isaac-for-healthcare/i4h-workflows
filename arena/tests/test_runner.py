# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab.utils.string import ResolvableString

from i4h_arena import runner
from i4h_engine.status import WorkflowStatus
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
    simulation._view = SimpleNamespace(
        on_reset=lambda: calls.append("invalidate"), on_step=lambda: None, invalidate=lambda: calls.append("invalidate")
    )
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
    simulation._view = SimpleNamespace(on_reset=lambda: None, on_step=lambda: None, invalidate=lambda: None)
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


class _App:
    """A Kit app whose window closes on the n-th ``env.step``."""

    def __init__(self) -> None:
        self.running = True

    def is_running(self) -> bool:
        return self.running


class _Engine:
    def __init__(self) -> None:
        self.status = runner.WorkflowStatus.RUNNING
        self.step = 0
        self.detail = ""
        self.segments = ()
        self.active_nodes = ("drive",)
        self.advance_requested = True
        self.terminal_advance_requested = False

    def start(self, _ctx: TickContext) -> None:
        pass

    def tick(self, _ctx: TickContext) -> None:
        self.step += 1

    def abort(self, _ctx: TickContext, *, detail: str = "aborted externally") -> None:
        self.status = runner.WorkflowStatus.ABORTED
        self.detail = detail


class _Recorder:
    def __init__(self) -> None:
        self.steps = 0
        self.ended: list[tuple[runner.EpisodeResult, bool]] = []

    def begin_episode(self, _episode: int, _attempt: int, *, initial_state: object = None) -> None:
        pass

    def on_step(self, _action: object, _view: object) -> None:
        self.steps += 1

    def end_episode(self, result: runner.EpisodeResult, *, keep: bool) -> None:
        self.ended.append((result, keep))


def _closing_runner(step, *, record_failures: bool = True) -> runner.SimulationRunner:
    actuation = SimpleNamespace(tensor=lambda: torch.zeros(1, 3), numpy=lambda: np.zeros((1, 3)))
    simulation = object.__new__(runner.SimulationRunner)
    simulation.app = _App()
    simulation.env = SimpleNamespace(step=lambda command: step(simulation.app), unwrapped=SimpleNamespace(step_dt=0.02))
    simulation.scene = SimpleNamespace(
        fresh_physics_per_episode=False,
        on_reset=lambda _env, _view: None,
        make_actuation=lambda _env, _view: actuation,
    )
    simulation._view = SimpleNamespace(on_reset=lambda: None, on_step=lambda: None, invalidate=lambda: None)
    simulation._reset_env = lambda seed=None: None
    simulation.workflow = SimpleNamespace(mode="teleop")
    simulation.keys = None
    simulation.bus = None
    simulation.publisher = None
    simulation.run_id = "run"
    simulation.seed = None
    simulation._controls = None
    simulation._sensor_windows = ()
    simulation.recorder = _Recorder()
    simulation.record_failures = record_failures
    return simulation


def _closes_on_step(n: int):
    calls = {"steps": 0}

    def step(app: _App) -> None:
        calls["steps"] += 1
        if calls["steps"] == n:
            # What IsaacLab raises when a sub-step reads an articulation after
            # the render inside env.step let Kit handle the closed window.
            app.running = False
            raise ReferenceError("weakly-referenced object no longer exists")

    return step


def test_closing_the_window_mid_step_ends_the_attempt_and_records_it() -> None:
    """NVBug 6941881: closing Isaac Sim during G1 teleop crashed and lost the attempt."""
    simulation = _closing_runner(_closes_on_step(3))

    result = simulation._run_episode(_Engine(), episode=0, attempt=1)

    assert result.status is runner.WorkflowStatus.ABORTED
    assert result.detail == "Isaac Sim was closed"
    assert not result.succeeded
    assert simulation.recorder.steps == 2
    assert simulation.recorder.ended == [(result, True)]


def test_closing_the_window_discards_the_attempt_without_record_failures() -> None:
    simulation = _closing_runner(_closes_on_step(1), record_failures=False)

    result = simulation._run_episode(_Engine(), episode=0, attempt=1)

    assert simulation.recorder.ended == [(result, False)]


def test_a_step_failure_while_isaac_sim_runs_still_raises() -> None:
    def step(_app: _App) -> None:
        raise ReferenceError("weakly-referenced object no longer exists")

    with pytest.raises(ReferenceError):
        _closing_runner(step)._run_episode(_Engine(), episode=0, attempt=1)


def test_a_window_closed_between_steps_is_recorded_as_aborted_not_running() -> None:
    def step(app: _App) -> None:
        app.running = False

    simulation = _closing_runner(step)

    result = simulation._run_episode(_Engine(), episode=0, attempt=1)

    assert result.status is runner.WorkflowStatus.ABORTED
    assert simulation.recorder.steps == 1
    assert simulation.recorder.ended == [(result, True)]


# -- a step that resets the env is not recorded (NVBug 6941887) -------------


def test_reset_during_step_reads_env_zero_of_reset_buf() -> None:
    assert runner._reset_during_step(SimpleNamespace(reset_buf=torch.tensor([True, False])))
    assert not runner._reset_during_step(SimpleNamespace(reset_buf=torch.tensor([False, True])))
    assert not runner._reset_during_step(SimpleNamespace())  # a ManagerBasedEnv has no reset buffer
    assert runner._reset_during_step(SimpleNamespace(unwrapped=SimpleNamespace(reset_buf=np.array([True]))))


class _TerminatingEnv:
    """An RL env whose success termination fires, and auto-resets, on step ``terminal_step``."""

    def __init__(self, terminal_step: int) -> None:
        self.terminal_step = terminal_step
        self.steps = 0
        self.reset_buf = torch.zeros(1, dtype=torch.bool)
        self.step_dt = 1.0 / 30.0

    @property
    def unwrapped(self) -> _TerminatingEnv:
        return self

    def step(self, _command: torch.Tensor) -> None:
        self.steps += 1
        self.reset_buf[0] = self.steps == self.terminal_step


class _UntilTerminated:
    """An engine whose single node succeeds once it observes the termination pulse."""

    def __init__(self, env: _TerminatingEnv) -> None:
        self.env = env
        self.status = WorkflowStatus.RUNNING
        self.step = 0
        self.detail = ""
        self.segments = ()
        self.advance_requested = True
        self.terminal_advance_requested = False
        self.active_nodes = ("policy",)

    def start(self, _ctx: TickContext) -> None:
        self.status = WorkflowStatus.RUNNING

    def tick(self, _ctx: TickContext) -> None:
        if bool(self.env.reset_buf[0]):
            self.status = WorkflowStatus.SUCCEEDED
        else:
            self.step += 1


def test_the_step_that_resets_a_successful_episode_is_not_recorded() -> None:
    """IsaacLab resets a terminated env inside ``step``; that frame shows the next episode."""
    recorded: list[int] = []
    ended: list[bool] = []
    env = _TerminatingEnv(terminal_step=4)
    simulation = object.__new__(runner.SimulationRunner)
    simulation.env = env
    simulation.app = SimpleNamespace(is_running=lambda: True)
    simulation.scene = SimpleNamespace(
        fresh_physics_per_episode=False,
        on_reset=lambda _env, _view: None,
        make_actuation=lambda _env, _view: SimpleNamespace(
            tensor=lambda: torch.zeros(1, 6), numpy=lambda: np.zeros((1, 6))
        ),
    )
    simulation._reset_env = lambda _seed=None: None
    simulation._view = SimpleNamespace(on_reset=lambda: None, on_step=lambda: None, invalidate=lambda: None, num_envs=1)
    simulation._sensor_windows = ()
    simulation._controls = {}
    simulation.keys = simulation.bus = simulation.publisher = None
    simulation.run_id = "run"
    simulation.seed = None
    simulation.workflow = SimpleNamespace(mode="policy")
    simulation.record_failures = False
    simulation.recorder = SimpleNamespace(
        begin_episode=lambda *_args, **_kwargs: None,
        on_step=lambda _action, _view: recorded.append(env.steps),
        end_episode=lambda result, keep: ended.append(keep),
    )

    result = simulation._run_episode(_UntilTerminated(env), episode=0, attempt=1)

    assert result.succeeded
    assert env.steps == 4
    assert recorded == [1, 2, 3]
    assert ended == [True]


# -- a reset the workflow does not finish on ends the attempt (PR 275 review) -


class _TermManager:
    def __init__(self, env: _TerminatingEnv, term: str) -> None:
        self.env = env
        self.active_terms = ["success", term] if term != "success" else ["success"]
        self.term = term

    def get_term(self, name: str) -> torch.Tensor:
        return self.env.reset_buf.clone() if name == self.term else torch.zeros(1, dtype=torch.bool)


class _KeepsRunning(_UntilTerminated):
    """A workflow whose success never reads the termination that fired, like G1 ``object_dropped``.

    ``waiting`` ticks after the reset ask for no step, as a remote task does
    while it waits for its backend.
    """

    def __init__(self, env: _TerminatingEnv, *, waiting: int = 0, succeed_after: int | None = None) -> None:
        super().__init__(env)
        self.waiting = waiting
        self.succeed_after = succeed_after
        self.ticks_since_reset: int | None = None

    def tick(self, _ctx: TickContext) -> None:
        if self.ticks_since_reset is None and bool(self.env.reset_buf[0]):
            self.ticks_since_reset = 0
        if self.ticks_since_reset is not None:
            self.ticks_since_reset += 1
            if self.succeed_after is not None and self.ticks_since_reset >= self.succeed_after:
                self.status = WorkflowStatus.SUCCEEDED
                return
            self.advance_requested = self.ticks_since_reset > self.waiting
            if not self.advance_requested:
                return
        self.step += 1

    def abort(self, _ctx: TickContext, *, detail: str = "aborted externally") -> None:
        self.status = WorkflowStatus.ABORTED
        self.detail = detail


def _terminating_runner(env: _TerminatingEnv, *, record_failures: bool = True):
    recorded: list[int] = []
    ended: list[tuple[runner.EpisodeResult, bool]] = []
    simulation = object.__new__(runner.SimulationRunner)
    simulation.env = env
    simulation.app = SimpleNamespace(is_running=lambda: True, update=lambda: None)
    simulation.scene = SimpleNamespace(
        fresh_physics_per_episode=False,
        on_reset=lambda _env, _view: None,
        make_actuation=lambda _env, _view: SimpleNamespace(
            tensor=lambda: torch.zeros(1, 6), numpy=lambda: np.zeros((1, 6))
        ),
    )
    simulation._reset_env = lambda _seed=None: None
    simulation._view = SimpleNamespace(on_reset=lambda: None, on_step=lambda: None, invalidate=lambda: None, num_envs=1)
    simulation._sensor_windows = ()
    simulation._controls = {}
    simulation.keys = simulation.bus = simulation.publisher = None
    simulation.run_id = "run"
    simulation.seed = None
    simulation.workflow = SimpleNamespace(mode="policy")
    simulation.record_failures = record_failures
    simulation.recorder = SimpleNamespace(
        begin_episode=lambda *_args, **_kwargs: None,
        on_step=lambda _action, _view: recorded.append(env.steps),
        end_episode=lambda result, keep: ended.append((result, keep)),
    )
    return simulation, recorded, ended


def test_a_reset_the_workflow_does_not_end_on_ends_the_attempt(caplog) -> None:
    """An object_dropped reset used to let the demo run on into the next, re-randomized episode."""
    env = _TerminatingEnv(terminal_step=4)
    env.termination_manager = _TermManager(env, "object_dropped")
    simulation, recorded, ended = _terminating_runner(env)

    with caplog.at_level("INFO", logger="i4h_arena.runner"):
        result = simulation._run_episode(_KeepsRunning(env), episode=0, attempt=1)

    assert result.status is WorkflowStatus.ABORTED
    assert result.detail == (
        "IsaacLab reset the scene after termination object_dropped while the workflow was still running"
    )
    assert env.steps == 4  # no step is taken in the next episode
    assert recorded == [1, 2, 3]
    assert ended == [(result, True)]
    assert "IsaacLab reset the scene after termination object_dropped" in caplog.text


def test_after_a_success_reset_an_active_workflow_aborts_before_another_step(caplog) -> None:
    """A closing stage cannot advance physics in the next, randomized episode."""
    env = _TerminatingEnv(terminal_step=4)
    env.termination_manager = _TermManager(env, "success")
    simulation, recorded, ended = _terminating_runner(env)

    with caplog.at_level("INFO", logger="i4h_arena.runner"):
        result = simulation._run_episode(_KeepsRunning(env, succeed_after=3), episode=0, attempt=1)

    assert result.status is WorkflowStatus.ABORTED
    assert env.steps == 4
    assert recorded == [1, 2, 3]
    assert ended == [(result, True)]
    assert "termination success while the workflow was still running" in result.detail


def test_a_success_reset_alongside_another_termination_still_ends_the_attempt() -> None:
    env = _TerminatingEnv(terminal_step=2)
    env.termination_manager = _TermManager(env, "object_drop")
    env.termination_manager.get_term = lambda name: env.reset_buf.clone()  # success and object_drop both fired
    simulation, _recorded, _ended = _terminating_runner(env)

    result = simulation._run_episode(_KeepsRunning(env), episode=0, attempt=1)

    assert result.status is WorkflowStatus.ABORTED
    assert "termination success, object_drop" in result.detail


def test_ticks_that_wait_after_a_reset_do_not_end_the_attempt_early() -> None:
    """A remote task may tick WAITING before it reads success again; only the next step is too late."""
    env = _TerminatingEnv(terminal_step=4)
    env.termination_manager = _TermManager(env, "success")
    simulation, recorded, _ended = _terminating_runner(env)
    simulation.app.update = lambda: None
    rendered: list[int] = []
    original = runner._render_scene_only
    runner._render_scene_only = lambda _env, _app: rendered.append(env.steps)
    try:
        result = simulation._run_episode(_KeepsRunning(env, waiting=2, succeed_after=3), episode=0, attempt=1)
    finally:
        runner._render_scene_only = original

    assert result.succeeded
    assert rendered == [4, 4]
    assert env.steps == 4
    assert recorded == [1, 2, 3]


def test_a_finished_workflow_takes_no_extra_step_into_a_reset_episode() -> None:
    """``advance_on_success`` asks for one more step; not after IsaacLab already reset the scene."""
    env = _TerminatingEnv(terminal_step=4)
    engine = _UntilTerminated(env)
    original_tick = engine.tick

    def tick(ctx: TickContext) -> None:
        original_tick(ctx)
        engine.terminal_advance_requested = engine.status is WorkflowStatus.SUCCEEDED

    engine.tick = tick
    simulation, recorded, _ended = _terminating_runner(env)

    result = simulation._run_episode(engine, episode=0, attempt=1)

    assert result.succeeded
    assert env.steps == 4
    assert recorded == [1, 2, 3]


def test_an_operator_scene_reset_clears_a_pending_termination_reset() -> None:
    """Teleop's own reset restarts the attempt in a clean scene; that is not the next episode."""
    env = _TerminatingEnv(terminal_step=2)
    engine = _KeepsRunning(env)
    original_tick = engine.tick
    calls = {"ticks": 0}

    def tick(ctx: TickContext) -> None:
        calls["ticks"] += 1
        if calls["ticks"] == 3:  # the tick after the termination reset
            ctx.request_scene_reset()
            env.reset_buf[0] = False
            engine.ticks_since_reset = None
            return
        if calls["ticks"] == 6:
            engine.status = WorkflowStatus.SUCCEEDED
            return
        original_tick(ctx)

    engine.tick = tick
    simulation, recorded, _ended = _terminating_runner(env)
    simulation._reset_active_scene = lambda _engine, _ctx: None

    result = simulation._run_episode(engine, episode=0, attempt=1)

    assert result.succeeded
    assert env.steps == 4


def test_a_swallowed_step_failure_is_named_at_warning_level(caplog) -> None:
    """A real fault that coincides with Kit closing must still reach workflow.log (PR 275 review)."""
    simulation = _closing_runner(_closes_on_step(2))

    with caplog.at_level("WARNING", logger="i4h_arena.runner"):
        simulation._run_episode(_Engine(), episode=0, attempt=1)

    assert (
        "env.step raised ReferenceError('weakly-referenced object no longer exists') while Isaac Sim was closing"
        in caplog.text
    )


class _BoundaryEnv(_TerminatingEnv):
    """Reject any physics step past a reset until an explicit reset starts fresh."""

    def __init__(self, terminal_step=3, *, terms=True):
        super().__init__(terminal_step)
        self.total_steps = 0
        self.epoch = 0
        if terms:
            self.termination_manager = SimpleNamespace(
                active_terms=("time_out", "success", "other_env"),
                get_term=lambda name: torch.tensor([name == "time_out", name == "other_env"]),
            )

    def reset(self, _seed=None):
        self.steps = 0
        self.reset_buf[0] = False
        self.epoch += 1

    def step(self, command):
        assert not self.reset_buf[0], "physics advanced after an automatic reset"
        super().step(command)
        self.total_steps += 1
        if self.reset_buf[0]:
            self.epoch += 1

    @property
    def pixel(self):
        return self.epoch * 10 + self.steps


def _boundary_runner(tmp_path, *, record_failures=False, terminal_step=3, terms=True):
    from i4h_arena.recording.hdf5 import EpisodeRecorder

    simulation = _closing_runner(lambda _app: None, record_failures=record_failures)
    env = _BoundaryEnv(terminal_step, terms=terms)
    simulation.env = env
    simulation._reset_env = env.reset
    simulation._view = SimpleNamespace(
        on_reset=lambda: None,
        on_step=lambda: None,
        invalidate=lambda: None,
        num_envs=1,
        joints=lambda: SimpleNamespace(pos=np.full((1, 3), env.pixel)),
        camera=lambda _name: SimpleNamespace(to_array=lambda: np.full((2, 2, 3), env.pixel, dtype=np.uint8)),
    )
    simulation.scene.make_actuation = lambda _env, _view: SimpleNamespace(
        tensor=lambda: torch.full((1, 3), env.steps + 1.0),
        numpy=lambda: np.full((1, 3), env.steps),
    )
    simulation.workflow = SimpleNamespace(name="boundary", mode="policy", scene="fake")
    path = tmp_path / "recording.hdf5"
    simulation.recorder = EpisodeRecorder(path, workflow=simulation.workflow, cameras=("room",))
    return simulation, path


def _boundary_graph(env, *, outcome, waiting=0, final_advance=False, operator_reset=False):
    from i4h_engine.graph import TaskGraph, node
    from i4h_engine.status import Status
    from i4h_engine.task import Task

    class BoundaryTask(Task):
        advance_on_success = final_advance

        def __init__(self):
            super().__init__(name="drive")
            self.waited = 0
            self.operator_reset = False

        def tick(self, ctx):
            if env.reset_buf[0]:
                if operator_reset and not self.operator_reset:
                    self.operator_reset = True
                    ctx.request_scene_reset()
                    return Status.RUNNING
                if self.waited < waiting:
                    self.waited += 1
                    return Status.WAITING
                return Status(outcome)
            return Status.RUNNING

    return TaskGraph().flow(node(BoundaryTask()))


@pytest.mark.parametrize("record_failures", [False, True])
@pytest.mark.parametrize("waiting", [0, 3])
@pytest.mark.parametrize("outcome", ["success", "failure", "running"])
@pytest.mark.parametrize("final_advance", [False, True])
def test_automatic_reset_boundaries_in_saved_recording(
    tmp_path, monkeypatch, record_failures, waiting, outcome, final_advance
):
    import h5py

    from i4h_engine.executor import Engine

    simulation, path = _boundary_runner(tmp_path, record_failures=record_failures)
    renders = []
    monkeypatch.setattr(runner, "_render_scene_only", lambda *_args: renders.append(True))
    graph = _boundary_graph(simulation.env, outcome=outcome, waiting=waiting, final_advance=final_advance)
    engine = Engine(graph, on_event=simulation._on_event, max_steps=10)
    try:
        result = simulation._run_episode(engine, 0, 1)
    finally:
        simulation.recorder.close()
    expected = {
        "success": WorkflowStatus.SUCCEEDED,
        "failure": WorkflowStatus.FAILED,
        "running": WorkflowStatus.ABORTED,
    }
    assert result.status is expected[outcome]
    assert simulation.env.total_steps == 3
    assert len(renders) == waiting
    if outcome == "running":
        assert (
            result.detail == "IsaacLab reset the scene after termination time_out while the workflow was still running"
        )
    with h5py.File(path, "r") as handle:
        if outcome == "success" or record_failures:
            demo = handle["data/demo_0"]
            assert demo.attrs["status"] == result.status.value
            assert bool(demo.attrs["success"]) == result.succeeded
            np.testing.assert_array_equal(demo["actions"][:, 0], [1, 2])
            np.testing.assert_array_equal(demo["obs/joint_pos"][:, 0], [11, 12])
            np.testing.assert_array_equal(demo["obs/room"][:, 0, 0, 0], [11, 12])
        else:
            assert list(handle["data"]) == []


def test_operator_reset_clears_pending_reset_and_restarts_recording(tmp_path):
    import h5py

    from i4h_engine.executor import Engine

    simulation, path = _boundary_runner(tmp_path)
    graph = _boundary_graph(simulation.env, outcome="success", operator_reset=True)
    try:
        result = simulation._run_episode(Engine(graph, on_event=simulation._on_event), 0, 1)
    finally:
        simulation.recorder.close()
    assert result.succeeded
    assert simulation.env.total_steps == 6
    with h5py.File(path, "r") as handle:
        # Samples from before the operator reset have been discarded.
        np.testing.assert_array_equal(handle["data/demo_0/obs/room"][:, 0, 0, 0], [31, 32])


@pytest.mark.parametrize("record_failures", [False, True])
@pytest.mark.parametrize("terminal_step", [1, 3])
def test_reset_abort_obeys_retries_and_empty_recording_policy(tmp_path, record_failures, terminal_step):
    import h5py

    simulation, path = _boundary_runner(
        tmp_path, record_failures=record_failures, terminal_step=terminal_step, terms=False
    )
    simulation.workflow.graph = _boundary_graph(simulation.env, outcome="running")
    simulation.episodes = 2
    simulation.attempts = 2
    simulation.max_steps = 10
    try:
        summary = simulation.run()
    finally:
        simulation.recorder.close()
    assert [(result.index, result.attempt) for result in summary.episodes] == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert all(result.status is WorkflowStatus.ABORTED for result in summary.episodes)
    assert all(
        result.detail
        == "IsaacLab reset the scene after termination an unnamed term while the workflow was still running"
        for result in summary.episodes
    )
    assert simulation.env.total_steps == 4 * terminal_step
    with h5py.File(path, "r") as handle:
        assert len(handle["data"]) == (4 if record_failures and terminal_step > 1 else 0)


def test_engine_timeout_result_is_preserved_after_reset(tmp_path):
    from i4h_engine.executor import Engine

    simulation, _path = _boundary_runner(tmp_path)
    graph = _boundary_graph(simulation.env, outcome="running")
    try:
        result = simulation._run_episode(Engine(graph, max_steps=3), 0, 1)
    finally:
        simulation.recorder.close()
    assert result.status is WorkflowStatus.FAILED
    assert result.detail == "workflow exceeded max_steps=3"
    assert simulation.env.total_steps == 3


def test_terminal_advance_without_pending_reset_still_applies_final_action(tmp_path):
    import h5py

    from i4h_engine.executor import Engine
    from i4h_engine.graph import TaskGraph, node
    from i4h_engine.status import Status
    from i4h_engine.task import Task

    class FinalAction(Task):
        advance_on_success = True

        def tick(self, ctx):
            return Status.SUCCESS if ctx.step == 1 else Status.RUNNING

    simulation, path = _boundary_runner(tmp_path)
    graph = TaskGraph().flow(node(FinalAction(name="replay")))
    try:
        result = simulation._run_episode(Engine(graph), 0, 1)
    finally:
        simulation.recorder.close()
    assert result.succeeded
    assert simulation.env.total_steps == 2
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["data/demo_0/actions"][:, 0], [1, 2])


@pytest.mark.parametrize("term", ["success", "object_dropped"])
def test_scissor_final_home_consumes_success_without_crossing_the_reset(tmp_path, term):
    import h5py

    from i4h_engine.executor import Engine
    from i4h_engine.graph import TaskGraph
    from i4h_engine.loader import resolve_workflow

    simulation, path = _boundary_runner(tmp_path, record_failures=True)
    env = simulation.env
    env.termination_manager = _TermManager(env, term)
    simulation._view.termination = lambda name: env.termination_manager.get_term(name).numpy()
    simulation._view.home_joints = lambda: np.zeros((1, 3))
    make_actuation = simulation.scene.make_actuation

    def actuation(*args):
        act = make_actuation(*args)
        act.set_joint_targets = lambda *_args: None
        act.set_gripper = lambda *_args: None
        return act

    simulation.scene.make_actuation = actuation
    graph = resolve_workflow("scissor_pick_and_place", "rule-based").graph
    home = next(node for node in graph.nodes if node.id == "home")
    try:
        result = simulation._run_episode(Engine(TaskGraph().flow(home)), 0, 1)
    finally:
        simulation.recorder.close()

    assert result.status is (WorkflowStatus.SUCCEEDED if term == "success" else WorkflowStatus.ABORTED)
    assert env.total_steps == 3
    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        assert demo.attrs["status"] == result.status.value
        assert bool(demo.attrs["success"]) == (term == "success")
        np.testing.assert_array_equal(demo["obs/room"][:, 0, 0, 0], [11, 12])


def test_three_reset_episodes_commit_three_new_demos_when_appending(tmp_path):
    import h5py

    from i4h_arena.adapters.scene_view import ArenaSceneView
    from i4h_engine.graph import TaskGraph, node
    from i4h_engine.status import Status
    from i4h_engine.task import Task

    simulation, path = _boundary_runner(tmp_path)
    env = simulation.env
    env.num_envs = 1
    # IsaacLab retains the terminating step's terms across explicit resets.
    retained = np.array([True])
    env.termination_manager = SimpleNamespace(active_terms=("success",), get_term=lambda name: retained)
    step = env.step

    def step_and_terminate(command):
        step(command)
        retained[:] = bool(env.reset_buf[0])

    env.step = step_and_terminate
    view = ArenaSceneView(env)
    view.joints = simulation._view.joints
    view.camera = simulation._view.camera
    simulation._view = view
    simulation.initial_state = None
    # Restore the real runner reset path; the generic fixture uses a simpler fake.
    simulation._reset_env = runner.SimulationRunner._reset_env.__get__(simulation)

    class SucceedsOnTerm(Task):
        def tick(self, ctx):
            return Status.SUCCESS if ctx.scene.termination("success").any() else Status.RUNNING

    simulation.workflow.graph = TaskGraph().flow(node(SucceedsOnTerm(name="drive")))
    simulation.episodes = 3
    simulation.attempts = 1
    simulation.max_steps = 10
    # This demo belongs to an earlier run and must not be included in the summary.
    simulation.recorder._data.create_group("demo_0")
    try:
        summary = simulation.run()
    finally:
        simulation.recorder.close()
    assert summary.succeeded == 3 and summary.saved == 3 and summary.complete
    assert all(episode.steps > 0 for episode in summary.episodes)
    with h5py.File(path) as file:
        assert len(file["data"]) == 4
        for name in ("demo_1", "demo_2", "demo_3"):
            assert file[f"data/{name}"].attrs["num_samples"] == 2


def test_empty_success_counts_as_success_without_claiming_a_saved_demo():
    result = runner.EpisodeResult(index=0, attempt=1, status=WorkflowStatus.SUCCEEDED, steps=0, detail="")
    summary = runner.RunSummary(run_id="test", workflow="test", mode="policy", requested=1, episodes=[result])
    assert summary.complete and summary.succeeded == 1 and summary.saved == 0
    assert "0 new demonstrations saved" in summary.render()


# -- a fresh PhysX scene for every episode (NVBug 6953999) ------------------


class _PhysxManager:
    """Stands in for ``isaaclab_physx.physics.PhysxManager``."""


class _NewtonManager:
    """Stands in for another backend's physics manager."""


def _kit_physx(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Install the Kit modules ``_rebuild_physx_scene`` imports."""
    simulation = SimpleNamespace(detach_stage=lambda: calls.append("detach_stage"))
    monkeypatch.setitem(sys.modules, "isaaclab_physx.physics", SimpleNamespace(PhysxManager=_PhysxManager))
    monkeypatch.setitem(sys.modules, "omni.physx", SimpleNamespace(get_physx_simulation_interface=lambda: simulation))


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

    assert runner._rebuild_physx_scene(_physics_env(_PhysxManager, calls))

    # Without the detach, play hangs on "Stage ... already attached".
    assert calls == ["stop", "detach_stage", "play", "update:0.005"]


def test_rebuild_leaves_another_physics_backend_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _kit_physx(monkeypatch, calls)

    assert not runner._rebuild_physx_scene(_physics_env(_NewtonManager, calls))
    assert calls == []


def test_rebuild_is_a_no_op_when_kit_has_no_physx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "isaaclab_physx.physics", None)  # import raises ImportError

    assert not runner._rebuild_physx_scene(_physics_env(_PhysxManager, calls))
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
        runner._rebuild_physx_scene(env)

    assert str(raised.value.__cause__) == f"{failing} failed"
    # A play after a failed detach could hang on the still-attached stage.
    assert calls == ["stop", "detach_stage"] if failing == "detach_stage" else ["stop", "detach_stage", "play"]


class _StepsThenSucceeds(_Engine):
    """A workflow that takes ``steps`` physics steps and then succeeds."""

    def __init__(self, steps: int) -> None:
        super().__init__()
        self._steps = steps

    def start(self, _ctx: TickContext) -> None:
        self.status, self.step = runner.WorkflowStatus.RUNNING, 0

    def tick(self, _ctx: TickContext) -> None:
        self.step += 1
        if self.step > self._steps:
            self.status = runner.WorkflowStatus.SUCCEEDED


def _episode_runner(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    *,
    mode: str = "policy",
    task: str = "gr00t_n16/locomanip_push_cart",
    fresh: bool = True,
) -> runner.SimulationRunner:
    actuation = SimpleNamespace(tensor=lambda: torch.zeros(1, 3), numpy=lambda: np.zeros((1, 3)))
    simulation = _closing_runner(lambda _app: calls.append("step"))
    simulation.scene = SimpleNamespace(
        fresh_physics_per_episode=fresh,
        on_reset=lambda _env, _view: None,
        make_actuation=lambda _env, _view: actuation,
    )
    simulation._reset_env = lambda seed=None: calls.append("reset")
    simulation.workflow = SimpleNamespace(mode=mode, graph=[SimpleNamespace(task_id=task)])
    simulation.recorder = None

    def rebuild(env: object) -> bool:
        assert env is simulation.env
        calls.append("rebuild")
        return True

    monkeypatch.setattr(runner, "_rebuild_physx_scene", rebuild)
    return simulation


def test_a_retry_starts_from_a_rebuilt_physx_scene_like_a_fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """NVBug 6953999: a push-cart demo recorded on attempt 2 drifted 0.1 rad on replay."""
    calls: list[str] = []
    simulation = _episode_runner(monkeypatch, calls)
    engine = _StepsThenSucceeds(steps=2)

    for attempt in (1, 2, 3):
        simulation._run_episode(engine, episode=0, attempt=attempt)

    # The first episode in the process is left alone; every later one gets a
    # new PhysX scene before its reset writes the initial state into it.
    assert calls == ["reset", "step", "step"] + ["rebuild", "reset", "step", "step"] * 2


def test_an_episode_that_took_no_physics_step_leaves_the_physx_scene_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    simulation = _episode_runner(monkeypatch, calls)

    simulation._run_episode(_StepsThenSucceeds(steps=2), episode=0, attempt=1)
    simulation._run_episode(_StepsThenSucceeds(steps=0), episode=1, attempt=1)
    simulation._run_episode(_StepsThenSucceeds(steps=0), episode=2, attempt=1)

    assert calls == ["reset", "step", "step", "rebuild", "reset", "reset"]


@pytest.mark.parametrize(
    ("mode", "task", "fresh"),
    [
        ("policy", "gr00t_n16/locomanip_push_cart", False),
        ("teleop", "teleop/drive", True),
        # A live operator keeps the scene under a mode named after its device too.
        ("xr", "teleop/xr", True),
    ],
)
def test_scenes_that_do_not_opt_in_and_teleop_keep_their_physx_scene(
    monkeypatch: pytest.MonkeyPatch, mode: str, task: str, fresh: bool
) -> None:
    calls: list[str] = []
    simulation = _episode_runner(monkeypatch, calls, mode=mode, task=task, fresh=fresh)

    for attempt in (1, 2):
        simulation._run_episode(_StepsThenSucceeds(steps=1), episode=0, attempt=attempt)

    assert calls == ["reset", "step"] * 2
