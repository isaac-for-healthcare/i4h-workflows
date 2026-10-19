# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One simulation loop for every scene, workflow, and mode.

The mode is resolved before the loop starts::

    engine.tick(ctx)                  active nodes write into ctx.act
    env.step(actuation.tensor())  the runner, and only the runner, advances time
    view.invalidate()             per-tick read cache is now stale

Everything else in this module is bookkeeping around that: episode retries,
event fan-out to the log / bus / recorder, and teardown.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from i4h_arena.recording.scene_state import capture_scene_state, restore_scene_state
from i4h_arena.scenes.base import Scene
from i4h_common.bus.base import Bus
from i4h_common.bus.keys import Keys
from i4h_common.bus.messages import WorkflowEventMsg, encode
from i4h_engine.events import WorkflowEvent
from i4h_engine.executor import Engine
from i4h_engine.loader import ResolvedWorkflow
from i4h_engine.status import WorkflowStatus
from i4h_engine.task import TickContext

logger = logging.getLogger("i4h_arena.runner")
_ACTIVE_RUNNER: SimulationRunner | None = None


def active_runner() -> SimulationRunner:
    """Return the runner currently serving the simulator process.

    The Python server executes diagnostics inside this process, so this is the
    narrow replacement for the retired custom bridge context.
    """

    if _ACTIVE_RUNNER is None:
        raise RuntimeError("no active SimulationRunner; launch a workflow with ./run.sh <workflow> --live")
    return _ACTIVE_RUNNER


def _kit_settings_manager() -> Any:
    """Resolve Kit settings lazily so importing the runner stays simulator-free."""
    from isaaclab.app.settings_manager import get_settings_manager

    return get_settings_manager()


def _reset_during_step(env: Any) -> bool:
    """Whether the last ``env.step`` reset the recorded environment (env 0).

    ``ManagerBasedRLEnv.step`` resets a terminated environment before it
    returns, so the scene, joints and camera images it leaves behind already
    belong to the next episode. A workflow whose success is a termination term
    (the liver scan's is) therefore ends on such a step every time it succeeds.
    That step is not recorded; ``i4h_common.episode`` documents what this means
    for a recording.
    """
    reset_buf = getattr(getattr(env, "unwrapped", env), "reset_buf", None)
    if reset_buf is None or len(reset_buf) == 0:
        return False
    return bool(reset_buf[0])


def _terms_that_reset(env: Any) -> tuple[str, ...]:
    """The termination terms that ended env 0's episode on the last ``env.step``.

    ``TerminationManager.get_term`` keeps the values of the step that triggered
    the reset, so they still name the cause after the environment was reset.
    """
    manager = getattr(getattr(env, "unwrapped", env), "termination_manager", None)
    fired: list[str] = []
    for name in getattr(manager, "active_terms", ()) or ():
        value = manager.get_term(name)
        if len(value) and bool(value[0]):
            fired.append(str(name))
    return tuple(fired)


def _reset_detail(terms: tuple[str, ...]) -> str:
    named = ", ".join(terms) or "an unnamed term"
    return f"IsaacLab reset the scene after termination {named} while the workflow was still running"


def _render_scene_only(env: Any, app: Any) -> None:
    """Refresh the scene and UI while preventing Kit from advancing physics."""
    sim = getattr(getattr(env, "unwrapped", env), "sim", None)
    render = getattr(sim, "render", None)
    if not callable(render):
        raise TypeError("idle scene inspection requires env.unwrapped.sim.render()")
    render()

    # SimulationContext.render() updates standalone visualizers, but an Arena
    # scene may have no Kit visualizer to pump the window. Pump Kit explicitly
    # so viewport orbit/pan/zoom remains interactive, with simulation playback
    # disabled for the update. This preserves the reset pose without relying on
    # an active WBC command.
    settings = _kit_settings_manager()
    play_simulations = "/app/player/playSimulations"
    previous = bool(settings.get(play_simulations, True))
    settings.set_bool(play_simulations, False)
    try:
        app.update()
    finally:
        settings.set_bool(play_simulations, previous)


def _rebuild_physx_scene(env: Any) -> bool:
    """Replace the PhysX scene with one parsed afresh from the stage (NVBug 6953999).

    A reset writes back every state IsaacLab exposes, yet the GPU PhysX scene
    keeps internal state across it (contact or constraint ordering is the
    suspect). An episode that follows others in one process then drifts from
    the same episode in a fresh process within a few steps, and a balancing G1
    amplifies that to tenths of a radian, so a retry's recording does not
    replay.

    Stopping the timeline invalidates the PhysX views and IsaacLab's assets and
    sensors. Detaching the stage makes the next play parse it into a new PhysX
    scene; without it, play hangs on "Stage ... already attached". Play then
    recreates the views and re-initializes every asset and sensor, as the
    env's construction did. USD keeps everything authored on the stage,
    ``prestartup`` events included; IsaacLab ``startup`` events, which ran once
    against the old scene, do not run again.

    Returns ``False`` and leaves the simulation alone when it does not run on
    Kit PhysX.
    """
    unwrapped = getattr(env, "unwrapped", env)
    sim = unwrapped.sim
    try:
        from isaaclab_physx.physics import PhysxManager  # noqa: PLC0415 - needs a running Kit
    except ImportError:  # no Kit PhysX to rebuild
        return False
    manager = getattr(sim, "physics_manager", None)
    if isinstance(manager, str):
        # IsaacLab keeps the configured manager as a lazily resolved "module:Class" string.
        from isaaclab.utils.string import string_to_callable  # noqa: PLC0415

        manager = string_to_callable(manager)
    if not (isinstance(manager, type) and issubclass(manager, PhysxManager)):
        return False
    from omni.physx import get_physx_simulation_interface  # noqa: PLC0415 - needs a running Kit

    sim.stop()
    get_physx_simulation_interface().detach_stage()
    sim.play()
    unwrapped.scene.update(dt=unwrapped.physics_dt)
    return True


@dataclass(slots=True)
class EpisodeResult:
    index: int
    attempt: int
    status: WorkflowStatus
    steps: int
    detail: str = ""
    segments: tuple[tuple[str, str, int, int], ...] = ()
    saved_demo: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is WorkflowStatus.SUCCEEDED


@dataclass(slots=True)
class RunSummary:
    run_id: str
    workflow: str
    mode: str
    requested: int
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(1 for e in self.episodes if e.succeeded)

    @property
    def saved(self) -> int:
        return sum(e.saved_demo is not None for e in self.episodes)

    @property
    def complete(self) -> bool:
        return self.succeeded == self.requested

    def render(self) -> str:
        attempts = len(self.episodes)
        return (
            f"{self.workflow} [{self.mode}] run={self.run_id}: "
            f"{self.succeeded}/{self.requested} episodes succeeded ({attempts} attempts); "
            f"{self.saved} new demonstrations saved"
        )


class SimulationRunner:
    """Drives one workflow against one scene for N episodes."""

    #: Whether physics stepped since the PhysX scene was built, so that the
    #: next episode would inherit another's PhysX state. Cleared by a rebuild.
    _physics_stepped: bool = False

    def __init__(
        self,
        *,
        scene: Scene,
        workflow: ResolvedWorkflow,
        env: Any,
        app: Any,
        bus: Bus | None = None,
        keys: Keys | None = None,
        recorder: Any | None = None,
        publisher: Any | None = None,
        episodes: int = 1,
        attempts: int = 1,
        max_steps: int | None = None,
        record_failures: bool = False,
        seed: int | None = None,
        sensor_views: tuple[str, ...] = (),
        sensor_view_titles: dict[str, str] | None = None,
        sensor_view_outputs: dict[str, tuple[tuple[str, str], ...]] | None = None,
        sensor_view_keyboard_toggles: dict[str, dict[str, tuple[tuple[str, str], ...]]] | None = None,
        sensor_view_projection_presets: dict[str, tuple[tuple[str, str, float], ...]] | None = None,
        sensor_view_projection_defaults: dict[str, int] | None = None,
        sensor_view_appearances: dict[str, tuple[tuple[str, str], ...]] | None = None,
        sensor_view_display_controls: dict[str, tuple[Any, ...]] | None = None,
        sensor_view_sliders: dict[str, tuple[Any, ...]] | None = None,
        sensor_view_readouts: dict[str, Any] | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self.scene = scene
        self.workflow = workflow
        self.env = env
        self.app = app
        self.bus = bus
        self.keys = keys or Keys(workflow.name)
        self.recorder = recorder
        self.publisher = publisher
        self.episodes = max(1, episodes)
        self.attempts = max(1, attempts)
        self.max_steps = max_steps or workflow.max_steps or scene.spec.max_steps
        # Kept apart from the resolved budget above, which cannot say whether a
        # cap was asked for or merely inherited. Teleop needs that distinction.
        self._requested_max_steps = max_steps
        self.record_failures = record_failures
        self.seed = seed
        #: Recorded scene state every episode starts from instead of a
        #: randomized reset. Replay sets it so the recorded actions meet the
        #: objects where they were when the episode was recorded.
        self.initial_state = initial_state
        self.run_id = uuid.uuid4().hex[:8]
        self._view = scene.make_view(env)
        self._controls: dict[str, float] = {}
        self._sensor_windows = self._make_sensor_windows(
            sensor_views,
            sensor_view_titles or {},
            sensor_view_outputs or {},
            sensor_view_keyboard_toggles or {},
            sensor_view_projection_presets or {},
            sensor_view_projection_defaults or {},
            sensor_view_appearances or {},
            sensor_view_display_controls or {},
            sensor_view_sliders or {},
            sensor_view_readouts or {},
        )
        # Construct only after the episode reset, when sensor-backed TCP data
        # and randomized state belong to the same reset.
        self._actuation = None

    # -- events ----------------------------------------------------------
    def _on_event(self, event: WorkflowEvent) -> None:
        logger.info("%s", event)
        if self.recorder is not None:
            self.recorder.on_event(event)
        if self.bus is not None:
            self.bus.publish(
                self.keys.workflow_events,
                encode(
                    WorkflowEventMsg(
                        run_id=event.run_id,
                        workflow=event.workflow,
                        episode_index=event.episode_index,
                        step=event.step,
                        event=event.kind,
                        node=event.node,
                        task_id=event.task_id,
                        outputs=event.outputs,
                        detail=event.detail,
                    )
                ),
            )

    # -- the loop --------------------------------------------------------
    def _engine_max_steps(self) -> int | None:
        """Step budget the engine enforces, or ``None`` to run until the graph stops.

        Idle is a render-only inspection mode: its ticks take no physics step,
        so they cannot consume the scene's budget and a cap would only limit how
        long someone may look at the scene.

        Teleop does step, and is left uncapped so that a demonstration is not
        cut off underneath the person giving it. An explicit ``--episode-steps``
        is honoured though, because an attempt that will not succeed otherwise
        has no way to end: teleop runs ``until=success`` with no time limit, so
        without a cap the episode is never finalized, nothing reaches
        ``/data``, and ``--record-failures`` has no failed episode to write.
        """
        if self.workflow.mode == "idle":
            return None
        if self.workflow.mode == "teleop":
            return self._requested_max_steps
        return self.max_steps

    def run(self) -> RunSummary:
        global _ACTIVE_RUNNER
        if _ACTIVE_RUNNER is not None:
            raise RuntimeError("another SimulationRunner is already active")
        _ACTIVE_RUNNER = self
        summary = RunSummary(
            run_id=self.run_id,
            workflow=self.workflow.name,
            mode=self.workflow.mode,
            requested=self.episodes,
        )
        engine = Engine(
            self.workflow.graph,
            workflow_name=self.workflow.name,
            on_event=self._on_event,
            max_steps=self._engine_max_steps(),
        )

        try:
            episode = 0
            while episode < self.episodes and self.app.is_running():
                for attempt in range(1, self.attempts + 1):
                    if not self.app.is_running():
                        break
                    result = self._run_episode(engine, episode, attempt)
                    summary.episodes.append(result)
                    if result.succeeded or attempt == self.attempts:
                        break
                    logger.info("episode %s attempt %s failed (%s); retrying", episode, attempt, result.detail)
                episode += 1

            logger.info("%s", summary.render())
            return summary
        finally:
            for window in self._sensor_windows:
                window.close()
            _ACTIVE_RUNNER = None

    def _reset_env(self, seed: int | None = None) -> dict[str, Any] | None:
        """Reset the env and return the state it starts from, when recording.

        The state is taken before ``Scene.on_reset``, so a replay that restores
        it runs the same pre-roll the recording did.
        """
        if self.initial_state is not None:
            restore_scene_state(self.env, self.initial_state, seed=seed)
        elif seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()
        self._view.on_reset()
        return capture_scene_state(self.env) if self.recorder is not None else None

    def _fresh_physics_for_episode(self) -> None:
        """Start this episode in a PhysX scene no earlier episode stepped, when the scene asks for it.

        The first episode in a process already has one. Teleop keeps its
        scene: an operator's reset happens mid-episode, and rebuilding under
        a live teleop session is not validated.
        """
        if not self._physics_stepped or self.workflow.mode == "teleop" or not self.scene.fresh_physics_per_episode:
            return
        started = time.monotonic()
        if _rebuild_physx_scene(self.env):
            logger.info("rebuilt the PhysX scene for a fresh episode in %.1f s", time.monotonic() - started)
            self._physics_stepped = False

    def _run_episode(self, engine: Engine, episode: int, attempt: int) -> EpisodeResult:
        self._fresh_physics_for_episode()
        initial_state = self._reset_env(self.seed if episode == 0 and attempt == 1 else None)
        self.scene.on_reset(self.env, self._view)
        # Reset invalidates the command that was seeded when the runner was
        # constructed (and every previous episode's last command).  Rebuild it
        # from the freshly reset articulation/TCP before any Wait/Hold node can
        # repeat stale targets into the controller.
        self._actuation = self.scene.make_actuation(self.env, self._view)
        assert self._actuation is not None

        ctx = TickContext(
            keys=self.keys,
            scene=self._view,
            act=self._actuation,
            dt=self._step_dt(),
            bus=self.bus,
            run_id=self.run_id,
            episode_index=episode,
            attempt_index=attempt,
            controls=self._controls,
        )
        engine.start(ctx)
        self._update_sensor_windows()
        if self.recorder is not None:
            self.recorder.begin_episode(episode, attempt, initial_state=initial_state)

        #: Termination terms that made IsaacLab reset env 0 inside the last
        #: ``env.step`` and that the loop has not acted on yet.
        reset_by: tuple[str, ...] | None = None
        #: Cleared by such a reset: every later frame belongs to another episode.
        recording = True
        while not engine.status.is_terminal and self.app.is_running():
            engine.tick(ctx)
            if ctx.consume_scene_reset():
                self._reset_active_scene(engine, ctx)
                reset_by, recording = None, True
                continue
            terminal_advance = engine.status is WorkflowStatus.SUCCEEDED and engine.terminal_advance_requested
            if engine.status.is_terminal and (not terminal_advance or not recording):
                # A finished workflow may ask for one more step to apply its
                # last action, but not into an episode IsaacLab already reset.
                break
            if not engine.advance_requested:
                _render_scene_only(self.env, self.app)
                continue
            if self.workflow.mode == "idle":
                # Match the original scene-edit behavior: render the reset
                # state while the timeline stays frozen. A balancing humanoid
                # should not visibly sway merely because its scene is open.
                frame_started = time.monotonic()
                _render_scene_only(self.env, self.app)
                remaining = ctx.dt - (time.monotonic() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)
            else:
                if reset_by is not None:
                    # IsaacLab reset env 0 inside the last step and the workflow
                    # did not finish on the tick after it. Checked here, at the
                    # next step, rather than on the next tick: a remote task may
                    # tick WAITING for its backend before it reads success again.
                    detail = _reset_detail(reset_by)
                    logger.warning("episode %s attempt %s ends: %s", episode, attempt, detail)
                    engine.abort(ctx, detail=detail)
                    break
                # Task terms may lazily allocate persistent tensors during
                # ``env.step``. ``no_grad`` keeps those tensors mutable for the
                # next episode reset.
                with torch.no_grad():
                    command = self._actuation.tensor()
                    if not bool(torch.isfinite(command).all()):
                        # Physics turns a single non-finite target into NaN joint
                        # state and the articulation never recovers, so refuse it
                        # here where the culprit is still identifiable.
                        raise RuntimeError(
                            f"step {engine.step}: non-finite action {command.detach().cpu().numpy().tolist()}"
                        )
                    self._physics_stepped = True
                    try:
                        self.env.step(command)
                    except Exception as exc:
                        # ``env.step`` renders between its physics sub-steps, and
                        # that render is where Kit handles a closed window. Kit
                        # then releases the PhysX views the remaining sub-steps
                        # read (a ReferenceError from the articulation data). The
                        # attempt ends there rather than taking the run down with
                        # it; any failure while Kit is still running is a real one.
                        if self.app.is_running():
                            raise
                        # Still named at the default level: a real fault that
                        # coincides with, or causes, Kit stopping must not be
                        # reported only as "Isaac Sim was closed".
                        logger.warning("env.step raised %r while Isaac Sim was closing; ending the attempt", exc)
                        logger.debug("env.step failure while Isaac Sim was closing", exc_info=True)
                        break
                self._view.on_step()
                if _reset_during_step(self.env):
                    reset_by, recording = _terms_that_reset(self.env), False
                    logger.info(
                        "step %s: IsaacLab reset the scene after termination %s",
                        engine.step,
                        ", ".join(reset_by) or "(unnamed)",
                    )
            self._update_sensor_windows()
            if self.recorder is not None and recording:
                # A step that reset the environment leaves the next episode's
                # images and joints behind; recording them would end every
                # such demo on a frame from a different episode.
                self.recorder.on_step(self._actuation.numpy(), self._view)
            if self.publisher is not None:
                active = engine.active_nodes
                self.publisher.publish(self._view, node=active[0] if active else "", episode_index=episode)

        if not engine.status.is_terminal:
            # Only a closed simulator stops the loop above with the workflow
            # still running. Finish it, so the attempt is reported and recorded
            # (``--record-failures``) as aborted rather than as still running.
            logger.info("Isaac Sim was closed during episode %s attempt %s; ending it", episode, attempt)
            engine.abort(ctx, detail="Isaac Sim was closed")
        result = EpisodeResult(
            index=episode,
            attempt=attempt,
            status=engine.status,
            steps=engine.step,
            detail=engine.detail,
            segments=engine.segments,
        )
        if self.recorder is not None:
            keep = result.succeeded or self.record_failures
            result.saved_demo = self.recorder.end_episode(result, keep=keep)
        return result

    def _reset_active_scene(self, engine: Engine, ctx: TickContext) -> None:
        """Reset simulator state without tearing down the active teleop node."""
        initial_state = self._reset_env()
        self.scene.on_reset(self.env, self._view)
        self._actuation = self.scene.make_actuation(self.env, self._view)
        engine.replace_actuation(ctx, self._actuation)
        if self.recorder is not None:
            active = engine.active_nodes
            node = active[0] if active else ""
            task_id = engine.states[node].node.task_id if node else ""
            self.recorder.restart_episode(node=node, task_id=task_id, initial_state=initial_state)
        for window in self._sensor_windows:
            window.on_scene_reset()
        logger.info("teleop scene reset")

    def _make_sensor_windows(
        self,
        names: tuple[str, ...],
        titles: dict[str, str],
        outputs: dict[str, tuple[tuple[str, str], ...]],
        keyboard_toggles: dict[str, dict[str, tuple[tuple[str, str], ...]]],
        projection_presets: dict[str, tuple[tuple[str, str, float], ...]],
        projection_defaults: dict[str, int],
        appearances: dict[str, tuple[tuple[str, str], ...]],
        display_controls: dict[str, tuple[Any, ...]],
        sliders: dict[str, tuple[Any, ...]],
        readouts: dict[str, Any],
    ) -> tuple[Any, ...]:
        if not names:
            return ()
        from i4h_arena.ui.sensor_image_window import SensorImageWindow

        return tuple(
            SensorImageWindow(
                name=name,
                title=titles.get(name),
                view=self._view,
                outputs=outputs.get(name, (("RGB", "rgb"),)),
                keyboard_toggles=keyboard_toggles.get(name, {}),
                projection_presets=projection_presets.get(name, ()),
                projection_default_index=projection_defaults.get(name, 0),
                appearances=appearances.get(name, ()),
                display_controls=display_controls.get(name, ()),
                sliders=sliders.get(name, ()),
                readout=readouts.get(name),
                controls=self._controls,
            )
            for name in names
        )

    def _update_sensor_windows(self) -> None:
        for window in self._sensor_windows:
            window.update()

    def _step_dt(self) -> float:
        unwrapped = self.env.unwrapped
        for attribute in ("step_dt", "physics_dt"):
            value = getattr(unwrapped, attribute, None)
            if value:
                return float(value)
        sim_dt = getattr(getattr(unwrapped, "sim", None), "get_physics_dt", None)
        if callable(sim_dt):
            return float(sim_dt())
        return 1.0 / 60.0
