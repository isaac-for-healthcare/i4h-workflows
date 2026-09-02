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


@dataclass(slots=True)
class EpisodeResult:
    index: int
    attempt: int
    status: WorkflowStatus
    steps: int
    detail: str = ""
    segments: tuple[tuple[str, str, int, int], ...] = ()

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
    def saved(self) -> int:
        return sum(1 for e in self.episodes if e.succeeded)

    @property
    def complete(self) -> bool:
        return self.saved == self.requested

    def render(self) -> str:
        attempts = len(self.episodes)
        return (
            f"{self.workflow} [{self.mode}] run={self.run_id}: "
            f"{self.saved}/{self.requested} episodes succeeded ({attempts} attempts)"
        )


class SimulationRunner:
    """Drives one workflow against one scene for N episodes."""

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
        self.record_failures = record_failures
        self.seed = seed
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
        # Idle is a render-only inspection mode: its ticks do not consume the
        # scene's simulation-step budget because no physics step is taken.
        engine = Engine(
            self.workflow.graph,
            workflow_name=self.workflow.name,
            on_event=self._on_event,
            max_steps=None if self.workflow.mode in {"idle", "teleop"} else self.max_steps,
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

    def _run_episode(self, engine: Engine, episode: int, attempt: int) -> EpisodeResult:
        if episode == 0 and attempt == 1 and self.seed is not None:
            self.env.reset(seed=self.seed)
        else:
            self.env.reset()
        self._view.invalidate()
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
            self.recorder.begin_episode(episode, attempt)

        while not engine.status.is_terminal and self.app.is_running():
            engine.tick(ctx)
            if ctx.consume_scene_reset():
                self._reset_active_scene(engine, ctx)
                continue
            terminal_advance = engine.status is WorkflowStatus.SUCCEEDED and engine.terminal_advance_requested
            if engine.status.is_terminal and not terminal_advance:
                break
            if not engine.advance_requested:
                self.app.update()
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
                    self.env.step(command)
                self._view.invalidate()
            self._update_sensor_windows()
            if self.recorder is not None:
                self.recorder.on_step(self._actuation.numpy(), self._view)
            if self.publisher is not None:
                active = engine.active_nodes
                self.publisher.publish(self._view, node=active[0] if active else "", episode_index=episode)

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
            self.recorder.end_episode(result, keep=keep)
        return result

    def _reset_active_scene(self, engine: Engine, ctx: TickContext) -> None:
        """Reset simulator state without tearing down the active teleop node."""
        self.env.reset()
        self._view.invalidate()
        self.scene.on_reset(self.env, self._view)
        self._actuation = self.scene.make_actuation(self.env, self._view)
        engine.replace_actuation(ctx, self._actuation)
        if self.recorder is not None:
            active = engine.active_nodes
            node = active[0] if active else ""
            task_id = engine.states[node].node.task_id if node else ""
            self.recorder.restart_episode(node=node, task_id=task_id)
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
