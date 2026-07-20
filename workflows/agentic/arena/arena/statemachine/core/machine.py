# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Object-oriented base for scripted Arena state-machine rollouts.

A :class:`StateMachine` drives one env with a scripted action sequence, without a
policy: it resets, then steps through a list of :class:`Stage`s (each produces an
action per step) to completion; with ``hold_last`` the final stage repeats to
fill the fixed timestep budget (the reference episode length). Episodes are
optionally recorded to HDF5. ``succeeded`` is evaluated
every step (which also advances any stateful predicate in lock-step) and success
is **latched** -- an episode is successful if the task condition (reached /
picked up / ...) held at any point during the rollout. The rollout is never
stopped or reset just because success was met, and reaching the timeout never by
itself counts as success; only a hard env termination (dropped object, abort)
ends it early.

Subclass and implement :meth:`build_stages` + :meth:`succeeded`; override
:meth:`initial_action`, :meth:`on_reset`, :meth:`log_fields`, or the
``default_max_steps`` / ``hold_last`` class attributes as needed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from arena.recording import close_recording, discard_episode, save_episode, save_successful_episode, setup_recording
from arena.statemachine.core.common import controller_should_abort, disabled_termination, should_stop
from common.utils import resolve_path

logger = logging.getLogger("arena")

# (env, step_within_stage, action_at_stage_start) -> action for this step.
StageAction = Callable[[Any, int, torch.Tensor], torch.Tensor]


@dataclass
class Stage:
    """One rollout phase: call ``action`` for ``steps`` environment steps."""

    name: str
    action: StageAction
    steps: int


class StateMachine(ABC):
    """Base class for a scripted, success-terminated rollout."""

    env_id: str
    default_max_steps: int = 600
    #: When True, hold the final stage's action until success or timeout (for
    #: servo-to-target plans whose scripted stages are shorter than the budget).
    hold_last: bool = False

    @abstractmethod
    def build_stages(self, env: Any) -> Sequence[Stage]:
        """Return the ordered stages for one episode (rebuilt after each reset)."""

    @abstractmethod
    def succeeded(self, env: Any) -> torch.Tensor:
        """Per-env success mask; evaluated every step (advances stateful predicates)."""

    def initial_action(self, env: Any) -> torch.Tensor:
        """Action applied before the first stage (and the lerp origin for stage 0)."""
        return torch.zeros(env.num_envs, int(env.action_space.shape[-1]), dtype=torch.float32, device=env.device)

    def on_reset(self, env: Any) -> None:
        """Hook after ``env.reset`` -- snapshot randomized state or pre-roll to a start pose."""

    def log_fields(self, env: Any) -> dict[str, Any]:
        """Extra key=value fields for the per-episode completion log line."""
        return {}

    # -- driver ----------------------------------------------------------
    def run(self, *, args: Any, env: Any, app: Any, controller: Any) -> None:
        max_steps = int(getattr(args, "max_timesteps", 0) or self.default_max_steps)
        episodes = int(getattr(args, "episodes", 1) or 1)
        record_to = resolve_path(getattr(args, "record_to", None), self.env_id)
        recorder = (
            _Recorder(
                env,
                app,
                controller,
                self.env_id,
                str(record_to),
                save_all=bool(getattr(args, "save_all_episodes", False)),
            )
            if record_to
            else None
        )
        if recorder is not None:
            recorder.setup()
        logger.info("state machine: env=%s episodes=%s max_timesteps=%s", self.env_id, episodes, max_steps)
        try:
            with ExitStack() as stack:
                # In filter-success recording, setup_recording already disabled these; otherwise
                # disable them ourselves so the env never auto-resets mid-rollout.
                if recorder is None or recorder.save_all:
                    for term in ("success", "time_out"):
                        stack.enter_context(disabled_termination(env, term))
                for episode in range(episodes):
                    if controller_should_abort(controller) or not app.is_running():
                        break
                    self._run_episode(env, app, controller, max_steps, episode, recorder)
        finally:
            if recorder is not None:
                recorder.close()

    def _run_episode(
        self, env: Any, app: Any, controller: Any, max_steps: int, episode: int, recorder: _Recorder | None
    ) -> None:
        env.reset()
        action = self.initial_action(env)
        self.on_reset(env)
        stages = self.build_stages(env)

        steps = 0
        accepted = False
        stopped = False
        for stage in stages:
            if steps >= max_steps or stopped:
                break
            budget = min(stage.steps, max_steps - steps)
            stage_start = action.clone()
            logger.info(
                "state machine stage: env=%s episode=%s stage=%s steps=%s", self.env_id, episode + 1, stage.name, budget
            )
            action, steps, hit, stopped = self._run_stage(
                env, app, controller, stage.action, budget, stage_start, steps
            )
            accepted = accepted or hit

        # Hold the final stage for the rest of the fixed budget (the reference episode length):
        # success is judged over the whole rollout, so we never stop early just because it was met.
        if self.hold_last and stages and not stopped and steps < max_steps:
            stage_start = action.clone()
            action, steps, hit, stopped = self._run_stage(
                env, app, controller, stages[-1].action, max_steps - steps, stage_start, steps
            )
            accepted = accepted or hit

        if accepted and controller is not None and not controller_should_abort(controller):
            controller.episode_completed()
        if recorder is not None:
            recorder.finish(accepted, {"env_id": self.env_id, "episode_index": episode, "success": accepted})
        extra = " ".join(f"{k}={v}" for k, v in self.log_fields(env).items())
        logger.info(
            "state machine complete: env=%s episode=%s steps=%s success=%s stopped_early=%s %s",
            self.env_id,
            episode + 1,
            steps,
            accepted,
            stopped,
            extra,
        )

    def _run_stage(
        self, env, app, controller, stage_action: StageAction, budget: int, stage_start: torch.Tensor, steps: int
    ):
        # Run the full budget: latch task success (never stop on it) so the fixed-length rollout
        # matches the reference episode; only a hard env termination (dropped object / abort) stops early.
        action = stage_start
        hit = False
        stopped = False
        for i in range(budget):
            action = stage_action(env, i, stage_start)
            with torch.no_grad():
                _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if bool(torch.as_tensor(self.succeeded(env)).any().item()):
                hit = True  # latch task success; keep running the fixed budget
            if should_stop(env, app, controller, terminated, truncated):
                stopped = True
                break
        return action, steps, hit, stopped


class _Recorder:
    """Thin wrapper over ``arena.recording`` for the state-machine path."""

    def __init__(self, env: Any, app: Any, controller: Any, env_id: str, path: str, *, save_all: bool) -> None:
        self.ctx = SimpleNamespace(env=env, controller=controller, simulation_app=app, env_id=env_id)
        self.path = path
        self.save_all = save_all

    def setup(self) -> None:
        setup_recording(self.ctx, self.path, streaming=True, filter_success=not self.save_all)

    def finish(self, accepted: bool, metadata: dict[str, Any]) -> None:
        if self.save_all:
            save_episode(self.ctx, metadata)
        elif accepted:
            save_successful_episode(self.ctx, metadata)
        else:
            discard_episode(self.ctx)

    def close(self) -> None:
        close_recording(self.ctx)
