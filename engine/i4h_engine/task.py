# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task contract.

**A task never calls** ``env.step``. It reads the world through ``ctx.scene``,
writes commands into ``ctx.act``, and returns ``RUNNING``/``SUCCESS``/
``FAILURE``. The runner owns stepping.

That one rule is what buys the design its two best properties: tasks are
unit-testable on CPU against a fake scene, and two branches of a workflow can be
active in the same tick without fighting over who advances time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from i4h_common.bus.base import Bus
from i4h_common.world import Actuation, SceneView
from i4h_engine.status import Status

In = TypeVar("In")
Out = TypeVar("Out")


@dataclass(frozen=True, slots=True)
class Empty:
    """Declared port set for a task with no inputs or no outputs."""


EMPTY = Empty()


@dataclass(slots=True)
class TickContext:
    """Everything a task may touch during one simulation step."""

    scene: SceneView
    act: Actuation
    dt: float = 1.0 / 60.0
    step: int = 0
    node_step: int = 0
    #: Present only when something crosses a process boundary; ``None`` in tests.
    bus: Bus | None = None
    #: Stable identity for this rollout, used in bus keys and recordings.
    run_id: str = ""
    #: Bus key builder for this rollout. Remote tasks must publish under the
    #: same namespace the backend subscribed to, which is the workflow name — not
    #: the run id, which changes every episode.
    keys: Any = None
    episode_index: int = 0
    #: One-based retry number for this episode. Zero outside the arena runner.
    attempt_index: int = 0
    #: Free-form scratch shared across nodes. Prefer typed data edges; this
    #: exists for genuinely cross-cutting state, and lint does not check it.
    blackboard: dict[str, Any] = field(default_factory=dict)
    #: Live operator controls owned by the runner (for example teleop speed).
    controls: dict[str, float] = field(default_factory=dict)
    _scene_reset_requested: bool = False

    @property
    def num_envs(self) -> int:
        return self.scene.num_envs

    def elapsed(self) -> float:
        """Seconds since the active node was entered."""
        return self.node_step * self.dt

    def request_scene_reset(self) -> None:
        """Ask the runner to reset the environment before its next physics step."""
        self._scene_reset_requested = True

    def consume_scene_reset(self) -> bool:
        """Return and clear the pending runner-owned reset request."""
        requested = self._scene_reset_requested
        self._scene_reset_requested = False
        return requested


class Task(ABC, Generic[In, Out]):
    """Base class for an in-process skill.

    Subclasses declare ``Inputs``/``Outputs`` dataclasses and implement
    :meth:`tick`. The manifest entry for the task must match those declarations;
    ``tests/test_manifest_drift.py`` asserts it.
    """

    #: Dataclass of input ports. Fields with defaults are optional (``?`` in TOML).
    Inputs: ClassVar[type] = Empty
    #: Dataclass of output ports.
    Outputs: ClassVar[type] = Empty
    #: Wall-clock budget; the engine fails the node when exceeded. ``None`` = unbounded.
    timeout_s: ClassVar[float | None] = None
    #: Apply this tick's command even when it is also the task's successful
    #: terminal tick. Most tasks succeed from already-observed world state and
    #: leave this false; action playback needs the final command to reach sim.
    advance_on_success: ClassVar[bool] = False

    # -- registry metadata -------------------------------------------------
    # Behavior-derived metadata stays here; author-facing summary/prompt live
    # in the task manifest for every task kind.
    #: What the scene must provide, matched against its `provides` by workflow-lint.
    requires: ClassVar[dict[str, object]] = {}
    #: Optional pre/post conditions — the hook for automatic workflow composition.
    precondition: ClassVar[dict[str, object]] = {}
    postcondition: ClassVar[dict[str, object]] = {}

    def __init__(self, *, name: str | None = None) -> None:
        # The module stem, which is also the task name discovery assigns —
        # one rule for both rather than a second CamelCase-to-snake converter.
        self._name = name or type(self).__module__.rpartition(".")[2]

    @property
    def name(self) -> str:
        """Default node id when this task is placed in a workflow."""
        return self._name

    def on_enter(self, ctx: TickContext, inputs: In) -> None:
        """Called once when the node becomes active, with its resolved inputs."""

    @abstractmethod
    def tick(self, ctx: TickContext) -> Status:
        """Advance one simulation step. Write into ``ctx.act``; do not step the sim."""

    def on_exit(self, ctx: TickContext) -> Out:
        """Called once on SUCCESS. Return this node's outputs."""
        return self.Outputs()  # type: ignore[return-value]

    def on_abort(self, ctx: TickContext) -> None:
        """Called when the workflow tears the node down without it succeeding."""

    def describe(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    # Ergonomics: a bare task instance can start a chain, so
    # `Locate("scissors") >> Grasp()` works without wrapping.
    def __rshift__(self, other: Any) -> Any:
        from i4h_engine.graph import node  # noqa: PLC0415 - avoids an import cycle

        return node(self) >> other
