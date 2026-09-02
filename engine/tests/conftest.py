# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal fakes: enough world to exercise the engine, no simulator.

These deliberately live in the test tree rather than in ``i4h_tasks.basic.testing``
so ``engine`` stays dependency-free — it must not need the skill library
in order to be tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from i4h_common.types import CameraFrame, JointState, ObjectState, Pose
from i4h_common.world import UnsupportedActuation
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


class FakeScene:
    """A world of static object poses and one 6-DOF arm."""

    def __init__(self, *, num_envs: int = 1, dof: int = 6, objects: dict[str, Pose] | None = None) -> None:
        self._num_envs = num_envs
        self._dof = dof
        self._objects = objects or {}
        self.joint_pos = np.zeros((num_envs, dof), dtype=np.float32)
        self.home = np.zeros((num_envs, dof), dtype=np.float32)
        self.contacts: set[tuple[str, str]] = set()
        self.tcp_pose = Pose.identity(num_envs)
        self.gripper = np.zeros(num_envs, dtype=np.float32)
        self.terminations: dict[str, np.ndarray] = {}

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def objects(self) -> tuple[str, ...]:
        return tuple(self._objects)

    @property
    def robots(self) -> tuple[str, ...]:
        return ("robot",)

    def object(self, name: str) -> ObjectState:
        if name not in self._objects:
            raise KeyError(f"no object {name!r}; have {list(self._objects)}")
        return ObjectState(name, self._objects[name], np.zeros((self._num_envs, 3), dtype=np.float32))

    def joints(self, robot: str = "robot") -> JointState:
        return JointState(
            pos=self.joint_pos,
            vel=np.zeros_like(self.joint_pos),
            names=tuple(f"j{i}" for i in range(self._dof)),
        )

    def home_joints(self, robot: str = "robot") -> np.ndarray:
        return self.home

    def tcp(self, robot: str = "robot") -> Pose:
        return self.tcp_pose

    def gripper_width(self, robot: str = "robot") -> np.ndarray:
        return self.gripper

    def contact(self, a: str, b: str) -> np.ndarray:
        touching = (a, b) in self.contacts or (b, a) in self.contacts
        return np.full(self._num_envs, touching, dtype=bool)

    def camera(self, name: str) -> CameraFrame | None:
        return None

    def termination(self, name: str) -> np.ndarray:
        return self.terminations.get(name, np.zeros(self._num_envs, dtype=bool))


@dataclass
class FakeActuation:
    """Records what was written, so tests can assert on commands."""

    dof: int = 6
    action_space: str = "joint_position"
    joint_targets: dict[str, Any] = field(default_factory=dict)
    gripper_cmd: dict[str, Any] = field(default_factory=dict)
    ee_targets: dict[str, Any] = field(default_factory=dict)
    ee_deltas: dict[str, Any] = field(default_factory=dict)
    raw_actions: dict[str, Any] = field(default_factory=dict)
    holds: list[str] = field(default_factory=list)

    def set_joint_targets(self, values: Any, robot: str = "robot") -> None:
        self.joint_targets[robot] = np.asarray(values, dtype=np.float32)

    def set_gripper(self, width: Any, robot: str = "robot") -> None:
        self.gripper_cmd[robot] = width

    def set_ee_target(self, pose: Any, robot: str = "robot") -> None:
        if self.action_space != "ee_pose":
            raise UnsupportedActuation(f"{self.action_space} embodiment has no Cartesian target")
        self.ee_targets[robot] = pose

    def set_ee_delta(self, values: Any, robot: str = "robot") -> None:
        if self.action_space != "ee_pose":
            raise UnsupportedActuation(f"{self.action_space} embodiment has no Cartesian target")
        self.ee_deltas[robot] = np.asarray(values, dtype=np.float32)

    def hold(self, robot: str = "robot") -> None:
        self.holds.append(robot)

    def set_raw_action(self, values: Any, robot: str = "robot") -> None:
        self.raw_actions[robot] = np.asarray(values, dtype=np.float32)


# -- tasks used across the engine tests ----------------------------------


class Counter(Task):
    """Succeeds after ``steps`` ticks. The workhorse of these tests."""

    @dataclass
    class Outputs:
        ticks: int

    def __init__(self, steps: int = 1, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.steps = steps
        self.seen = 0
        self.entered = 0
        self.aborted = 0

    def on_enter(self, ctx: TickContext, inputs: Any) -> None:
        self.entered += 1
        self.seen = 0

    def tick(self, ctx: TickContext) -> Status:
        self.seen += 1
        ctx.act.set_joint_targets(np.zeros((ctx.num_envs, ctx.act.dof), dtype=np.float32))
        return Status.SUCCESS if self.seen >= self.steps else Status.RUNNING

    def on_exit(self, ctx: TickContext) -> Any:
        return self.Outputs(ticks=self.seen)

    def on_abort(self, ctx: TickContext) -> None:
        self.aborted += 1


class Failing(Task):
    """Fails after ``after`` ticks; succeeds instead once ``succeed_on_attempt`` is reached."""

    def __init__(self, after: int = 1, *, succeed_on_attempt: int | None = None, name: str | None = None) -> None:
        super().__init__(name=name)
        self.after = after
        self.succeed_on_attempt = succeed_on_attempt
        self.attempts = 0
        self.seen = 0

    def on_enter(self, ctx: TickContext, inputs: Any) -> None:
        self.attempts += 1
        self.seen = 0

    def tick(self, ctx: TickContext) -> Status:
        self.seen += 1
        if self.seen < self.after:
            return Status.RUNNING
        if self.succeed_on_attempt is not None and self.attempts >= self.succeed_on_attempt:
            return Status.SUCCESS
        return Status.FAILURE


class Producer(Task):
    @dataclass
    class Outputs:
        pose: Pose

    def __init__(self, pose: Pose | None = None, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.pose = pose or Pose.from_xyz(0.1, 0.2, 0.3)

    def tick(self, ctx: TickContext) -> Status:
        return Status.SUCCESS

    def on_exit(self, ctx: TickContext) -> Any:
        return self.Outputs(pose=self.pose)


class Consumer(Task):
    @dataclass
    class Inputs:
        target: Pose
        scale: float = 1.0

    @dataclass
    class Outputs:
        distance: float

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.received: Any = None

    def on_enter(self, ctx: TickContext, inputs: Any) -> None:
        self.received = inputs

    def tick(self, ctx: TickContext) -> Status:
        return Status.SUCCESS

    def on_exit(self, ctx: TickContext) -> Any:
        return self.Outputs(distance=float(np.linalg.norm(self.received.target.pos)))


class Writer(Task):
    """Writes one named actuator channel forever — for conflict tests."""

    def __init__(self, channel: str = "joint_targets", robot: str = "robot", *, name: str) -> None:
        super().__init__(name=name)
        self.channel = channel
        self.robot = robot

    def tick(self, ctx: TickContext) -> Status:
        if self.channel == "joint_targets":
            ctx.act.set_joint_targets(np.zeros((ctx.num_envs, ctx.act.dof), dtype=np.float32), self.robot)
        elif self.channel == "gripper":
            ctx.act.set_gripper(0.0, self.robot)
        else:
            ctx.act.hold(self.robot)
        return Status.RUNNING


class Exploding(Task):
    def tick(self, ctx: TickContext) -> Status:
        raise RuntimeError("boom")


@pytest.fixture
def scene() -> FakeScene:
    return FakeScene(objects={"scissors": Pose.from_xyz(0.1, 0.0, 0.25), "tray": Pose.from_xyz(0.1, 0.25, 0.26)})


@pytest.fixture
def ctx(scene: FakeScene) -> TickContext:
    return TickContext(scene=scene, act=FakeActuation(), dt=1 / 60, run_id="test-run")
