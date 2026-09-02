# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numpy implementations of the world contract, for testing skills without Isaac.

Shipped as part of ``i4h_tasks.basic`` rather than hidden in a test directory
because every downstream task project needs it: writing a new skill for a new
policy stack should not require a simulator to verify.

:class:`FakeScene` optionally integrates commanded joint targets with a
first-order lag, so tasks that check "did I actually arrive?" — :class:`~i4h_tasks.
basic.motion.home.Home`, the settle predicates — exercise their real path
instead of trivially passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from i4h_common.types import CameraFrame, JointState, ObjectState, Pose
from i4h_common.world import UnsupportedActuation


class FakeScene:
    """A world of posed objects plus one articulated robot."""

    def __init__(
        self,
        *,
        num_envs: int = 1,
        dof: int = 6,
        joint_names: tuple[str, ...] | None = None,
        objects: dict[str, Pose] | None = None,
        footprint_half_extents: dict[str, tuple[float, float]] | None = None,
        robot_pose: Pose | None = None,
        robot_velocity: np.ndarray | None = None,
        robot_angular_velocity: np.ndarray | None = None,
        home: np.ndarray | None = None,
        follow_rate: float = 0.0,
    ) -> None:
        self._num_envs = num_envs
        self._dof = dof
        self._joint_names = joint_names or tuple(f"joint_{i}" for i in range(dof))
        self._objects: dict[str, Pose] = dict(objects or {})
        self._footprint_half_extents = dict(footprint_half_extents or {})
        self._velocities: dict[str, np.ndarray] = {}
        self.robot_pose = robot_pose or Pose.identity(num_envs)
        self.robot_velocity = (
            np.zeros((num_envs, 3), dtype=np.float32)
            if robot_velocity is None
            else np.atleast_2d(np.asarray(robot_velocity, dtype=np.float32))
        )
        self.robot_angular_velocity = (
            np.zeros((num_envs, 3), dtype=np.float32)
            if robot_angular_velocity is None
            else np.atleast_2d(np.asarray(robot_angular_velocity, dtype=np.float32))
        )
        self.home = (
            np.asarray(home, dtype=np.float32).reshape(1, dof).repeat(num_envs, axis=0)
            if home is not None
            else np.zeros((num_envs, dof), dtype=np.float32)
        )
        self.joint_pos = self.home.copy()
        self.joint_vel = np.zeros((num_envs, dof), dtype=np.float32)
        self.gripper = np.zeros(num_envs, dtype=np.float32)
        self.tcp_pose = Pose.identity(num_envs)
        self.contacts: set[tuple[str, str]] = set()
        self.frames: dict[str, CameraFrame] = {}
        self.observations: dict[str, dict[str, np.ndarray]] = {}
        self.terminations: dict[str, np.ndarray] = {}
        #: 0 = joints teleport to the command; 1 = they never move. Anything in
        #: between models a lagging controller.
        self.follow_rate = follow_rate

    # -- SceneView -------------------------------------------------------
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
            raise KeyError(f"no object {name!r} in fake scene; have {list(self._objects)}")
        velocity = self._velocities.get(name, np.zeros((self._num_envs, 3), dtype=np.float32))
        return ObjectState(name, self._objects[name], velocity)

    def robot_root(self, robot: str = "robot") -> ObjectState:
        return ObjectState(
            robot,
            self.robot_pose,
            self.robot_velocity,
            self.robot_angular_velocity,
        )

    def footprint_half_extents(self, name: str) -> np.ndarray:
        try:
            values = self._footprint_half_extents[name]
        except KeyError as exc:
            raise KeyError(
                f"no collision footprint for {name!r}; fake scene has {sorted(self._footprint_half_extents)}"
            ) from exc
        return np.broadcast_to(
            np.asarray(values, dtype=np.float32),
            (self._num_envs, 2),
        ).copy()

    def joints(self, robot: str = "robot") -> JointState:
        return JointState(pos=self.joint_pos, vel=self.joint_vel, names=self._joint_names)

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
        return self.frames.get(name)

    def observation(self, group: str, name: str) -> np.ndarray:
        try:
            return self.observations[group][name]
        except KeyError as exc:
            raise KeyError(f"no fake observation {group}.{name}") from exc

    def termination(self, name: str) -> np.ndarray:
        return self.terminations.get(name, np.zeros(self._num_envs, dtype=bool))

    # -- test helpers ----------------------------------------------------
    def place(self, name: str, pose: Pose, velocity: np.ndarray | None = None) -> None:
        self._objects[name] = pose
        if velocity is not None:
            self._velocities[name] = np.atleast_2d(np.asarray(velocity, dtype=np.float32))

    def set_velocity(self, name: str, velocity: Any) -> None:
        self._velocities[name] = np.atleast_2d(np.asarray(velocity, dtype=np.float32))

    def set_robot_root(
        self,
        pose: Pose,
        velocity: Any = (0.0, 0.0, 0.0),
        angular_velocity: Any = (0.0, 0.0, 0.0),
    ) -> None:
        self.robot_pose = pose
        self.robot_velocity = np.atleast_2d(np.asarray(velocity, dtype=np.float32))
        self.robot_angular_velocity = np.atleast_2d(np.asarray(angular_velocity, dtype=np.float32))

    def step(self, actuation: FakeActuation) -> None:
        """Apply one step of the toy dynamics. The runner's job, faked."""
        command = actuation.joint_targets.get("robot")
        if command is not None:
            target = np.asarray(command, dtype=np.float32)
            previous = self.joint_pos
            self.joint_pos = previous + (target - previous) * (1.0 - self.follow_rate)
            self.joint_vel = self.joint_pos - previous
        jaw = actuation.gripper_cmd.get("robot")
        if jaw is not None:
            # The jaw lags like the joints do. That matters: Grasp's fallback
            # check reads "commanded closed but did not get there" as "something
            # is in the way", which is untestable if the fake snaps instantly.
            goal = np.full(self._num_envs, float(np.asarray(jaw).reshape(-1)[0]), dtype=np.float32)
            self.gripper = self.gripper + (goal - self.gripper) * (1.0 - self.follow_rate)


@dataclass
class FakeActuation:
    """Records commands so a test can assert on what was driven."""

    dof: int = 6
    action_space: str = "joint_position"
    joint_targets: dict[str, np.ndarray] = field(default_factory=dict)
    gripper_cmd: dict[str, Any] = field(default_factory=dict)
    ee_targets: dict[str, Pose] = field(default_factory=dict)
    ee_deltas: dict[str, np.ndarray] = field(default_factory=dict)
    raw_actions: dict[str, np.ndarray] = field(default_factory=dict)
    holds: list[str] = field(default_factory=list)

    def set_joint_targets(self, values: Any, robot: str = "robot") -> None:
        if self.action_space != "joint_position":
            raise UnsupportedActuation(f"{self.action_space} embodiment has no joint-position target")
        self.joint_targets[robot] = np.asarray(values, dtype=np.float32)

    def set_gripper(self, width: Any, robot: str = "robot") -> None:
        self.gripper_cmd[robot] = width

    def set_ee_target(self, pose: Pose, robot: str = "robot") -> None:
        if self.action_space != "ee_pose":
            raise UnsupportedActuation(
                f"{self.action_space} embodiment has no Cartesian target; "
                f"workflow-lint should have caught this via requires/provides"
            )
        self.ee_targets[robot] = pose

    def set_ee_delta(self, values: Any, robot: str = "robot") -> None:
        if self.action_space != "ee_pose":
            raise UnsupportedActuation(f"{self.action_space} embodiment has no Cartesian target")
        self.ee_deltas[robot] = np.asarray(values, dtype=np.float32)

    def hold(self, robot: str = "robot") -> None:
        self.holds.append(robot)

    def set_raw_action(self, values: Any, robot: str = "robot") -> None:
        self.raw_actions[robot] = np.asarray(values, dtype=np.float32)


def fake_world(**kwargs: Any) -> tuple[FakeScene, FakeActuation]:
    """A scene and a matching actuation, wired to the same DOF."""
    scene = FakeScene(**kwargs)
    return scene, FakeActuation(dof=scene.joints().dof)
