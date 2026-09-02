# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The world contract: what a task may read, and what it may write.

These two protocols are the entire boundary between the capability library
(``tasks/*``) and the simulator (``arena``). ``arena`` implements them against
IsaacLab; ``i4h_tasks.basic.testing.FakeScene`` implements them against plain numpy.
Neither side imports the other.

A task **never** calls ``env.step``. It reads through :class:`SceneView` and
writes through :class:`Actuation`; the runner owns stepping. That single rule is
what makes tasks unit-testable on CPU and what lets parallel workflow branches
coexist safely.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from i4h_common.types import CameraFrame, JointState, ObjectState, Pose


class UnsupportedActuation(RuntimeError):
    """Raised when a task asks for an actuation mode the scene does not provide.

    Normally prevented by ``workflow-lint`` matching a task's ``requires`` against a
    scene's ``provides``, so hitting this at run time means a manifest is wrong.
    """


class ActuationConflict(RuntimeError):
    """Raised when two concurrently active nodes write the same actuator in one tick.

    This is what makes parallel branches safe: overlapping writes are a workflow bug,
    not a last-writer-wins race.
    """


@runtime_checkable
class SceneView(Protocol):
    """Read side of the world. All returns are batched over ``num_envs``."""

    @property
    def num_envs(self) -> int: ...

    @property
    def objects(self) -> tuple[str, ...]:
        """Names of addressable scene objects (validated by workflow-lint)."""
        ...

    @property
    def robots(self) -> tuple[str, ...]:
        """Names of addressable robots. Single-arm scenes expose ``("robot",)``."""
        ...

    def object(self, name: str) -> ObjectState:
        """Pose + velocity of a scene object. Raises ``KeyError`` if absent."""
        ...

    def robot_root(self, robot: str = "robot") -> ObjectState:
        """Floating-base pose plus linear and angular velocity for an articulated robot."""
        ...

    def footprint_half_extents(self, name: str) -> np.ndarray:
        """Scene-owned horizontal collision footprint half-extents, shape ``(N, 2)``."""
        ...

    def joints(self, robot: str = "robot") -> JointState:
        """Current joint positions/velocities for a robot."""
        ...

    def home_joints(self, robot: str = "robot") -> np.ndarray:
        """The robot's home joint positions, shape ``(num_envs, dof)``."""
        ...

    def tcp(self, robot: str = "robot") -> Pose:
        """Tool-centre-point pose in the world frame."""
        ...

    def gripper_width(self, robot: str = "robot") -> np.ndarray:
        """Current jaw opening in metres, shape ``(num_envs,)``."""
        ...

    def contact(self, a: str, b: str) -> np.ndarray:
        """Per-env boolean mask: are ``a`` and ``b`` in contact?"""
        ...

    def camera(self, name: str) -> CameraFrame | None:
        """Latest frame for a camera, or ``None`` if cameras are disabled."""
        ...

    def observation(self, group: str, name: str) -> np.ndarray:
        """Latest named simulator observation, preserving its controller-facing encoding."""
        ...

    def termination(self, name: str) -> np.ndarray:
        """Latest named environment-termination mask.

        Term values remain available across the automatic reset following a
        terminal step, so workflows can use task criteria directly.
        """
        ...


@runtime_checkable
class Actuation(Protocol):
    """Write side of the world. One tick's worth of commands.

    The runner clears this between steps, collects whatever active nodes wrote,
    and composes the action tensor. Two nodes writing the same actuator in one
    tick raises :class:`ActuationConflict`.
    """

    @property
    def dof(self) -> int: ...

    @property
    def action_space(self) -> str:
        """``joint_position`` | ``ee_pose`` | ``joint_velocity``."""
        ...

    def set_joint_targets(self, values: np.ndarray, robot: str = "robot") -> None:
        """Absolute joint position targets, ``(num_envs, dof)``."""
        ...

    def set_gripper(self, width: np.ndarray | float, robot: str = "robot") -> None:
        """Jaw command. Units follow the embodiment's own convention."""
        ...

    def set_ee_target(self, pose: Pose, robot: str = "robot") -> None:
        """Cartesian end-effector target.

        Raises :class:`UnsupportedActuation` on joint-only embodiments such as
        the SO-ARM 101.
        """
        ...

    def set_ee_delta(self, values: np.ndarray, robot: str = "robot") -> None:
        """Relative Cartesian command ``[dx, dy, dz, rx, ry, rz]``."""
        ...

    def hold(self, robot: str = "robot") -> None:
        """Repeat the previous command for this robot (explicit no-op)."""
        ...

    def set_raw_action(self, values: np.ndarray, robot: str = "robot") -> None:
        """Write an action already encoded for this scene's controller."""
        ...


def apply_action(act: Actuation, values: np.ndarray, robot: str = "robot") -> None:
    """Write an action whose controller-specific encoding must remain unchanged.

    Replay and teleop are pass-through tasks. An absolute pose recording uses
    the simulator's ``xyzw`` boundary convention, while a relative pose policy
    emits a six-value delta; decoding either here would silently alter it.
    """
    act.set_raw_action(np.atleast_2d(np.asarray(values, dtype=np.float32)), robot)
