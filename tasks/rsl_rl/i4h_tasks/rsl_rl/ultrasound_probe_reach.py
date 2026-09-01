# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the exported ultrasound probe-reach policy as an in-process Task."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from i4h_common.types import satisfied
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


def policy_observation(ctx: TickContext, previous_action: np.ndarray) -> np.ndarray:
    """Recreate the 34-D observation used to train the probe-reach policy."""
    joints = ctx.scene.joints("robot")
    home = ctx.scene.home_joints("robot")
    probe = ctx.scene.tcp("robot")
    target = ctx.scene.object("target").pose
    # SceneView exposes ee-pose scenes in the robot frame and converts simulator
    # xyzw quaternions to the common wxyz contract. Training consumed the raw
    # Isaac tensors, so convert only the quaternion layout back to xyzw here.
    probe_quat_xyzw = probe.quat[:, [1, 2, 3, 0]]
    target_quat_xyzw = target.quat[:, [1, 2, 3, 0]]
    observation = np.concatenate(
        (
            joints.pos - home,
            joints.vel,
            previous_action,
            probe.pos,
            probe_quat_xyzw,
            target.pos,
            target_quat_xyzw,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    if observation.shape != (ctx.num_envs, 34):
        raise ValueError(f"ultrasound probe-reach policy expects {(ctx.num_envs, 34)}, got {observation.shape}")
    return observation


class UltrasoundProbeReachPolicy(Task):
    """Evaluate the exported probe-reach actor without owning simulator stepping."""

    requires = {"action_space": "ee_pose", "dof": 6, "objects": ["target"], "robots": ["robot"]}

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        until: Callable[[TickContext], Any] | None = None,
        device: str = "cuda:0",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.checkpoint = Path(checkpoint).expanduser() if checkpoint else None
        self.until = until
        self.device = device
        self._policy: Any = None
        self._previous_action = np.empty((0, 6), dtype=np.float32)
        self._ticks = 0

    def _checkpoint_file(self) -> Path:
        if self.checkpoint is None:
            raise ValueError("ultrasound probe-reach policy mode requires --checkpoint pointing to exported policy.pt")
        candidate = self.checkpoint / "policy.pt" if self.checkpoint.is_dir() else self.checkpoint
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"exported ultrasound probe-reach policy does not exist: {candidate}")
        return candidate

    def on_enter(self, ctx: TickContext, inputs: Any) -> None:
        del inputs
        if self._policy is None:
            import torch  # noqa: PLC0415 - the simulator runtime owns the Torch install

            self._policy = torch.jit.load(str(self._checkpoint_file()), map_location=self.device).eval()
        self._previous_action = np.zeros((ctx.num_envs, 6), dtype=np.float32)
        self._ticks = 0

    def tick(self, ctx: TickContext) -> Status:
        # IsaacLab intentionally retains a terminal term across its automatic
        # reset. Require one fresh policy step before accepting success so a new
        # episode cannot inherit the previous episode's terminal pulse.
        if self._ticks > 0 and self.until is not None and satisfied(self.until(ctx)):
            return Status.SUCCESS
        if self._policy is None:
            raise RuntimeError("ultrasound probe-reach policy was not loaded; on_enter was not called")
        import torch  # noqa: PLC0415 - optional outside the simulator runtime

        observation = policy_observation(ctx, self._previous_action)
        with torch.inference_mode():
            action = self._policy(torch.as_tensor(observation, device=self.device)).detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ctx.num_envs, 6):
            raise ValueError(f"ultrasound probe-reach policy returned {action.shape}; expected {(ctx.num_envs, 6)}")
        self._previous_action = action.copy()
        ctx.act.set_raw_action(action, "robot")
        self._ticks += 1
        return Status.RUNNING
