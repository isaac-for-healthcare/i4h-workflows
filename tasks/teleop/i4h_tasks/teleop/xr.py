# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manually graded XR demonstrations through the normal workflow recorder."""

from __future__ import annotations

import logging
import weakref
from typing import Any

import numpy as np

from i4h_common.world import apply_action
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext
from i4h_tasks.teleop.devices import keyboard_event_input_name

logger = logging.getLogger(__name__)


class XRInput:
    """IsaacTeleop lifetime and keyboard controls; never advances simulation."""

    def __init__(self, cfg: Any, *, cloudxr_env: str | None, auto_launch_cloudxr: bool) -> None:
        from isaaclab_teleop import create_isaac_teleop_device

        self._device = create_isaac_teleop_device(
            cfg, cloudxr_env_file=cloudxr_env, auto_launch_cloudxr=auto_launch_cloudxr
        )
        self._events: set[str] = set()
        self._input = self._keyboard = self._subscription = None
        self._closed = False

    def open(self) -> None:
        import carb.input
        import omni.appwindow

        try:
            self._device.__enter__()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._subscription = self._input.subscribe_to_keyboard_events(
                self._keyboard, lambda event, *args, obj=weakref.proxy(self): obj._on_key(event)
            )
        except BaseException:
            self.close()
            raise

    def _on_key(self, event: Any) -> bool:
        import carb.input

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = keyboard_event_input_name(event)
            if key in {"B", "S", "R"}:
                self._events.add(key)
        return True

    def read(self) -> tuple[np.ndarray | None, set[str]]:
        command = self._device.advance()
        events, self._events = self._events, set()
        # B/S/R own recording, as in v0.7. Do not treat the retargeter's
        # acknowledgement of our reset() as another operator reset request.
        if command is not None:
            command = command.detach().cpu().numpy()
        return command, events

    def reset(self) -> None:
        self._device.reset()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._subscription is not None:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
        finally:
            self._subscription = None
            self._device.__exit__(None, None, None)


class XR(Task):
    requires = {"action_space": "bimanual_pose"}

    def __init__(
        self,
        *,
        max_seconds: float = 300.0,
        cloudxr_env: str | None = None,
        auto_launch_cloudxr: bool = False,
        name=None,
    ) -> None:
        super().__init__(name=name)
        self.max_seconds = max_seconds
        self.cloudxr_env = cloudxr_env
        self.auto_launch_cloudxr = auto_launch_cloudxr
        self._input: XRInput | None = None
        self._recording = False
        self._frames = 0
        self._reset_input = False

    def on_enter(self, ctx: TickContext, inputs: Any) -> None:
        if ctx.num_envs != 1:
            raise ValueError("XR demonstrations require one environment")
        self._input = XRInput(
            ctx.scene.teleop_config(), cloudxr_env=self.cloudxr_env, auto_launch_cloudxr=self.auto_launch_cloudxr
        )
        self._input.open()
        self._recording = False
        self._frames = 0
        self._reset_input = True
        logger.info("XR ready: B starts a fresh demo; S saves; R discards and resets")

    def tick(self, ctx: TickContext) -> Status:
        try:
            return self._tick(ctx)
        except Exception:
            self._close()
            raise

    def _tick(self, ctx: TickContext) -> Status:
        assert self._input is not None
        if self._reset_input:
            self._input.reset()
            self._reset_input = False
        command, events = self._input.read()
        if "R" in events or "B" in events:
            self._recording = "B" in events and "R" not in events
            self._frames = 0
            ctx.request_scene_reset()
            self._reset_input = True
            return Status.WAITING
        if "S" in events and self._recording and self._frames:
            return Status.SUCCESS  # Operator acceptance; no geometric success claim.
        if not self._recording or command is None:
            return Status.WAITING
        if self._frames * ctx.dt >= self.max_seconds:
            self._close()
            return Status.FAILURE
        command = np.asarray(command, dtype=np.float32).reshape(1, -1)
        if command.shape[1] != ctx.act.dof or not np.isfinite(command).all():
            raise ValueError(f"invalid XR action: expected {ctx.act.dof} finite values, got {command.shape}")
        apply_action(ctx.act, command)
        self._frames += 1
        return Status.RUNNING

    def on_exit(self, ctx: TickContext) -> Any:
        self._close()
        return self.Outputs()

    def on_abort(self, ctx: TickContext) -> None:
        self._close()

    def _close(self) -> None:
        if self._input is not None:
            self._input.close()
            self._input = None
