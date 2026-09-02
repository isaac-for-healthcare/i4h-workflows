# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Close the jaw and verify something is actually held.

Verification is by contact when the scene reports it, falling back to "the
jaw stopped short of fully closed" — the signature of a blocked jaw."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from i4h_engine.status import Status
from i4h_engine.task import TickContext
from i4h_tasks.basic.gripper.set_gripper import SetGripper


class Grasp(SetGripper):
    """Close the jaw and verify something is actually held.

    Verification is by contact with ``object`` when the scene reports contacts,
    falling back to "the jaw stopped short of fully closed", which is the usual
    signature of a jaw blocked by the thing it grabbed.
    """

    requires = {"gripper": True}
    precondition = {"holding": "none"}
    postcondition = {"holding": "$object"}

    @dataclass
    class Inputs:
        object: str = ""

    @dataclass
    class Outputs:
        grasped: bool = False
        width: float = 0.0

    def __init__(
        self,
        *,
        width: float = -0.16,
        duration_s: float = 0.3,
        object: str = "",  # noqa: A002 - matches the manifest port name
        verify: bool = True,
        closed_epsilon: float = 0.01,
        name: str | None = None,
    ) -> None:
        super().__init__(width, duration_s=duration_s, name=name)
        self.object = object
        self.verify = verify
        self.closed_epsilon = closed_epsilon
        self._grasped = False

    def on_enter(self, ctx: TickContext, inputs: Inputs) -> None:
        super().on_enter(ctx, inputs)
        self.object = getattr(inputs, "object", "") or self.object
        self._grasped = False

    def tick(self, ctx: TickContext) -> Status:
        status = super().tick(ctx)
        if status is not Status.SUCCESS:
            return status
        self._grasped = self._check(ctx)
        return Status.SUCCESS if (self._grasped or not self.verify) else Status.FAILURE

    def _check(self, ctx: TickContext) -> bool:
        if self.object and self.object in ctx.scene.objects:
            try:
                contact = np.asarray(ctx.scene.contact("robot", self.object))
                if bool(contact.all()):
                    return True
            except (KeyError, NotImplementedError):
                pass
            # No contact sensor, or no reported contact: retain the documented
            # blocked-jaw fallback instead of returning False immediately.
        actual = np.asarray(ctx.scene.gripper_width())
        return bool((np.abs(actual - self.width) > self.closed_epsilon).all())

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(grasped=self._grasped, width=self.width)
