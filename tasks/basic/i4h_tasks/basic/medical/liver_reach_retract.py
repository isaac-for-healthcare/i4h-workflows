# SPDX-License-Identifier: Apache-2.0
"""Reference controller Task; all simulation stepping remains with Isaac Lab."""

from i4h_common.world import apply_action
from i4h_engine.status import Status
from i4h_engine.task import Task


class LiverReachRetractController(Task):
    requires = {"action_space": "liver_cartesian_delta", "dof": 4}

    def tick(self, ctx):
        if ctx.scene.liver_success().all():
            return Status.SUCCESS
        apply_action(ctx.act, ctx.scene.liver_teacher_action())
        return Status.RUNNING
