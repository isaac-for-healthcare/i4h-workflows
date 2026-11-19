# SPDX-License-Identifier: Apache-2.0
"""Liver reach and retract through the normal Workflow lifecycle."""

from i4h_engine.graph import TaskGraph, task
from i4h_engine.interface import Workflow
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy
from i4h_workflow_modes.replay import replay


def success(ctx):
    return ctx.scene.liver_success()


WORKFLOW = Workflow(
    scene="liver_reach_retract",
    success=success,
    modes={
        "idle": idle,
        "rule-based": lambda: TaskGraph(description="Complete the liver manipulation cycle.").flow(
            task("basic/liver_reach_retract")
        ),
        "policy": lambda: policy("rsl_rl/liver_reach_retract", until=success),
        "replay": replay,
    },
)
