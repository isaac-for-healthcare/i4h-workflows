# SPDX-License-Identifier: Apache-2.0
"""Static-aorta fluoroscopy with a distinct normalized action contract."""

from i4h_engine.graph import TaskGraph, task
from i4h_engine.interface import Workflow
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.replay import replay

WORKFLOW = Workflow(
    scene="endoluminal_aorta_fluoroscopy",
    modes={
        "idle": idle,
        "demo": lambda: TaskGraph(description="115 normalized static-aorta commands.").flow(
            task("basic/aorta_fluoroscopy_sweep")
        ),
        "replay": replay,
    },
)
