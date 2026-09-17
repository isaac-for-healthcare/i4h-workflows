# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spread a tablecloth with H2 using XR hand teleoperation."""

from i4h_engine.graph import TaskGraph, task
from i4h_engine.interface import Workflow
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.replay import replay


def teleop(
    device: str = "xr", *, cloudxr_env: str | None = None, auto_launch_cloudxr: bool = False, **kwargs
) -> TaskGraph:
    if device != "xr":
        raise ValueError("tablecloth teleoperation requires the xr device")
    return TaskGraph(description="B: begin, S: save, R: discard/reset.").flow(
        task("teleop/xr", max_seconds=300.0, cloudxr_env=cloudxr_env, auto_launch_cloudxr=auto_launch_cloudxr)
    )


WORKFLOW = Workflow(
    scene="h2_tablecloth",
    modes={"idle": idle, "teleop": teleop, "replay": replay},
)
