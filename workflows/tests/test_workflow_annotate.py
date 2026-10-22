# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lint checks a workflow's grading cameras against the cameras its scene records (NVBug 6941887)."""

from __future__ import annotations

from i4h_common.annotate import Annotate
from i4h_engine.graph import TaskGraph
from i4h_engine.lint import lint_workflow
from i4h_engine.loader import ResolvedWorkflow
from i4h_engine.registry import default_registry


def test_lint_rejects_a_grading_camera_the_scene_does_not_record() -> None:
    workflow = ResolvedWorkflow(
        name="probe",
        scene="panda_phantom",
        mode="idle",
        graph=TaskGraph(),
        annotate=Annotate(cameras=("room", "overview")),
    )

    report = lint_workflow(workflow, default_registry())

    (issue,) = (issue for issue in report.issues if issue.code == "annotate-camera")
    assert "['overview']" in issue.message
    assert "['room', 'wrist']" in issue.message
