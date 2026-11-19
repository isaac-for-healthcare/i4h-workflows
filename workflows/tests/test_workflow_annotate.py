# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A workflow, not its policy task, says how its recordings are graded (NVBug 6941887)."""

from __future__ import annotations

import pytest

from i4h_common.annotate import Annotate
from i4h_engine.graph import TaskGraph
from i4h_engine.lint import lint_workflow
from i4h_engine.loader import ResolvedWorkflow, available_workflows, load_workflow_module, resolve_workflow
from i4h_engine.registry import default_registry
from i4h_engine.render import to_text

ANNOTATED = [name for name in available_workflows() if load_workflow_module(name).workflow.annotate is not None]


def test_the_liver_scan_declares_a_visible_criterion_over_both_cameras() -> None:
    annotate = load_workflow_module("ultrasound_liver_scan").workflow.annotate

    assert annotate is not None
    assert "phantom" in annotate.criterion
    assert annotate.cameras == ("room", "wrist")
    assert annotate.judge == "trajectory"


@pytest.mark.parametrize("mode", ["policy", "rule-based", "teleop", "replay"])
def test_every_recording_mode_sees_the_same_grading_settings(mode: str) -> None:
    """Teleop and rule-based recordings have no policy task, but are graded the same way."""
    workflow = resolve_workflow("ultrasound_liver_scan", mode)

    assert workflow.annotate is load_workflow_module("ultrasound_liver_scan").workflow.annotate


@pytest.mark.parametrize("name", ANNOTATED)
def test_declared_grading_cameras_are_cameras_the_scene_records(name: str) -> None:
    registry = default_registry()
    for mode in load_workflow_module(name).modes:
        report = lint_workflow(resolve_workflow(name, mode), registry)
        assert not [issue for issue in report.issues if issue.code == "annotate-camera"], report.render()


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


def test_show_prints_the_grading_settings() -> None:
    text = to_text(resolve_workflow("ultrasound_liver_scan", "policy"))

    assert "annotate: judge=trajectory cameras=room,wrist" in text
    assert "criterion: The ultrasound probe on the robot's wrist rests on the abdominal phantom" in text
    assert "annotate:" not in to_text(resolve_workflow("scissor_pick_and_place", "policy"))
