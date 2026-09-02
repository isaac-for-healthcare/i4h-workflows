# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every skill on disk is discoverable, and says what it needs.

There is no manifest to drift from any more — the class is the declaration —
so what is worth asserting changed: that discovery finds each one, and that a
task which needs something from the scene actually says so.
"""

from __future__ import annotations

import pytest

from i4h_common.paths import workflow_root
from i4h_engine.discover import discover_tasks

SPECS = {
    task_id: spec.resolve() for task_id, spec in discover_tasks(workflow_root())[0].items() if spec.project == "basic"
}


def test_discovery_finds_every_skill():
    assert SPECS, "no tasks discovered under i4h_tasks.basic"
    names = {task_id.rpartition("/")[2] for task_id in SPECS}
    assert {"grasp", "release", "home", "locate", "wait", "settle"} <= names


@pytest.mark.parametrize("task_id", sorted(SPECS), ids=lambda t: t.rpartition("/")[2])
def test_task_name_matches_its_module(task_id):
    """The module stem is the task name — the whole basis for import-discovery."""
    spec = SPECS[task_id]
    assert spec.impl.partition(":")[0].rpartition(".")[2] == spec.name


@pytest.mark.parametrize("task_id", sorted(SPECS), ids=lambda t: t.rpartition("/")[2])
def test_ports_come_from_the_class(task_id):
    spec = SPECS[task_id]
    for declared in list(spec.inputs.values()) + list(spec.outputs.values()):
        assert not declared.startswith("None"), f"{task_id}: unresolved annotation {declared!r}"


def test_gripper_tasks_require_a_gripper():
    for name in ("grasp", "release", "set_gripper"):
        assert SPECS[f"basic/{name}"].requires.get("gripper") is True
