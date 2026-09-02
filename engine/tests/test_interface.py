# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from i4h_engine.graph import TaskGraph
from i4h_engine.interface import Workflow


def _idle() -> TaskGraph:
    return TaskGraph()


def test_workflow_is_a_small_author_value() -> None:
    def success(_ctx) -> bool:
        return True

    workflow = Workflow(scene="example", modes={"idle": _idle}, success=success)

    assert workflow.scene == "example"
    assert workflow.default_mode == "idle"
    assert workflow.modes["idle"] is _idle
    assert workflow.success is success


def test_workflow_copies_and_freezes_the_mode_map() -> None:
    modes = {"idle": _idle}
    workflow = Workflow(scene="example", modes=modes)
    modes["other"] = _idle

    assert tuple(workflow.modes) == ("idle",)
    with pytest.raises(TypeError):
        workflow.modes["other"] = _idle  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scene": "", "modes": {"idle": _idle}}, "scene must not be empty"),
        ({"scene": "example", "modes": {}}, "at least one mode"),
        ({"scene": "example", "modes": {"rule-based": _idle}}, "default mode 'idle' is not exposed"),
        ({"scene": "example", "modes": {"idle": None}}, "every workflow mode must be callable"),
    ],
)
def test_workflow_rejects_an_invalid_contract(kwargs, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Workflow(**kwargs)
