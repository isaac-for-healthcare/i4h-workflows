# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from i4h_engine.graph import TaskGraph
from i4h_engine.interface import Workflow
from i4h_engine.loader import available_workflows, load_workflow_module, resolve_workflow

EXPECTED_SPECIALTIES = {
    # The surgical workflows run as IsaacLab-Arena environments with i4h policies (see the README).
    "laparoscopic-robotics": set(),
    "ultrasound-robotics": set(),
    "endoluminal-robotics": {"endoluminal_navigation", "endoluminal_navigation_arm"},
    "hospital-automation-robotics": {
        "scissor_pick_and_place",
        "spread_tablecloth_g1",
        "spread_tablecloth_h2",
    },
}


def test_workflows_are_grouped_by_approved_specialty() -> None:
    authored_root = Path(__file__).parents[1] / "i4h_workflows"
    modes_root = Path(__file__).parents[1] / "i4h_workflow_modes"
    specialty_directories = {
        path.name for path in authored_root.iterdir() if path.is_dir() and not path.name.startswith("_")
    }

    actual = {
        specialty: {path.stem for path in (authored_root / specialty).glob("*.py")}
        for specialty in EXPECTED_SPECIALTIES
    }

    assert specialty_directories == set(EXPECTED_SPECIALTIES)
    assert not list(authored_root.glob("*.py"))
    assert actual == EXPECTED_SPECIALTIES
    assert set(available_workflows()) == set().union(*EXPECTED_SPECIALTIES.values())
    assert {path.stem for path in modes_root.glob("*.py")} == {"idle", "policy", "replay", "teleop"}


def test_authored_module_exports_one_workflow_value() -> None:
    source = load_workflow_module("scissor_pick_and_place")

    assert isinstance(source.workflow, Workflow)
    assert source.scene == "soarm_scissors"
    assert "rule-based" in source.modes
    assert source.default_mode == "idle"


def test_resolve_builds_a_mode_specific_task_graph() -> None:
    resolved = resolve_workflow("scissor_pick_and_place", "rule-based")

    assert resolved.name == "scissor_pick_and_place"
    assert resolved.scene == "soarm_scissors"
    assert resolved.mode == "rule-based"
    assert isinstance(resolved.graph, TaskGraph)


def test_resolve_uses_the_authored_default_mode() -> None:
    resolved = resolve_workflow("spread_tablecloth_g1")

    assert resolved.mode == "idle"


def test_policy_node_uses_the_task_owned_prompt() -> None:
    resolved = resolve_workflow("scissor_pick_and_place", "policy")

    assert resolved.graph.nodes[0].spec.prompt == ""
    assert resolved.graph.nodes[0].spec.effective_prompt == "Grip the scissors and put it into the tray"
    assert "prompt" not in resolved.graph.nodes[0].params


def test_scissor_rule_based_verifies_placement_before_returning_home() -> None:
    resolved = resolve_workflow("scissor_pick_and_place", "rule-based")

    ids = [node.id for node in resolved.graph.nodes]
    assert ids[-4:] == ["retreat", "settle", "verify_placement", "home"]
    assert [node.id for node in resolved.graph.terminals()] == ["home"]


def test_replay_builder_forwards_all_supported_arguments() -> None:
    resolved = resolve_workflow(
        "scissor_pick_and_place",
        "replay",
        dataset="/tmp/demo.hdf5",
        episode=2,
        node="grasp",
    )

    assert resolved.graph.nodes[0].params == {
        "dataset": "/tmp/demo.hdf5",
        "episode": 2,
        "node": "grasp",
    }


def test_resolve_rejects_an_unexposed_mode() -> None:
    with pytest.raises(KeyError, match="has no mode 'policy'"):
        resolve_workflow("spread_tablecloth_g1", "policy")
