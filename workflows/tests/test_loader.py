# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from i4h_engine.graph import TaskGraph
from i4h_engine.interface import Workflow
from i4h_engine.loader import available_workflows, load_workflow_module, resolve_workflow

EXPECTED_SPECIALTIES = {
    "laparoscopic-robotics": {
        "surgical_lift_block",
        "surgical_lift_needle",
        "surgical_lift_needle_organs",
        "surgical_reach_dual_psm",
        "surgical_reach_psm",
        "surgical_reach_star",
    },
    "ultrasound-robotics": {"ultrasound_liver_scan", "ultrasound_probe_reach"},
    "endoluminal-robotics": {"endoluminal_navigation"},
    "hospital-automation-robotics": {
        "assemble_trocar",
        "locomanip_push_cart",
        "locomanip_tray_pick_and_place",
        "scissor_pick_and_place",
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
    assert {path.stem for path in modes_root.glob("*.py")} == {"idle", "policy", "replay", "rule_based", "teleop"}


def test_authored_module_exports_one_workflow_value() -> None:
    source = load_workflow_module("scissor_pick_and_place")

    assert isinstance(source.workflow, Workflow)
    assert source.scene == "soarm_scissors"
    assert "rule-based" in source.modes
    assert source.default_mode == "idle"


def test_resolve_builds_a_mode_specific_task_graph() -> None:
    resolved = resolve_workflow("surgical_reach_psm", "rule-based")

    assert resolved.name == "surgical_reach_psm"
    assert resolved.scene == "psm_reach"
    assert resolved.mode == "rule-based"
    assert isinstance(resolved.graph, TaskGraph)
    assert [node.id for node in resolved.graph.nodes] == ["locate", "rest", "reach", "hold"]


def test_resolve_uses_the_authored_default_mode() -> None:
    resolved = resolve_workflow("assemble_trocar")

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


@pytest.mark.parametrize(
    "workflow",
    [
        "surgical_lift_block",
        "surgical_lift_needle",
        "surgical_lift_needle_organs",
    ],
)
def test_surgical_lift_verifies_the_object_height(workflow: str) -> None:
    resolved = resolve_workflow(workflow, "rule-based")

    ids = [node.id for node in resolved.graph.nodes]
    assert ids[:2] == ["rest", "locate"]
    assert ids[-2:] == ["verify_lift", "hold"]
    assert [node.id for node in resolved.graph.terminals()] == ["hold"]


def test_dual_psm_reach_holds_both_arms_before_moving() -> None:
    resolved = resolve_workflow("surgical_reach_dual_psm", "rule-based")

    ids = [node.id for node in resolved.graph.nodes]
    assert ids == ["locate_1", "rest_1", "reach_1", "settle", "locate_2", "rest_2", "reach_2"]


def test_ultrasound_rule_based_finishes_on_simulator_success() -> None:
    resolved = resolve_workflow("ultrasound_liver_scan", "rule-based")

    ids = [node.id for node in resolved.graph.nodes]
    assert ids[-2:] == ["hold", "verify_scan"]
    assert [node.id for node in resolved.graph.terminals()] == ["verify_scan"]


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
        resolve_workflow("surgical_reach_psm", "policy")
