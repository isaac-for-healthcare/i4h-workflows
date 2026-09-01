# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--checkpoint`` / ``--prompt`` reaching the nodes that can use them.

Applied at the workflow level rather than threaded through each workflow module's
signature, so "evaluate the checkpoint I just trained" works for all eleven
workflows without any of them opting in.
"""

from __future__ import annotations

from conftest import Counter

from i4h_common.manifest import BackendSpec, TaskSpec
from i4h_engine.graph import Node, TaskGraph, node
from i4h_engine.loader import apply_overrides


def _remote(name: str) -> Node:
    spec = TaskSpec(
        project="gr00t_n15",
        name=name,
        runtime="remote",
        backend=BackendSpec(project="tasks/gr00t_n15", entry="x:main"),
        outputs={"success": "bool"},
    )
    return Node(f"gr00t_n15/{name}", spec=spec)


def test_checkpoint_and_prompt_reach_a_remote_node():
    policy = _remote("scissor_pick_and_place")
    workflow = TaskGraph().flow(policy)
    touched = apply_overrides(workflow, checkpoint="/tmp/ckpt-1000", prompt="Put it in the tray")
    assert touched == ("scissor_pick_and_place",)
    assert policy.params["checkpoint"] == "/tmp/ckpt-1000"
    assert policy.params["prompt"] == "Put it in the tray"


def test_scripted_nodes_are_left_alone():
    # A keyframe task has no checkpoint; attaching one would make `show` lie.
    scripted = node(Counter(1, name="keyframes"))
    workflow = TaskGraph().flow(scripted)
    assert apply_overrides(workflow, checkpoint="/tmp/ckpt") == ()
    assert "checkpoint" not in scripted.params


def test_checkpoint_reaches_an_inprocess_model_task():
    modeled = Node(
        "rsl_rl/ultrasound_probe_reach",
        spec=TaskSpec(
            project="rsl_rl",
            name="ultrasound_probe_reach",
            runtime="inprocess",
            impl="x:UltrasoundProbeReachPolicy",
            model={"family": "rsl_rl", "format": "torchscript"},
        ),
    )
    workflow = TaskGraph().flow(modeled)

    assert apply_overrides(
        workflow,
        checkpoint="/tmp/policy.pt",
        model_device="cuda:1",
    ) == ("ultrasound_probe_reach",)
    assert modeled.params["checkpoint"] == "/tmp/policy.pt"
    assert modeled.params["device"] == "cuda:1"


def test_mixed_workflow_touches_only_the_remote_half():
    scripted = node(Counter(1, name="approach"))
    policy = _remote("scissor_pick_and_place")
    workflow = TaskGraph().flow(scripted >> policy)
    assert apply_overrides(workflow, checkpoint="/tmp/ckpt") == ("scissor_pick_and_place",)
    assert "checkpoint" not in scripted.params
    assert policy.params["checkpoint"] == "/tmp/ckpt"


def test_every_remote_node_is_updated():
    first = _remote("surgical_lift_block")
    second = Node(
        "gr00t_n15/surgical_lift_needle",
        id="second",
        spec=TaskSpec(
            project="gr00t_n15",
            name="surgical_lift_needle",
            runtime="remote",
            backend=BackendSpec(project="tasks/gr00t_n15", entry="x:main"),
        ),
    )
    workflow = TaskGraph().flow(first >> second)
    assert set(apply_overrides(workflow, prompt="lift it")) == {"surgical_lift_block", "second"}


def test_no_overrides_is_a_no_op():
    policy = _remote("scissor_pick_and_place")
    workflow = TaskGraph().flow(policy)
    assert apply_overrides(workflow) == ()
    assert policy.params == {}


def test_partial_override_leaves_the_other_alone():
    policy = _remote("scissor_pick_and_place")
    workflow = TaskGraph().flow(policy)
    apply_overrides(workflow, checkpoint="/tmp/ckpt")
    assert policy.params["checkpoint"] == "/tmp/ckpt"
    assert "prompt" not in policy.params


def test_extra_overrides_pass_through():
    policy = _remote("scissor_pick_and_place")
    workflow = TaskGraph().flow(policy)
    apply_overrides(workflow, extra={"max_steps": 300})
    assert policy.params["max_steps"] == 300


def test_override_survives_into_the_task_spec_message():
    """The override has to reach the backend, not just the node dict."""
    from conftest import FakeActuation, FakeScene

    from i4h_common.bus.inproc import InProcBus
    from i4h_common.bus.keys import Keys
    from i4h_common.bus.messages import TaskSpecMsg, TaskStatusMsg, decode, encode
    from i4h_engine.remote import RemoteTask
    from i4h_engine.task import TickContext

    policy = _remote("scissor_pick_and_place")
    workflow = TaskGraph().flow(policy)
    apply_overrides(workflow, checkpoint="/tmp/ckpt-1000", prompt="new instruction")

    bus, keys = InProcBus(), Keys("run")
    seen: list[TaskSpecMsg] = []
    bus.subscribe(f"{keys.root}/task/*/spec", lambda _k, p: seen.append(decode(p, TaskSpecMsg)))
    bus.subscribe(
        f"{keys.root}/task/*/spec",
        lambda _k, _p: bus.publish(
            keys.task_status(f"{policy.spec.name}-0"),
            encode(
                TaskStatusMsg(
                    task_uid=f"{policy.spec.name}-0",
                    status="ready",
                    action_space="joint_position",
                    action_layout="joints",
                    action_dof=6,
                )
            ),
        ),
    )
    ctx = TickContext(scene=FakeScene(dof=6), act=FakeActuation(dof=6), dt=1 / 60, bus=bus, run_id="run")
    RemoteTask(policy.spec, keys=keys, **policy.params).on_enter(ctx, None)

    assert seen[0].checkpoint == "/tmp/ckpt-1000"
    assert seen[0].prompt == "new instruction"
