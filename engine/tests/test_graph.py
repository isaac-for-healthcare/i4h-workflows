# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from conftest import Consumer, Counter, Producer

from i4h_engine.graph import TaskGraph, node


def test_chain_builds_linear_edges():
    a, b, c = node(Counter(name="a")), node(Counter(name="b")), node(Counter(name="c"))
    workflow = TaskGraph().flow(a >> b >> c)
    assert [n.id for n in workflow.nodes] == ["a", "b", "c"]
    assert set(workflow.edges) == {("a", "b"), ("b", "c")} or [(e.src, e.dst) for e in workflow.edges] == [
        ("a", "b"),
        ("b", "c"),
    ]


def test_task_instance_can_start_a_chain():
    # `Task.__rshift__` wraps implicitly, so workflows read without node() noise.
    workflow = TaskGraph().flow(Counter(name="a") >> Counter(name="b"))
    assert [n.id for n in workflow.nodes] == ["a", "b"]


def test_fan_out_and_fan_in():
    root = node(Counter(name="root"))
    left, right = node(Counter(name="left")), node(Counter(name="right"))
    join = node(Counter(name="join"))
    workflow = TaskGraph().flow(root >> (left, right) >> join)
    edges = {(e.src, e.dst) for e in workflow.edges}
    assert edges == {("root", "left"), ("root", "right"), ("left", "join"), ("right", "join")}
    assert workflow.predecessors("join") == ("left", "right")


def test_roots_and_terminals():
    workflow = TaskGraph().flow(Counter(name="a") >> Counter(name="b"))
    assert [n.id for n in workflow.roots()] == ["a"]
    assert [n.id for n in workflow.terminals()] == ["b"]


def test_topological_order():
    root = node(Counter(name="root"))
    workflow = TaskGraph().flow(root >> (node(Counter(name="l")), node(Counter(name="r"))) >> node(Counter(name="j")))
    order = workflow.topological_order()
    assert order[0] == "root"
    assert order[-1] == "j"


def test_cycle_detected():
    a, b = node(Counter(name="a")), node(Counter(name="b"))
    workflow = TaskGraph().flow(a >> b)
    workflow._edges.append(type(workflow.edges[0])(src="b", dst="a"))
    assert workflow.find_cycle()
    with pytest.raises(ValueError, match="cycle"):
        workflow.topological_order()


def test_duplicate_node_id_rejected():
    workflow = TaskGraph().flow(node(Counter(name="a")))
    with pytest.raises(ValueError, match="already has a different node"):
        workflow.flow(node(Counter(name="a")))


def test_explicit_id_disambiguates():
    workflow = TaskGraph().flow(node(Counter(), id="first") >> node(Counter(), id="second"))
    assert [n.id for n in workflow.nodes] == ["first", "second"]


def test_wire_requires_correct_directions():
    producer, consumer = node(Producer(name="p")), node(Consumer(name="c"))
    workflow = TaskGraph().flow(producer >> consumer)
    with pytest.raises(ValueError, match="must be an output"):
        workflow.wire(consumer.in_.target, consumer.in_.target)
    with pytest.raises(ValueError, match="must be an input"):
        workflow.wire(producer.out.pose, producer.out.pose)


def test_wire_rejects_double_binding():
    p1, p2 = node(Producer(name="p1")), node(Producer(name="p2"))
    consumer = node(Consumer(name="c"))
    workflow = TaskGraph().flow(p1 >> p2 >> consumer)
    workflow.wire(p1.out.pose, consumer.in_.target)
    with pytest.raises(ValueError, match="already wired"):
        workflow.wire(p2.out.pose, consumer.in_.target)


def test_wire_rejects_unknown_node():
    stray = node(Producer(name="stray"))
    consumer = node(Consumer(name="c"))
    workflow = TaskGraph().flow(consumer)
    with pytest.raises(KeyError, match="not in the task graph"):
        workflow.wire(stray.out.pose, consumer.in_.target)


def test_unknown_port_raises_at_build_time():
    producer = node(Producer(name="p"))
    with pytest.raises(AttributeError, match="no output port 'psoe'"):
        _ = producer.out.psoe


def test_required_inputs_exclude_defaulted_fields():
    consumer = node(Consumer(name="c"))
    # Consumer.Inputs.scale has a default, so it is optional.
    assert consumer.required_inputs == ("target",)
    assert consumer.input_ports["scale"].endswith("?")


def test_on_failure_validates():
    a = node(Counter(name="a"))
    workflow = TaskGraph().flow(a)
    with pytest.raises(ValueError, match="unknown failure action"):
        workflow.on_failure(a, "explode")
    with pytest.raises(KeyError, match="not in the task graph"):
        workflow.on_failure("ghost", "abort")
