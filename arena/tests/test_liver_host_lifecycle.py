# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual dependency patch without booting Kit."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_mixed_reset_preserves_restored_history():
    path = (
        Path(__file__).parents[2]
        / "third_party/i4h-numerical-sim-267f48a/numerical_sim/isaaclab/src/i4h_isaaclab/manager.py"
    )
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "I4hHostMixin"
    )
    namespace = {"torch": torch, "I4hComponent": object}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    trace = []

    def event(label):
        return lambda ids=None, **kw: trace.append(
            (label, (ids if ids is not None else kw.get("env_ids", torch.tensor([]))).tolist())
        )

    class Base:
        def _reset_idx(self, ids):
            trace.append(("lab_events", ids.tolist()))

    class Host(namespace["I4hHostMixin"], Base):
        def i4h_restore_ids(self, ids):
            return ids[ids == 1]

        def i4h_restore_state(self, ids):
            event("restore")(ids)

        def i4h_save_terminal(self, ids):
            event("terminal")(ids)

        def i4h_after_reset(self, ids):
            event("arena_reset")(ids)

    host = Host.__new__(Host)
    host.i4h = SimpleNamespace(
        reset=event("physical"),
        sync_views=lambda: trace.append(("sync", [])),
        initialize_particle_history=event("history"),
        reset_procedure=event("procedure"),
    )
    host.i4h_task = None
    host.extras = {}
    host._reset_idx(torch.tensor([0, 1, 2]))
    assert trace == [
        ("terminal", [0, 1, 2]),
        ("physical", [0, 2]),
        ("sync", []),
        ("lab_events", [0, 1, 2]),
        ("history", [0, 2]),
        ("arena_reset", [0, 2]),
        ("restore", [1]),
        ("sync", []),
    ]


def test_merged_callbacks_run_after_procedure_for_each_decimated_physics_step():
    from i4h_common.paths import numerical_source

    source = numerical_source() / "isaaclab/src/i4h_isaaclab/newton_manager.py"
    cls = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "ConstructionContext"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "physics_step")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    trace = []
    binding = SimpleNamespace(
        procedure=SimpleNamespace(update=lambda dt: trace.append(("procedure", dt))),
        dt=1 / 60,
        post_physics=[lambda: trace.append(("first", None)), lambda: trace.append(("second", None))],
    )
    context = SimpleNamespace(binding=binding)
    for _ in range(2):
        namespace["physics_step"](context)
    assert trace == [("procedure", 1 / 60), ("first", None), ("second", None)] * 2
