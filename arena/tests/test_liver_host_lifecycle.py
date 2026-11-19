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
        / "third_party/i4h-numerical-sim-7edecea/numerical_sim/isaaclab/src/i4h_isaaclab/manager.py"
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
