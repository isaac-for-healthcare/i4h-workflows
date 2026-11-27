# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exact hosted-controller parity across batching, phases and selected restore."""

import ast
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from fixtures.liver_controller_reference import reference_teacher_action

from i4h_arena.medical.liver_controller import LiverReferenceController

ROOT = Path(__file__).resolve().parents[2]


def current_teacher_action():
    source = ROOT / "arena/i4h_arena/envcfg/liver_reach_retract/runtime.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "LiverRuntime"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "teacher_action")
    namespace = dict(np=np, torch=torch)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["teacher_action"]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("worlds", [1, 8, 16])
def test_hosted_commands_and_history_match_former_transfers(device, worlds):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    examples = json.loads((ROOT / "arena/i4h_arena/envcfg/liver_reach_retract/calibration.json").read_text())[
        "examples"
    ]
    initial = np.zeros((worlds, 10), np.float32)
    geometry = np.zeros((worlds, 4, 3), np.float32)
    for e in range(worlds):
        example = examples[e % len(examples)]
        initial[e, :3] = np.asarray(example["capture_root"]) + [0, 0, 0.03]
        initial[e, 3:7] = example["capture_quaternion_xyzw"]
        geometry[e, 0] = example["capture_particle_positions"][0]

    def make(method):
        def tensor(value):
            return torch.as_tensor(value.copy(), device=device)

        runtime = SimpleNamespace(
            env=SimpleNamespace(num_envs=worlds, device=device),
            component=SimpleNamespace(sync_views=lambda: None),
            physical=tensor(initial),
            controller_action=tensor(initial),
            controller_arrival=tensor(np.arange(worlds, dtype=np.int32)[:, None] % 4),
            objective=SimpleNamespace(
                s=tensor(np.zeros((worlds, 9), np.int32)), f=tensor(np.zeros((worlds, 12), np.float32))
            ),
            views={"geometry": tensor(geometry)},
            teachers=[LiverReferenceController(initial[e], None, examples[e % len(examples)]) for e in range(worlds)],
        )
        runtime.probe = SimpleNamespace(checks=0, reads=0)

        def check():
            runtime.probe.checks += 1

        def read():
            check()
            runtime.probe.reads += 1
            return {"geometry": runtime.views["geometry"].cpu().numpy()}

        runtime.probe.check = check
        runtime.probe.read = read
        runtime.teacher_action = MethodType(method, runtime)
        return runtime

    reference, candidate = make(reference_teacher_action), make(current_teacher_action())
    retained = []
    for tick in range(24):
        phases = (np.arange(worlds) + tick) % 9
        for runtime in (reference, candidate):
            runtime.objective.s[:, 0] = torch.as_tensor(phases, device=device)
            runtime.objective.s[:, 4] = torch.as_tensor(np.where(phases < 2, -1, 3), device=device)
            runtime.objective.f[:, :3] = runtime.views["geometry"][:, 0] + torch.tensor([0, 0.02, 0], device=device)
            if tick == 12:
                # A selected-row restore replaces serialized history, including
                # arrival hysteresis; Python teacher state must follow it.
                runtime.controller_action[worlds // 2] = torch.as_tensor(initial[worlds // 2], device=device)
                runtime.controller_arrival[worlds // 2, 0] = 3
        before = {key: value.clone() for key, value in candidate.views.items()}
        expected, actual = reference.teacher_action(), candidate.teacher_action()
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
        assert actual.shape == (worlds, 4) and actual.dtype == np.float32
        for name in ("controller_action", "controller_arrival", "physical"):
            torch.testing.assert_close(getattr(candidate, name), getattr(reference, name), rtol=0, atol=0)
        for key, value in before.items():
            torch.testing.assert_close(candidate.views[key], value, rtol=0, atol=0)
        for a, b in zip(candidate.teachers, reference.teachers, strict=True):
            np.testing.assert_array_equal(a.action, b.action)
            assert a.arrival_ticks == b.arrival_ticks
        for value, copy in retained:
            np.testing.assert_array_equal(value, copy)
        retained.append((actual, actual.copy()))
    assert candidate.probe.checks == reference.probe.checks == 24
    assert candidate.probe.reads == 0 and reference.probe.reads == 24
