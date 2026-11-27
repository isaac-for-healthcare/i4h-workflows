# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Physical action contract exercised through the shared drive implementation, without Isaac."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from i4h_arena.medical.catheter_drive import RouteRailedIntroducer  # noqa: E402


def test_decimated_actions_integrate_once_per_physics_tick_and_hold_limits():
    source = Path(__file__).parents[1] / "i4h_arena/embodiments/catheter.py"
    tree = ast.parse(source.read_text())
    original = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CatheterVelocityAction"
    )
    methods = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in ("apply_actions", "process_actions", "_feed")
    ]
    cls = ast.ClassDef(name="Drive", bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), namespace)
    drive = namespace["Drive"]()
    drive._env = SimpleNamespace(physics_dt=1 / 120, step_dt=1 / 30)
    drive.cfg = SimpleNamespace(
        max_insertion_velocity_mps=0.06, max_rotation_rate_radps=1.5, max_tip_bend_rate_radps=1.5, max_tip_bend_rad=1.5
    )
    drive._raw_actions = torch.zeros(2, 3)
    drive._processed_actions = torch.zeros_like(drive._raw_actions)
    drive._tip_bend_angle = torch.zeros(2)
    drive._rail = RouteRailedIntroducer(torch.tensor([[0.0, 0, 0], [0.1, 0, 0]]), 2)
    poses = []
    drive._asset = SimpleNamespace(
        place_proximal=lambda position, quat, spent, dt: poses.append(position.clone()), set_tip_bend=lambda angle: None
    )
    for _ in range(30):
        drive.process_actions(torch.tensor([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]]))
        for _ in range(4):
            drive.apply_actions()
    torch.testing.assert_close(poses[-1][:, 0], torch.tensor([0.06, 0.0]))
    torch.testing.assert_close(drive._tip_bend_angle, torch.tensor([1.5, -1.5]))
    for _ in range(120):
        drive.apply_actions()
    torch.testing.assert_close(poses[-1][:, 0], torch.tensor([0.1, 0.0]))
    torch.testing.assert_close(drive._tip_bend_angle, torch.tensor([1.5, -1.5]))
