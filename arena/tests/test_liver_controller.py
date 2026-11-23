# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve the pinned reference's physical commands without a second phase reducer."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from i4h_arena.medical.liver_controller import LiverReferenceController


def test_commands_match_pinned_reference():
    root = Path(__file__).parents[2]
    reference = root / "third_party/i4h-physics-simulation/numerical_sim/examples/liver_grasp/manual_teacher.py"
    spec = importlib.util.spec_from_file_location("pinned_manual_teacher", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    examples = json.loads((root / "arena/i4h_arena/envcfg/liver_reach_retract/calibration.json").read_text())[
        "examples"
    ]
    for example in examples:
        initial = np.zeros(10, np.float32)
        initial[:3] = np.array(example["capture_root"]) + [0.0, 0.0, 0.03]
        workspace = np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
        old = module.ManualGeometryTeacher(initial, workspace, example)
        new = LiverReferenceController(initial, workspace, example)
        target = np.array(example["capture_particle_positions"][0])
        measurement = {"geometry": np.array([target, [0.0, 0.0, 1.0], target, target]), "gap": 0.0}
        phases = (
            ["approach"] * 70
            + ["capture"] * 8
            + ["qualify"] * 15
            + ["retract"] * 20
            + ["hold"] * 30
            + ["release"]
            + ["relax"] * 30
        )
        destination = target + [0.0, 0.02, 0.0]
        for phase in phases:
            old.phase = "capture" if phase == "approach" and old.arrival_ticks >= 3 else phase
            old.destination = destination
            expected = old.command(measurement, 0.002, 0.001, 0.2).copy()
            actual = new.command(measurement, phase, destination).copy()
            np.testing.assert_array_equal(actual, expected)
        assert not hasattr(new, "phase") and not hasattr(new, "step") and not hasattr(new, "failure")
