# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for workflow predicates that span simulator resets."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from i4h_engine.loader import load_workflow_module
from i4h_engine.task import TickContext
from i4h_tasks.basic.predicates import env_success

endoluminal = load_workflow_module("endoluminal_navigation").workflow


class _CatheterScene:
    """Reports the simulator term, which is absent without a patient twin."""

    def __init__(self, arrived: np.ndarray) -> None:
        self.arrived = arrived

    def termination(self, _name: str) -> np.ndarray:
        return self.arrived


def test_env_success_reads_the_simulator_arrival_term() -> None:
    ctx = TickContext(scene=_CatheterScene(np.array([False])), act=SimpleNamespace())
    assert not env_success(ctx).any()

    ctx.scene.arrived = np.array([True])
    assert env_success(ctx).all()


def test_env_success_is_unmet_without_a_centerline() -> None:
    """No twin means no arrival term, so the scene view returns absent-term zeros."""
    ctx = TickContext(scene=_CatheterScene(np.zeros(1, dtype=bool)), act=SimpleNamespace())

    assert not env_success(ctx).any()


def test_endoluminal_exposes_replay() -> None:
    assert "replay" in endoluminal.modes
