# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The capture range and its NVTX phases, on CPU.

Neither the capture nor the ranges can be observed here, so these check the
property that makes them safe to leave in the loop: disabled, nothing is
called, and the step counter never moves.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from i4h_arena.profiling import DEFAULT_WARMUP_STEPS, CaptureRange, phase

RUNNER = Path(__file__).resolve().parents[1] / "i4h_arena" / "runner.py"


def test_capture_is_disabled_by_default() -> None:
    """The loop calls into this every step, so the default has to cost nothing."""
    capture = CaptureRange()

    assert not capture.enabled
    assert not capture.started


def test_a_disabled_capture_never_opens() -> None:
    capture = CaptureRange(enabled=False, warmup_steps=0)

    for _ in range(5):
        capture.step()

    assert not capture.started


def test_a_disabled_capture_closes_without_a_cuda_call() -> None:
    """Closed from a ``finally``, including on a run that never stepped."""
    CaptureRange().close()


def test_disabled_phases_do_not_touch_nvtx() -> None:
    with phase("tick", enabled=False):
        pass


def test_warmup_defaults_to_the_numerical_sim_convention() -> None:
    assert DEFAULT_WARMUP_STEPS == 20
    assert CaptureRange()._warmup_steps == DEFAULT_WARMUP_STEPS


def test_a_negative_warmup_is_rejected() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        CaptureRange(warmup_steps=-1)


def test_requesting_a_capture_without_cuda_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rather than after a full run that produced an empty report."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="--nsys-range needs a CUDA device"):
        CaptureRange(enabled=True)


def _phase_names(function: str) -> list[str]:
    """NVTX phase names opened inside ``function`` of the runner."""
    tree = ast.parse(RUNNER.read_text())
    target = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function)
    return [
        item.context_expr.args[0].value
        for node in ast.walk(target)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and getattr(item.context_expr.func, "id", None) == "phase"
        and item.context_expr.args
        and isinstance(item.context_expr.args[0], ast.Constant)
    ]


def test_the_step_loop_ranges_the_phases_it_needs_to_attribute() -> None:
    """Deciding an action, applying it and recording it are separable costs.

    Without the split, a capture says the workflow spent its time in CUDA
    kernels without saying whether inference, physics, rendering or the
    recording read-back launched them -- which is the question profiling a
    vectorized catheter environment is asked to answer.
    """
    assert set(_phase_names("_run_episode")) == {"tick", "env.step", "record"}


def test_the_capture_advances_once_per_completed_step() -> None:
    """One call, after the step, so a failed step is not counted as warmup."""
    source = RUNNER.read_text()

    assert source.count("self.capture.step()") == 1
    assert source.count("self.capture.close()") == 1
