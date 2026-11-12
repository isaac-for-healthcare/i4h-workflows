# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nsight Systems capture of the runner's steady-state stepping.

``nsys profile`` over a whole workflow run is dominated by what the run spends
most of its wall clock on, which is Kit booting, assets loading and kernels
compiling on first use. None of that is the per-step cost anyone profiling a
workflow is asking about. ``--nsys-range`` instead brackets the stepping loop
with ``cudaProfilerStart``/``cudaProfilerStop`` so that

    nsys profile --capture-range=cudaProfilerApi ./run.sh <workflow> --nsys-range

records only the steps after warmup, and ``--nsys-warmup`` chooses how many
steps warmup is.

The phase ranges exist because the per-step cost has to be *attributed* to be
actionable. A catheter environment steps an XPBD rod against a deformable
vessel and renders fluoroscopy, and the open question for vectorized RL is
which of those dominates -- a kernel timeline alone does not say which phase a
kernel was launched from. ``phase`` wraps each one in NVTX so the capture reads
as tick, step and record rather than as an undifferentiated kernel list.

Everything here is inert unless ``--nsys-range`` is passed: ``CaptureRange``
holds ``enabled=False``, ``phase`` yields without touching NVTX, and the step
counter is never advanced.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("i4h_arena.profiling")

#: Steps skipped before the capture opens. Twenty is the same default the
#: numerical_sim runner uses for its untimed steps, and it is enough to clear
#: first-use kernel compilation and lazy tensor allocation in the task terms.
DEFAULT_WARMUP_STEPS = 20


@contextmanager
def phase(name: str, *, enabled: bool = True) -> Iterator[None]:
    """Mark ``name`` as an NVTX range for the duration of the block.

    Disabled, this is a bare yield: the workflow loop runs this three times per
    step whether or not anyone is profiling, so it must not cost a CUDA call or
    an import when it is off.
    """
    if not enabled:
        yield
        return
    import torch

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class CaptureRange:
    """Opens the profiler's capture range once stepping reaches steady state.

    The range is global to the process, as ``cudaProfilerStart`` is, so this
    counts steps across every episode rather than restarting per episode. A run
    that is reset and stepped again stays inside one capture, which is what
    makes episode boundaries visible in the timeline instead of absent from it.
    """

    def __init__(self, *, enabled: bool = False, warmup_steps: int = DEFAULT_WARMUP_STEPS) -> None:
        """
        Args:
            enabled: Whether ``--nsys-range`` was passed. False makes every
                method a no-op.
            warmup_steps: Steps to skip before opening the capture.

        Raises:
            ValueError: ``warmup_steps`` is negative.
            RuntimeError: Profiling was requested without a CUDA device, which
                would otherwise produce an empty report after a full run.
        """
        warmup_steps = int(warmup_steps)
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        self._enabled = bool(enabled)
        self._warmup_steps = warmup_steps
        self._steps = 0
        self._started = False
        if self._enabled:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("--nsys-range needs a CUDA device; none is available")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def started(self) -> bool:
        """Whether the capture range is currently open."""
        return self._started

    def step(self) -> None:
        """Count one physics step, opening the capture when warmup is done.

        Called after the step rather than before it, so the step that crosses
        the warmup threshold is itself excluded. The first step inside the
        capture is then one whose caches are all warm.
        """
        if not self._enabled or self._started:
            return
        self._steps += 1
        if self._steps <= self._warmup_steps:
            return
        import torch

        torch.cuda.profiler.start()
        self._started = True
        logger.info("nsys capture range opened after %s warmup steps", self._warmup_steps)

    def close(self) -> None:
        """Close the capture range if it was opened.

        Idempotent, because the runner closes this from a ``finally`` and an
        aborted run may have never reached the threshold at all.
        """
        if not self._started:
            return
        import torch

        torch.cuda.profiler.stop()
        self._started = False
        logger.info("nsys capture range closed")
