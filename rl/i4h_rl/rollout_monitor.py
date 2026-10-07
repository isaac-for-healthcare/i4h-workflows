# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, eagerly-flushed progress artifacts for long RLinf rollouts."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


class _ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


def _summary_writer(log_dir: Path) -> _ScalarWriter:
    # TensorBoard belongs to the heavy GR00T environment, not the lightweight
    # profile-discovery environment, so keep this import lazy.
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(str(log_dir))


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


class RolloutProgressMonitor:
    """Write one TensorBoard point and one JSON snapshot per completed rollout."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        rollouts_per_update: int,
        writer_factory: Callable[[Path], _ScalarWriter] = _summary_writer,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if rollouts_per_update <= 0:
            raise ValueError("rollouts_per_update must be positive")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.run_dir / "rollout_progress.json"
        self._temporary_path = self.run_dir / ".rollout_progress.json.tmp"
        self.rollouts_per_update = int(rollouts_per_update)
        self._writer = writer_factory(self.run_dir / "tensorboard")
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._last_rollout_started = monotonic()
        self._completed_total = 0
        self._current_update_durations: list[float] = []
        # Make the run visible in TensorBoard immediately, before a long first
        # rollout has produced its first real metric point.
        self._writer.add_scalar("monitor/rollout_index", 0.0, 0)
        self._writer.add_scalar("monitor/progress_fraction", 0.0, 0)
        self._writer.flush()
        self._write_progress(status="waiting_for_rollout", metrics={})

    def _write_progress(self, *, status: str, metrics: dict[str, float | None]) -> None:
        rollout_index = (self._completed_total - 1) % self.rollouts_per_update + 1 if self._completed_total else 0
        update_index = (self._completed_total - 1) // self.rollouts_per_update if self._completed_total else 0
        average_duration = (
            sum(self._current_update_durations) / len(self._current_update_durations)
            if self._current_update_durations
            else None
        )
        remaining = self.rollouts_per_update - rollout_index
        payload = {
            "schema_version": 1,
            "status": status,
            "updated_at": self._wall_time().isoformat(),
            "update_index": update_index,
            "rollout_index": rollout_index,
            "rollouts_per_update": self.rollouts_per_update,
            "completed_rollouts_total": self._completed_total,
            "estimated_seconds_to_update": (average_duration * remaining if average_duration is not None else None),
            "metrics": metrics,
        }
        self._temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(self._temporary_path, self.progress_path)

    def record(self, **metrics: float) -> None:
        now = self._monotonic()
        duration = max(0.0, now - self._last_rollout_started)
        self._last_rollout_started = now
        self._completed_total += 1
        rollout_index = (self._completed_total - 1) % self.rollouts_per_update + 1
        if rollout_index == 1:
            self._current_update_durations = []
        self._current_update_durations.append(duration)

        finite_metrics = {key: _finite(value) for key, value in metrics.items()}
        scalars = {
            "monitor/rollout_index": float(rollout_index),
            "monitor/progress_fraction": rollout_index / self.rollouts_per_update,
            "monitor/rollout_duration_seconds": duration,
            **{f"monitor/{key}": value for key, value in finite_metrics.items()},
        }
        average_duration = sum(self._current_update_durations) / len(self._current_update_durations)
        scalars["monitor/estimated_seconds_to_update"] = average_duration * (self.rollouts_per_update - rollout_index)
        for tag, value in scalars.items():
            if value is not None:
                self._writer.add_scalar(tag, value, self._completed_total)
        self._writer.flush()

        status = (
            "rollouts_complete_waiting_for_optimizer"
            if rollout_index == self.rollouts_per_update
            else "collecting_rollouts"
        )
        self._write_progress(status=status, metrics=finite_metrics)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
