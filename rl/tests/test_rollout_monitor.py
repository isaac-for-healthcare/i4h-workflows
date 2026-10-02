# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from i4h_rl.rollout_monitor import RolloutProgressMonitor


class FakeWriter:
    def __init__(self):
        self.scalars = []
        self.flushes = 0
        self.closed = False

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, scalar_value, global_step))

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


def test_monitor_flushes_tensorboard_and_replaces_progress_json(tmp_path):
    writer = FakeWriter()
    times = iter((10.0, 32.0, 54.0))
    monitor = RolloutProgressMonitor(
        tmp_path,
        rollouts_per_update=8,
        writer_factory=lambda _path: writer,
        monotonic=lambda: next(times),
        wall_time=lambda: datetime(2026, 10, 2, tzinfo=UTC),
    )

    initial = json.loads((tmp_path / "rollout_progress.json").read_text())
    assert initial["status"] == "waiting_for_rollout"
    assert initial["rollout_index"] == 0
    assert ("monitor/rollout_index", 0.0, 0) in writer.scalars
    assert ("monitor/progress_fraction", 0.0, 0) in writer.scalars
    assert writer.flushes == 1

    monitor.record(mean_return=4.0, mean_reward=0.5, success_rate=1.0)
    progress = json.loads((tmp_path / "rollout_progress.json").read_text())
    assert progress["status"] == "collecting_rollouts"
    assert progress["rollout_index"] == 1
    assert progress["estimated_seconds_to_update"] == pytest.approx(154.0)
    assert progress["metrics"]["mean_return"] == 4.0
    assert ("monitor/mean_return", 4.0, 1) in writer.scalars
    assert writer.flushes == 2
    assert not (tmp_path / ".rollout_progress.json.tmp").exists()

    monitor.close()
    assert writer.flushes == 3
    assert writer.closed


def test_monitor_marks_the_eighth_rollout_waiting_for_optimizer(tmp_path):
    writer = FakeWriter()
    tick = iter(float(value) for value in range(9))
    monitor = RolloutProgressMonitor(
        tmp_path,
        rollouts_per_update=8,
        writer_factory=lambda _path: writer,
        monotonic=lambda: next(tick),
    )

    for index in range(8):
        monitor.record(mean_return=float(index))

    progress = json.loads((tmp_path / "rollout_progress.json").read_text())
    assert progress["status"] == "rollouts_complete_waiting_for_optimizer"
    assert progress["rollout_index"] == 8
    assert progress["completed_rollouts_total"] == 8
    assert progress["estimated_seconds_to_update"] == 0.0
    assert ("monitor/progress_fraction", 1.0, 8) in writer.scalars
