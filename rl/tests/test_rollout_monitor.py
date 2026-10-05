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


# --------------------------------------------------------------------------- #
# Which step records, and which worker writes
# --------------------------------------------------------------------------- #
def _env_cfg(**values):
    from types import SimpleNamespace

    return SimpleNamespace(**values)


def test_the_threshold_tracks_the_shorter_of_the_two_budgets():
    """A rollout epoch ends at whichever limit comes first.

    `auto_reset: false` makes RLinf reset at the top of every epoch, so the
    monitor does get one chance per rollout -- but the chance is only as long as
    the epoch, and reading the episode length alone made the threshold
    unreachable as soon as the epoch was the shorter of the two. Nothing in
    RLinf relates the keys, so that would have recorded nothing, silently.
    """
    from i4h_rl.adapters.endoluminal_navigation import monitor_step_threshold

    # The shipped profiles, where the two agree.
    assert monitor_step_threshold(_env_cfg(max_episode_steps=600, max_steps_per_rollout_epoch=600)) == 600
    # A shortened rollout against an unchanged episode length.
    assert monitor_step_threshold(_env_cfg(max_episode_steps=600, max_steps_per_rollout_epoch=300)) == 300
    # And the other way, where the episode is what ends collection.
    assert monitor_step_threshold(_env_cfg(max_episode_steps=200, max_steps_per_rollout_epoch=600)) == 200


def test_a_config_declaring_no_horizon_is_refused():
    """Better than a threshold of zero, which records on the first step of every
    epoch and reports averages over one step as if they were the rollout's."""
    import pytest

    from i4h_rl.adapters.endoluminal_navigation import monitor_step_threshold

    with pytest.raises(ValueError, match="max_episode_steps"):
        monitor_step_threshold(_env_cfg(max_episode_steps=0, max_steps_per_rollout_epoch=0))


def test_only_the_first_env_worker_writes_the_run_directory():
    """The monitor writes one fixed `rollout_progress.json` and one TensorBoard
    directory, so a second worker would overwrite the first rather than add to
    it. One worker exists today -- a single simulator behind a single bridge
    socket, `env` placed on one node -- so this guards a placement change.
    """
    from i4h_rl.adapters.endoluminal_navigation import monitor_owns_run_dir

    assert monitor_owns_run_dir(0)
    assert not monitor_owns_run_dir(1)
    assert not monitor_owns_run_dir(7)
