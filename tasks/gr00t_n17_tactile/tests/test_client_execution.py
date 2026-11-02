# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunk execution with fakes: the target math, the tail edits and the two worker threads.

The policy is a recording fake and every sleep is injected, so the whole file
runs in the light venv in well under a second.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17_tactile.client import execution as ex  # noqa: E402

DIMS = {"left_arm": 7, "right_arm": 7, "left_hand": 22, "right_hand": 22}


def _chunk(length: int = 32, base: float = 0.0, keys: tuple[str, ...] = ex.ACTION_KEYS) -> ex.ActionChunk:
    """Row ``t`` of every key holds ``base + t`` in every joint, so an index error shows in the value."""
    rows = base + np.arange(length, dtype=np.float32)
    return {k: np.repeat(rows[None, :, None], DIMS[k], axis=2) for k in keys}


def _wait_until(pred, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.001)
    return pred()


class _FakePolicy:
    """Records every call with its thread; optionally slow or failing once."""

    def __init__(self, chunk: ex.ActionChunk | None = None, delay_s: float = 0.0, fail_first: bool = False) -> None:
        self.chunk = _chunk() if chunk is None else chunk
        self.delay_s = delay_s
        self.fail_first = fail_first
        self.calls: list[tuple[dict | None, str]] = []
        self.resets = 0
        self.lock = threading.Lock()

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        with self.lock:
            if self.fail_first:
                self.fail_first = False
                raise RuntimeError("boom")
            self.calls.append((options, threading.current_thread().name))
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.chunk, {"options": options}

    def reset(self, options: dict | None = None) -> dict:
        with self.lock:
            self.resets += 1
        return {}

    @property
    def n_calls(self) -> int:
        with self.lock:
            return len(self.calls)


# ── chunk arithmetic ──────────────────────────────────────────────────────────


def test_start_step_for_skips_the_elapsed_steps_and_clamps() -> None:
    dt = 1.0 / 30.0
    assert ex.start_step_for(0.0, dt, 32) == 0
    assert ex.start_step_for(-1.0, dt, 32) == 0
    assert ex.start_step_for(3.5 * dt, dt, 32) == 3
    assert ex.start_step_for(10.0, dt, 32) == 31


def test_plan_horizon_is_the_tightest_of_three_bounds() -> None:
    assert ex.plan_horizon(16, 32, 0, 100) == 16
    assert ex.plan_horizon(16, 32, 20, 100) == 12
    assert ex.plan_horizon(16, 32, 0, 5) == 5
    assert ex.plan_horizon(16, 32, 31, 100) == 1
    assert ex.plan_horizon(16, 32, 32, 100) == 0


def test_chunk_length_prefers_left_arm_then_any_action_key_then_default() -> None:
    assert ex.chunk_length(_chunk(32), 16) == 32
    assert ex.chunk_length(_chunk(24, keys=("right_hand",)), 16) == 24
    assert ex.chunk_length({"other": np.zeros((1, 9, 3))}, 16) == 16


def test_copy_chunk_is_a_private_copy_and_passes_non_arrays_through() -> None:
    src = _chunk(8)
    src["meta"] = "keep"
    dup = ex.copy_chunk(src)
    dup["left_arm"][0, 3, :] = 99.0
    assert src["left_arm"][0, 3, 0] == 3.0
    assert dup["meta"] == "keep"
    assert not np.shares_memory(dup["right_hand"], src["right_hand"])


# ── smoothing numbers ─────────────────────────────────────────────────────────


def test_arm_blend_weights_decay_exponentially() -> None:
    w_new, w_old = ex.arm_blend_weights(0, 4, 2.0)
    assert (w_new, w_old) == (0.0, 1.0)
    w_new, w_old = ex.arm_blend_weights(2, 4, 2.0)
    assert w_old == pytest.approx(math.exp(-1.0))
    assert w_new == pytest.approx(1.0 - math.exp(-1.0))


def test_hand_blend_weights_are_linear_and_reach_the_new_chunk() -> None:
    assert ex.hand_blend_weights(0, 4) == (0.25, 0.75)
    assert ex.hand_blend_weights(3, 4) == (1.0, 0.0)


def test_ema_only_filters_with_history_and_alpha_below_one() -> None:
    raw = np.array([2.0, 4.0], dtype=np.float32)
    prev = np.array([0.0, 0.0])
    assert ex.ema(raw, None, 0.5).tolist() == [2.0, 4.0]
    assert ex.ema(raw, prev, 1.0).tolist() == [2.0, 4.0]
    assert ex.ema(raw, prev, 0.5).tolist() == [1.0, 2.0]
    assert ex.ema(raw, prev, 0.5).dtype == np.float64


# ── ChunkExecutor ─────────────────────────────────────────────────────────────


def test_executor_first_chunk_has_no_blend_and_walks_from_start_step() -> None:
    exe = ex.ChunkExecutor(ex.SmoothingConfig(boundary_blend_steps=4))
    chunk = _chunk(32)
    assert exe.begin_chunk(chunk, start_step=3, horizon=8) == 0
    targets = exe.step(0)
    for arr in (targets.left_arm, targets.right_arm, targets.left_hand, targets.right_hand):
        assert arr is not None
        assert arr.dtype == np.float64
        assert np.all(arr == 3.0)
    assert exe.step(2).left_hand[0] == 5.0
    assert exe.chunk is chunk


def test_executor_blends_the_second_chunk_against_the_previous_unused_future() -> None:
    exe = ex.ChunkExecutor(ex.SmoothingConfig(boundary_blend_steps=4, boundary_blend_alpha=2.0))
    first, second = _chunk(32, base=0.0), _chunk(32, base=100.0)
    exe.begin_chunk(first, 0, 8)
    for offset in range(8):
        exe.step(offset)
    exe.end_chunk()
    assert exe.prev_chunk_next_t == 8

    # old chunk has 24 unused steps, new horizon 8, requested 4 -> window 4
    assert exe.begin_chunk(second, start_step=2, horizon=8) == 4
    assert exe.blend_window == 4
    for offset in range(4):
        targets = exe.step(offset)
        new_row, old_row = 100.0 + 2 + offset, 8.0 + offset
        w_old = math.exp(-2.0 * offset / 4)
        arm = (1.0 - w_old) * new_row + w_old * old_row
        w_new_hand = (offset + 1) / 4.0
        hand = w_new_hand * new_row + (1.0 - w_new_hand) * old_row
        assert targets.left_arm[0] == pytest.approx(arm)
        assert targets.right_arm[6] == pytest.approx(arm)
        assert targets.left_hand[0] == pytest.approx(hand)
        assert targets.right_hand[21] == pytest.approx(hand)
    # offset 0 is exactly the old chunk's row 8; offset 3 of the hands is exactly the new chunk's row 5
    exe.begin_chunk(second, start_step=2, horizon=8)
    assert exe.step(0).left_arm[0] == 8.0
    for offset in range(1, 3):
        exe.step(offset)
    assert exe.step(3).left_hand[0] == 105.0
    # past the window: the new chunk alone
    assert exe.step(4).left_arm[0] == 106.0
    assert exe.step(4).left_hand[0] == 106.0


def test_executor_blend_window_is_bounded_by_the_old_chunk_and_the_horizon() -> None:
    exe = ex.ChunkExecutor(ex.SmoothingConfig(boundary_blend_steps=10))
    exe.begin_chunk(_chunk(32), 0, 30)
    exe.end_chunk()  # 2 unused steps left in the old chunk
    assert exe.begin_chunk(_chunk(32), 0, 16) == 2
    exe.end_chunk()
    assert exe.begin_chunk(_chunk(32), 0, 3) == 3
    # no blend requested -> never a window
    quiet = ex.ChunkExecutor(ex.SmoothingConfig())
    quiet.begin_chunk(_chunk(32), 0, 8)
    quiet.end_chunk()
    assert quiet.begin_chunk(_chunk(32), 0, 8) == 0


def test_executor_boundary_ema_for_n_steps_then_normal_alpha() -> None:
    exe = ex.ChunkExecutor(ex.SmoothingConfig(smooth_alpha=1.0, boundary_smooth_alpha=0.5, boundary_smooth_steps=2))
    chunk = {k: v * 10.0 for k, v in _chunk(8).items()}  # rows 0, 10, 20, ...
    exe.begin_chunk(chunk, 0, 8)
    assert exe.step(0).left_arm[0] == 0.0  # no history yet -> raw
    assert exe.step(1).left_arm[0] == 5.0  # boundary alpha 0.5 against 0
    assert exe.step(2).left_arm[0] == 20.0  # window elapsed -> alpha 1.0
    assert exe.step(2).left_hand[0] == 20.0


def test_executor_ema_history_survives_the_chunk_boundary() -> None:
    exe = ex.ChunkExecutor(ex.SmoothingConfig(smooth_alpha=0.5))
    exe.begin_chunk({k: np.full_like(v, 3.0) for k, v in _chunk(4).items()}, 0, 4)
    for offset in range(4):
        assert exe.step(offset).left_arm[0] == 3.0
    exe.end_chunk()
    exe.begin_chunk({k: np.full_like(v, 10.0) for k, v in _chunk(4).items()}, 0, 4)
    assert exe.step(0).left_arm[0] == pytest.approx(0.5 * 10.0 + 0.5 * 3.0)
    exe.reset()
    exe.begin_chunk({k: np.full_like(v, 10.0) for k, v in _chunk(4).items()}, 0, 4)
    assert exe.step(0).left_arm[0] == 10.0


def test_executor_holds_the_right_arm_and_skips_hands_the_chunk_lacks() -> None:
    hold = np.full(7, 0.3)
    exe = ex.ChunkExecutor(ex.SmoothingConfig(smooth_alpha=0.5, boundary_blend_steps=4), right_arm_hold=hold)
    left_only = _chunk(32, keys=("left_arm", "left_hand"))
    exe.begin_chunk(left_only, 0, 8)
    targets = exe.step(0)
    assert targets.right_arm is hold
    assert targets.right_hand is None
    assert targets.left_hand is not None and targets.left_arm is not None
    # the hold is never EMA'd or blended, even with a previous bimanual chunk
    exe.end_chunk()
    exe.begin_chunk(_chunk(32, base=50.0, keys=("left_arm", "left_hand")), 0, 8)
    assert exe.step(0).right_arm is hold
    exe.right_arm_hold = None
    assert exe.step(1).right_arm is None


def test_executor_end_chunk_keeps_the_refined_chunk_and_its_continuation() -> None:
    exe = ex.ChunkExecutor(ex.SmoothingConfig(boundary_blend_steps=4))
    chunk = _chunk(32)
    exe.begin_chunk(chunk, 2, 8)
    exe.chunk["left_arm"][0, 12, :] = 777.0  # a refine landing while the chunk is walked
    exe.end_chunk()
    assert exe.prev_chunk is chunk
    assert exe.prev_chunk["left_arm"][0, 12, 0] == 777.0
    assert exe.prev_chunk_next_t == 10


def test_executor_step_before_begin_chunk_is_an_error() -> None:
    with pytest.raises(RuntimeError):
        ex.ChunkExecutor(ex.SmoothingConfig()).step(0)


# ── tail edits and their schedules ────────────────────────────────────────────


def test_same_plan_refine_overwrites_only_the_tail_after_t() -> None:
    chunk = {k: np.zeros_like(v) for k, v in _chunk(32).items()}
    fresh = {k: np.ones_like(v) for k, v in _chunk(32).items()}
    assert ex.apply_same_plan_refine(chunk, fresh, t=5, blend_alpha=None) == 26
    for key in ex.ACTION_KEYS:
        assert np.all(chunk[key][0, :6] == 0.0)
        assert np.all(chunk[key][0, 6:] == 1.0)


def test_same_plan_refine_blends_and_skips_keys_the_refine_lacks() -> None:
    chunk = {k: np.zeros_like(v) for k, v in _chunk(32).items()}
    fresh = {k: np.ones_like(v) for k, v in _chunk(32, keys=("left_arm", "left_hand")).items()}
    assert ex.apply_same_plan_refine(chunk, fresh, t=30, blend_alpha=0.5) == 1
    assert chunk["left_arm"][0, 31, 0] == 0.5
    assert chunk["left_arm"][0, 30, 0] == 0.0
    assert np.all(chunk["right_arm"] == 0.0)
    assert ex.apply_same_plan_refine(chunk, fresh, t=31, blend_alpha=None) == 0
    # a second blend decays the earlier content (temporal aggregation)
    ex.apply_same_plan_refine(chunk, fresh, t=30, blend_alpha=0.5)
    assert chunk["left_arm"][0, 31, 0] == 0.75


def test_new_plan_refresh_splices_time_aligned_offsets() -> None:
    chunk = {k: np.zeros_like(v) for k, v in _chunk(32).items()}
    fresh = _chunk(32)  # row t holds t
    assert ex.apply_new_plan_refresh(chunk, fresh, dst_from=6, src_from=3) == 26
    assert np.all(chunk["left_arm"][0, :6] == 0.0)
    assert chunk["left_arm"][0, 6, 0] == 3.0
    assert chunk["right_hand"][0, 31, 0] == 28.0
    assert ex.apply_new_plan_refresh(chunk, fresh, dst_from=6, src_from=32) == 0


def test_refine_due_truth_table_on_the_old_defaults() -> None:
    every, cap, horizon = 4, 4, 16
    assert not ex.refine_due(0, 0, horizon, every, cap, True)  # never on the first tick
    assert ex.refine_due(4, 4, horizon, every, cap, True)
    assert ex.refine_due(2, 4, horizon, every, cap, True)  # cadence is on t, not offset
    assert not ex.refine_due(3, 5, horizon, every, cap, True)
    assert not ex.refine_due(8, 8, horizon, every, cap, True)  # past the delay-aug cap
    assert not ex.refine_due(4, 4, horizon, every, cap, False)  # refine in flight
    assert not ex.refine_due(1, 4, 2, every, cap, True)  # nothing left to refine
    assert ex.refine_due(1, 1, horizon, 0, cap, True)  # every <= 0 behaves as 1


def test_slow_due_fires_at_a_fixed_lead_before_the_boundary() -> None:
    horizon, lead = 16, 3
    assert [o for o in range(horizon) if ex.slow_due(o, horizon, lead, False)] == [13, 14]
    assert not ex.slow_due(13, horizon, lead, True)
    assert [o for o in range(8) if ex.slow_due(o, 8, 0, False)] == []


def test_refresh_due_is_periodic_inside_the_chunk() -> None:
    assert [o for o in range(17) if ex.refresh_due(o, 16, 4)] == [4, 8, 12]
    assert [o for o in range(8) if ex.refresh_due(o, 8, 0)] == [1, 2, 3, 4, 5, 6]


# ── pacing and console ────────────────────────────────────────────────────────


def test_pace_sleeps_the_remainder_of_the_tick() -> None:
    slept: list[float] = []
    ex.pace(time.perf_counter() - 0.010, 0.050, sleep=slept.append)
    assert 0.0 < slept[0] <= 0.040  # about 0.040 minus the scheduling jitter between the two clock reads
    ex.pace(time.perf_counter() - 1.0, 0.050, sleep=slept.append)
    assert slept[1] == 0.0


def test_countdown_with_injected_sleep_and_out() -> None:
    slept: list[float] = []
    lines: list[str] = []
    ex.countdown(3, "holding home", sleep=slept.append, out=lines.append)
    assert slept == [1.0, 1.0, 1.0]
    assert lines[:3] == ["  holding home: 3s ", "  holding home: 2s ", "  holding home: 1s "]
    assert lines[3].startswith("  holding home: done.")
    lines.clear()
    ex.countdown(0, "x", sleep=slept.append, out=lines.append)
    assert len(lines) == 1 and lines[0].startswith("  x: done.")
    assert len(slept) == 3


# ── AsyncPolicyWorker ─────────────────────────────────────────────────────────


def test_async_worker_starts_paused_produces_on_resume_and_stops_on_pause() -> None:
    policy = _FakePolicy(delay_s=0.002)
    worker = ex.AsyncPolicyWorker(policy, lambda: {"obs": 1}, log_every=0)
    try:
        worker.start()
        worker.start()  # idempotent
        time.sleep(0.03)
        assert policy.n_calls == 0 and worker.get_latest() is None
        t_before = time.monotonic()
        worker.resume()
        assert _wait_until(lambda: (worker.get_latest() or ex.CachedChunk({}, 0.0, 0)).version >= 3)
        latest = worker.get_latest()
        assert latest is not None and latest.chunk is policy.chunk
        assert t_before <= latest.t_obs <= time.monotonic()
        assert policy.calls[0][1] == "async-policy-worker"
        worker.pause()
        time.sleep(0.02)  # let the in-flight call finish
        n = policy.n_calls
        time.sleep(0.03)
        assert policy.n_calls == n
        # resume drops the stale chunk before anything new is produced
        policy.delay_s = 0.05
        worker.resume()
        assert worker.get_latest() is None
    finally:
        worker.stop()
    assert "async-policy-worker" not in [t.name for t in threading.enumerate()]


def test_async_worker_request_reset_calls_reset_once_paused_or_running() -> None:
    policy = _FakePolicy(delay_s=0.001)
    worker = ex.AsyncPolicyWorker(policy, lambda: {}, log_every=0)
    try:
        worker.start()
        worker.request_reset()  # while paused
        assert policy.resets == 1
        worker.resume()
        assert _wait_until(lambda: policy.n_calls >= 1)
        worker.request_reset()  # while producing
        assert policy.resets == 2
    finally:
        worker.stop()


def test_async_worker_survives_a_failing_iteration(caplog: pytest.LogCaptureFixture) -> None:
    policy = _FakePolicy(fail_first=True, delay_s=0.001)
    worker = ex.AsyncPolicyWorker(policy, lambda: {}, log_every=0)
    try:
        worker.start()
        with caplog.at_level(logging.WARNING, logger="i4h_tasks.gr00t_n17_tactile.client"):
            worker.resume()
            assert _wait_until(lambda: worker.get_latest() is not None)
    finally:
        worker.stop()
    assert "async inference iteration failed: boom" in caplog.text
    assert worker.get_latest().version >= 1 and policy.n_calls >= 1


# ── TRexPlanWorker ────────────────────────────────────────────────────────────


def test_trex_worker_runs_fifo_with_the_cascade_options_and_captures_on_its_thread() -> None:
    policy = _FakePolicy()
    worker = ex.TRexPlanWorker(policy)
    obs_threads: list[str] = []

    def obs_fn() -> dict:
        obs_threads.append(threading.current_thread().name)
        return {"obs": len(obs_threads)}

    try:
        worker.start()
        t_submit = time.monotonic()
        slots = [worker.submit(kind, obs_fn) for kind in ("slow", "refine", "refine", "slow")]
        for slot in slots:
            assert slot.event.wait(2.0)
    finally:
        worker.stop()
    assert [opts["trex_force_slow"] for opts, _ in policy.calls] == [True, False, False, True]
    assert all(opts["trex_manual"] is True for opts, _ in policy.calls)
    assert all(name == "trex-plan-worker" for _, name in policy.calls)
    assert obs_threads == ["trex-plan-worker"] * 4
    assert [s.kind for s in slots] == ["slow", "refine", "refine", "slow"]
    for slot in slots:
        assert slot.ok and slot.chunk is policy.chunk
        assert slot.t_obs is not None and t_submit <= slot.t_obs <= time.monotonic()
    assert slots[0].t_obs <= slots[1].t_obs <= slots[2].t_obs <= slots[3].t_obs


def test_trex_worker_pause_drops_queued_slots() -> None:
    policy = _FakePolicy()
    worker = ex.TRexPlanWorker(policy)  # not started: everything stays queued
    first, second = worker.submit("slow", lambda: {}), worker.submit("refine", lambda: {})
    worker.pause()
    worker.resume()
    for slot in (first, second):
        assert slot.event.is_set() and slot.chunk is None and not slot.ok
    worker.start()
    worker.stop()
    assert policy.n_calls == 0


def test_trex_worker_request_reset_waits_for_the_policy_reset() -> None:
    policy = _FakePolicy()
    worker = ex.TRexPlanWorker(policy)
    try:
        worker.start()
        worker.request_reset()
        assert policy.resets == 1
        slot = worker.submit("reset", None)
        assert slot.event.wait(2.0) and slot.ok and slot.chunk is None
        assert policy.resets == 2
    finally:
        worker.stop()
    assert "trex-plan-worker" not in [t.name for t in threading.enumerate()]


def test_trex_worker_failure_finishes_the_slot_without_a_chunk() -> None:
    policy = _FakePolicy(fail_first=True)
    worker = ex.TRexPlanWorker(policy)
    try:
        worker.start()
        failed = worker.submit("slow", lambda: {})
        fine = worker.submit("slow", lambda: {})
        assert failed.event.wait(2.0) and fine.event.wait(2.0)
    finally:
        worker.stop()
    assert failed.chunk is None and not failed.ok
    assert fine.chunk is policy.chunk and fine.ok
