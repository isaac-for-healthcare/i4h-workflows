# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turning action chunks into per-tick joint targets, and producing the chunks in time.

A GR00T chunk is ``{"left_arm": (1, T, 7), "right_arm": (1, T, 7), "left_hand":
(1, T, 22), "right_hand": (1, T, 22)}`` of absolute joint targets. Walking it is
the easy part; the rest of this module exists because inference takes longer
than one control tick. Chunks are produced on a worker thread while the main
thread executes the previous one (real-time chunking), the first steps of a
chunk are skipped by the time that passed since its observation, consecutive
chunks are cross-faded at the boundary, and with the server's cascade a chunk's
unexecuted tail is refined in place with fresh tactile (same-plan execution).

Nothing here touches ZMQ or the robot. The policy comes in as an object with
``get_action`` and ``reset`` (``protocol.GR00TClient`` or a fake) and the
observation as a zero-argument callable, so the whole module runs under test
with fakes. The REQ socket inside the real client has exactly one owner thread
at a time: once a worker is started it owns the policy, and the main thread
only talks to the worker.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger("i4h_tasks.gr00t_n17_tactile.client")

ACTION_KEYS = ("left_arm", "right_arm", "left_hand", "right_hand")
ARM_JOINTS = 7
ARM_DOF = 14
HAND_DOF = 22
ARM_LEFT_SLICE = slice(0, 7)
ARM_RIGHT_SLICE = slice(7, 14)
LEFT_HAND_SLICE = slice(14, 36)
RIGHT_HAND_SLICE = slice(36, 58)
#: key -> ``(1, T, D)``; ``T`` is the server's action horizon, ``D`` 7 or 22.
ActionChunk = dict[str, np.ndarray]


class Policy(Protocol):
    """What the executors need from a policy server; ``protocol.GR00TClient`` satisfies it."""

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]: ...

    def reset(self, options: dict | None = None) -> Any: ...


# ── chunk arithmetic ──────────────────────────────────────────────────────────


def chunk_length(chunk: ActionChunk, default: int) -> int:
    """Steps in a chunk: ``left_arm`` decides, then the first other action key, else ``default``."""
    if "left_arm" in chunk:
        return int(chunk["left_arm"].shape[1])
    for key in ACTION_KEYS:
        if key in chunk:
            return int(chunk[key].shape[1])
    return int(default)


def copy_chunk(chunk: ActionChunk) -> ActionChunk:
    """A private mutable copy, so in-place refines never alias a worker's cached chunk."""
    return {k: (np.array(v) if hasattr(v, "shape") else v) for k, v in chunk.items()}


def start_step_for(age_s: float, dt: float, chunk_len: int) -> int:
    """The chunk index whose intended execution time is now, given the chunk's observation age."""
    return max(0, min(int(max(0.0, age_s) / dt), chunk_len - 1))


def plan_horizon(execution_steps: int, chunk_len: int, start_step: int, steps_left: int) -> int:
    """Ticks to walk this chunk for; ``<= 0`` means the chunk is already stale and gets skipped."""
    return min(execution_steps, chunk_len - start_step, steps_left)


# ── smoothing and boundary blending ───────────────────────────────────────────


@dataclass(frozen=True)
class SmoothingConfig:
    """Per-step EMA and cross-chunk blend settings (the ``--smooth-*`` / ``--boundary-*`` flags)."""

    smooth_alpha: float = 1.0
    boundary_smooth_alpha: float = 1.0
    boundary_smooth_steps: int = 0
    boundary_blend_steps: int = 0
    boundary_blend_alpha: float = 2.0


def arm_blend_weights(offset: int, window: int, alpha: float) -> tuple[float, float]:
    """``(w_new, w_old)``: the old chunk decays as ``exp(-alpha * offset / window)``."""
    w_old = math.exp(-alpha * offset / window)
    return 1.0 - w_old, w_old


def hand_blend_weights(offset: int, window: int) -> tuple[float, float]:
    """``(w_new, w_old)`` for the hands: a linear ramp, deliberately unlike the arms' exponential."""
    w_new = (offset + 1) / float(window)
    return w_new, 1.0 - w_new


def ema(raw: np.ndarray, prev: np.ndarray | None, alpha: float) -> np.ndarray:
    """``alpha * raw + (1 - alpha) * prev`` once there is a previous target and ``alpha < 1``."""
    raw = np.asarray(raw, dtype=np.float64)
    if alpha < 1.0 and prev is not None:
        return alpha * raw + (1.0 - alpha) * prev
    return raw


@dataclass
class StepTargets:
    """Joint targets for one control tick, radians, float64."""

    left_arm: np.ndarray | None
    right_arm: np.ndarray | None
    left_hand: np.ndarray | None
    right_hand: np.ndarray | None


class ChunkExecutor:
    """Per-episode smoothing and cross-chunk blend state plus the per-tick target math.

    Owner: the main control thread. ``begin_chunk`` adopts a chunk, ``step`` yields
    the targets for one offset into it, ``end_chunk`` stashes it for the blend at
    the next boundary. ``chunk`` is public and mutable on purpose: same-plan
    refines and new-plan tail refreshes write into it between steps.
    """

    def __init__(self, smoothing: SmoothingConfig, right_arm_hold: np.ndarray | None = None) -> None:
        self.smoothing = smoothing
        self.right_arm_hold = right_arm_hold
        self.boundary_smooth_steps = max(0, int(smoothing.boundary_smooth_steps))
        self.boundary_blend_steps = max(0, int(smoothing.boundary_blend_steps))
        # alpha > 0 -> exp(-alpha * t / N) decays; clamp to a tiny positive.
        self.blend_alpha = max(1e-6, float(smoothing.boundary_blend_alpha))
        self.chunk: ActionChunk | None = None
        self.start_step = 0
        self.horizon = 0
        self.blend_window = 0
        self.prev_targets: dict[str, np.ndarray | None] = {}
        self.prev_chunk: ActionChunk | None = None
        self.prev_chunk_next_t = 0
        self.boundary_remaining = 0
        self.reset()

    def reset(self) -> None:
        """Forget the previous episode: no EMA history, no chunk to blend against."""
        self.prev_targets = {k: None for k in ACTION_KEYS}
        self.prev_chunk = None
        self.prev_chunk_next_t = 0
        self.boundary_remaining = 0
        self.chunk = None
        self.start_step = 0
        self.horizon = 0
        self.blend_window = 0

    def begin_chunk(self, chunk: ActionChunk, start_step: int, horizon: int) -> int:
        """Adopt a chunk; returns the blend window against the previous chunk's unused future."""
        self.chunk = chunk
        self.start_step = int(start_step)
        self.horizon = int(horizon)
        self.boundary_remaining = self.boundary_smooth_steps
        if self.boundary_blend_steps > 0 and self.prev_chunk is not None and "left_arm" in self.prev_chunk:
            prev_len = int(self.prev_chunk["left_arm"].shape[1])
            old_remaining = max(0, prev_len - self.prev_chunk_next_t)
            self.blend_window = min(self.boundary_blend_steps, old_remaining, self.horizon)
        else:
            self.blend_window = 0
        return self.blend_window

    def _row(self, chunk: ActionChunk, key: str, t: int) -> np.ndarray:
        return np.asarray(chunk[key][0][t], dtype=np.float64)

    def _blended(self, key: str, offset: int, t: int, weights: tuple[float, float]) -> np.ndarray:
        """The raw target for ``key`` at this offset: the chunk row, blended with the old chunk inside the window."""
        assert self.chunk is not None
        if offset < self.blend_window and self.prev_chunk is not None and key in self.prev_chunk:
            w_new, w_old = weights
            old_t = self.prev_chunk_next_t + offset
            return w_new * self._row(self.chunk, key, t) + w_old * self._row(self.prev_chunk, key, old_t)
        return self._row(self.chunk, key, t)

    def step(self, offset: int) -> StepTargets:
        """Targets for ``start_step + offset``: blend, then EMA, with the right arm held when the chunk lacks it."""
        if self.chunk is None:
            raise RuntimeError("ChunkExecutor.step() before begin_chunk()")
        t = self.start_step + offset
        # Heavier EMA right after a switch hides the seam between two independently sampled chunks.
        if self.boundary_remaining > 0:
            alpha = self.smoothing.boundary_smooth_alpha
            self.boundary_remaining -= 1
        else:
            alpha = self.smoothing.smooth_alpha
        arm_w = arm_blend_weights(offset, self.blend_window, self.blend_alpha) if self.blend_window > 0 else (1.0, 0.0)
        hand_w = hand_blend_weights(offset, self.blend_window) if self.blend_window > 0 else (1.0, 0.0)

        targets: dict[str, np.ndarray | None] = {}
        for key in ("left_arm", "right_arm"):
            if key == "right_arm" and key not in self.chunk:
                # Fixed hold: no blend or EMA so it never drifts off the pose.
                targets[key] = self.right_arm_hold
                self.prev_targets[key] = self.right_arm_hold
                continue
            target = ema(self._blended(key, offset, t, arm_w), self.prev_targets[key], alpha)
            self.prev_targets[key] = target
            targets[key] = target
        for key in ("left_hand", "right_hand"):
            if key not in self.chunk:
                targets[key] = None
                continue
            target = ema(self._blended(key, offset, t, hand_w), self.prev_targets[key], alpha)
            self.prev_targets[key] = target
            targets[key] = target
        return StepTargets(
            left_arm=targets["left_arm"],
            right_arm=targets["right_arm"],
            left_hand=targets["left_hand"],
            right_hand=targets["right_hand"],
        )

    def end_chunk(self) -> None:
        """Stash the chunk as walked (refines included) and where it would have continued."""
        self.prev_chunk = self.chunk
        self.prev_chunk_next_t = self.start_step + self.horizon


# ── in-place tail edits and their schedules ───────────────────────────────────


def apply_same_plan_refine(chunk: ActionChunk, fresh: ActionChunk, t: int, blend_alpha: float | None) -> int:
    """Overwrite (or EMA, with ``blend_alpha``) the tail after ``t`` index-for-index from a refine of the same plan.

    Returns the longest tail written, 0 when nothing fit. In place on ``chunk``."""
    applied = 0
    for key in ACTION_KEYS:
        if key not in chunk or key not in fresh:
            continue
        dst, src = chunk[key][0], fresh[key][0]
        n = min(dst.shape[0] - (t + 1), src.shape[0] - (t + 1))
        if n <= 0:
            continue
        if blend_alpha is None:
            dst[t + 1 : t + 1 + n] = src[t + 1 : t + 1 + n]
        else:
            b = float(blend_alpha)
            dst[t + 1 : t + 1 + n] = b * src[t + 1 : t + 1 + n] + (1.0 - b) * dst[t + 1 : t + 1 + n]
        applied = max(applied, n)
    return applied


def apply_new_plan_refresh(chunk: ActionChunk, fresh: ActionChunk, dst_from: int, src_from: int) -> int:
    """Splice a time-aligned tail of an independent fresh chunk into ``chunk`` from ``dst_from`` on.

    Returns the longest tail written, 0 when nothing fit. In place on ``chunk``."""
    applied = 0
    for key in ACTION_KEYS:
        if key not in chunk or key not in fresh:
            continue
        dst, src = chunk[key][0], fresh[key][0]
        n = min(dst.shape[0] - dst_from, src.shape[0] - src_from)
        if n <= 0:
            continue
        dst[dst_from : dst_from + n] = src[src_from : src_from + n]
        applied = max(applied, n)
    return applied


def refine_due(offset: int, t: int, horizon: int, every: int, max_offset: int, slot_free: bool) -> bool:
    """Same-plan refine cadence on the absolute plan index ``t``, capped inside the delay-aug coverage."""
    return offset > 0 and t % max(1, every) == 0 and t <= max_offset and offset + 1 < horizon and slot_free


def slow_due(offset: int, horizon: int, lead: int, pending: bool) -> bool:
    """Predict the next plan a fixed ``lead`` ticks before the chunk boundary, once per chunk."""
    return not pending and offset + 1 < horizon and offset >= horizon - max(1, lead)


def refresh_due(offset: int, horizon: int, every: int) -> bool:
    """New-plan tail refresh cadence on the intra-chunk offset."""
    return offset > 0 and offset % max(1, every) == 0 and offset + 1 < horizon


# ── pacing and console helpers ────────────────────────────────────────────────


def pace(t0: float, dt: float, sleep: Callable[[float], None] = time.sleep) -> None:
    """Sleep out the rest of a ``dt`` tick that started at ``t0`` (``time.perf_counter()``)."""
    sleep(max(0.0, dt - (time.perf_counter() - t0)))


def _emit(out: Callable[[str], None], text: str, end: str) -> None:
    # The default console output rewrites one line in place; an injected sink gets the bare text.
    if out is print:
        print(text, end=end, flush=True)
    else:
        out(text)


def countdown(
    seconds: int,
    label: str,
    sleep: Callable[[float], None] = time.sleep,
    out: Callable[[str], None] = print,
) -> None:
    """Count ``seconds`` down on one console line, one ``sleep(1.0)`` per second, then report done."""
    for i in range(int(seconds), 0, -1):
        _emit(out, f"  {label}: {i}s ", end="\r")
        sleep(1.0)
    _emit(out, f"  {label}: done." + " " * 10, end="\n")


# ── real-time chunking: the independent-chunk producer ────────────────────────


@dataclass
class CachedChunk:
    """The newest chunk a worker produced, with the moment its observation was taken."""

    chunk: ActionChunk
    t_obs: float
    version: int


class AsyncPolicyWorker:
    """Background producer of independent chunks at whatever rate the server sustains.

    Once started it is the sole owner of the policy object, so episode resets go
    through ``request_reset``. Starts paused; ``resume`` drops any stale chunk and
    begins producing. ``get_latest`` is the one method other threads read.
    """

    def __init__(self, policy: Policy, obs_fn: Callable[[], dict], log_every: int = 30) -> None:
        self._policy = policy
        self._obs_fn = obs_fn
        self._log_every = int(log_every)
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._reset_request = threading.Event()
        self._cache_lock = threading.Lock()
        self._latest: CachedChunk | None = None
        self._version = 0
        self._thread: threading.Thread | None = None
        self._infer_count = 0
        # End-to-end (obs capture -> chunk ready) accumulators, reset every log window. This is the
        # chunk-production rate the executor actually sees, not the server-only compute rate.
        self._e2e_n = 0
        self._e2e_obs_sum = 0.0
        self._e2e_inf_sum = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="async-policy-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pause.clear()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def pause(self) -> None:
        """Stop producing (go-home, reset)."""
        self._pause.set()

    def resume(self) -> None:
        """Produce again, after dropping any chunk from before the pause."""
        with self._cache_lock:
            self._latest = None
        self._pause.clear()

    def request_reset(self) -> None:
        """Have the worker call ``policy.reset()``; returns once it has, or after 5 s with a warning."""
        self._reset_request.set()
        deadline = time.monotonic() + 5.0
        while self._reset_request.is_set():
            if time.monotonic() > deadline or self._stop.is_set():
                logger.warning("async worker: reset request timed out")
                return
            time.sleep(0.005)

    def get_latest(self) -> CachedChunk | None:
        """Snapshot of the newest cached chunk, or ``None`` when nothing is cached."""
        with self._cache_lock:
            return self._latest

    def _handle_reset(self, where: str) -> None:
        try:
            self._policy.reset()
        except Exception as exc:
            logger.warning(f"async worker reset{where} failed: {exc}")
        self._reset_request.clear()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                # Honour resets while paused so they are not stuck behind the next resume().
                if self._reset_request.is_set():
                    self._handle_reset(" (paused)")
                time.sleep(0.01)
                continue
            if self._reset_request.is_set():
                self._handle_reset("")
                continue
            t_obs = time.monotonic()
            try:
                obs = self._obs_fn()
                t_obs_done = time.monotonic()
                chunk, _ = self._policy.get_action(obs)
                t_inf_done = time.monotonic()
            except Exception as exc:
                logger.warning(f"async inference iteration failed: {exc}")
                time.sleep(0.05)
                continue
            with self._cache_lock:
                self._version += 1
                self._latest = CachedChunk(chunk, t_obs, self._version)
            self._infer_count += 1
            self._e2e_n += 1
            self._e2e_obs_sum += t_obs_done - t_obs
            self._e2e_inf_sum += t_inf_done - t_obs_done
            if self._log_every > 0 and self._infer_count % self._log_every == 0:
                n = max(self._e2e_n, 1)
                obs_ms = 1e3 * self._e2e_obs_sum / n
                inf_ms = 1e3 * self._e2e_inf_sum / n
                tot_ms = obs_ms + inf_ms
                e2e_hz = (1e3 / tot_ms) if tot_ms > 0 else 0.0
                logger.info(
                    f"async worker: {self._infer_count} chunks cached | "
                    f"END-TO-END {tot_ms:.1f} ms (~{e2e_hz:.1f} Hz) = "
                    f"obs {obs_ms:.1f} ms + infer(zmq+server) {inf_ms:.1f} ms "
                    f"| latest version={self._version}"
                )
                self._e2e_n = 0
                self._e2e_obs_sum = 0.0
                self._e2e_inf_sum = 0.0


# ── same-plan execution against the server cascade ────────────────────────────


@dataclass
class PlanSlot:
    """Result slot of one ``TRexPlanWorker`` command; ``event`` is set when it is final."""

    kind: str
    event: threading.Event = field(default_factory=threading.Event)
    chunk: ActionChunk | None = None
    t_obs: float | None = None
    ok: bool = False


class TRexPlanWorker:
    """FIFO single-owner worker for same-plan execution against a ``--cascade`` server.

    The client drives the schedule so the server's single cascade cache always
    holds the plan being refined: ``"slow"`` asks for a new plan, ``"refine"``
    re-denoises the tail of the same cached plan with this observation's fresh
    tactile, ``"reset"`` resets the policy. FIFO order guarantees a chunk's
    refines reach the server before the next slow overwrites the cache.

    ``obs_fn`` runs on the worker thread when the command is dequeued, so camera
    and tactile capture never jitter the control loop and the observation is
    taken right before inference. ``slot.t_obs`` is stamped at that actual
    capture moment: the queue can delay a dequeue well past submit time, and
    aging a chunk off the submit time overshoots ``start_step``.
    """

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._cv = threading.Condition()
        self._queue: list[tuple[str, Callable[[], dict] | None, PlanSlot]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="trex-plan-worker", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def submit(self, kind: str, obs_fn: Callable[[], dict] | None) -> PlanSlot:
        """Enqueue a command; the returned slot's ``event`` is set when it has run (or been dropped)."""
        slot = PlanSlot(kind=kind)
        with self._cv:
            self._queue.append((kind, obs_fn, slot))
            self._cv.notify()
        return slot

    def pause(self) -> None:
        """Drop queued-but-unstarted work so a reset starts clean; their slots finish with ``chunk None``."""
        with self._cv:
            for _, _, slot in self._queue:
                slot.chunk = None
                slot.ok = False
                slot.event.set()
            self._queue.clear()

    def resume(self) -> None:
        """Nothing to do: the worker produces only on demand."""

    def request_reset(self) -> None:
        """Queue a ``policy.reset()`` and wait up to 5 s for it."""
        slot = self.submit("reset", None)
        slot.event.wait(5.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._queue and not self._stop.is_set():
                    self._cv.wait(timeout=0.1)
                if self._stop.is_set():
                    return
                kind, obs_fn, slot = self._queue.pop(0)
            try:
                if kind == "reset":
                    self._policy.reset()
                    slot.chunk = None
                    slot.ok = True
                else:
                    slot.t_obs = time.monotonic()
                    obs = obs_fn() if callable(obs_fn) else obs_fn
                    chunk, _ = self._policy.get_action(
                        obs, options={"trex_manual": True, "trex_force_slow": kind == "slow"}
                    )
                    slot.chunk = chunk
                    slot.ok = chunk is not None
            except Exception as exc:
                logger.warning(f"trex plan worker {kind} failed: {exc}")
                slot.chunk = None
                slot.ok = False
            slot.event.set()
