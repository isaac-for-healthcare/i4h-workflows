# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ZMQ subscriber for the SharpaWave tactile stream.

Vendored from ``teleop/robot_control/robot_hand_sharpa.py`` of the public
``binliunls/xr_teleoperate`` fork (pull request #12), reduced to the subscriber.
The publisher is a small bridge on the robot's onboard computer that reads the
SharpaWave SDK and republishes every fingertip frame in the layout of
``wire.py``; it is not part of this repository.

Two threads: one drains the socket into a per-channel latest-frame cache, the
other samples that cache at the training rate into a rolling force history, so
the force window handed to the model has the same per-row spacing as the
dataset's regardless of how fast inference runs.
"""

from __future__ import annotations

import collections
import logging
import threading
from typing import Any

import numpy as np

from i4h_tasks.gr00t_n17_tactile.tactile.sensor import wire

logger = logging.getLogger("i4h_tasks.gr00t_n17_tactile.tactile")


class SharpaTactileSubscriber:
    """Latest-frame cache over the bridge's PUB socket at ``tcp://host:port``.

    Args:
        host: bridge host.
        port: bridge tactile PUB port.
        force_window_len: rows of the rolling 6-axis force history per channel.
    """

    #: Rate the rolling force history is sampled at: the dataset's frame rate.
    FORCE_SAMPLE_HZ = 30.0

    def __init__(self, host: str, port: int, force_window_len: int = 16) -> None:
        import zmq

        self._host = host
        self._port = port
        self._zmq = zmq
        self._cache: dict[int, wire.TactileMessage] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._force_window_len = int(force_window_len)
        self._force_hist: dict[int, collections.deque[np.ndarray]] = {
            channel: collections.deque(maxlen=self._force_window_len) for channel in range(wire.NUM_CHANNELS)
        }
        logger.info("tactile subscriber connecting to tcp://%s:%s", host, port)
        self._thread = threading.Thread(target=self._sub_loop, daemon=True)
        self._thread.start()
        self._force_thread = threading.Thread(target=self._force_sampler_loop, daemon=True)
        self._force_thread.start()

    def _force_sampler_loop(self) -> None:
        period = 1.0 / self.FORCE_SAMPLE_HZ
        while not self._stop.is_set():
            with self._lock:
                for channel in range(wire.NUM_CHANNELS):
                    msg = self._cache.get(channel)
                    row = np.zeros(wire.F6_LEN, dtype=np.float32)
                    if msg is not None and msg.f6.size >= wire.F6_LEN:
                        row = np.asarray(msg.f6, dtype=np.float32).reshape(-1)[: wire.F6_LEN]
                    self._force_hist[channel].append(row)
            # Event.wait doubles as an interruptible sleep so close() returns promptly.
            if self._stop.wait(timeout=period):
                break

    def _sub_loop(self) -> None:
        zmq = self._zmq
        socket = zmq.Context.instance().socket(zmq.SUB)
        # One sensor tick is ten messages; keep a few ticks of slack and no more.
        socket.setsockopt(zmq.RCVHWM, 30)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(f"tcp://{self._host}:{self._port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if socket not in dict(poller.poll(timeout=100)):
                    continue
                while True:
                    try:
                        payload = socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        msg = wire.unpack(payload)
                    except ValueError as exc:
                        logger.warning("tactile frame dropped: %s", exc)
                        continue
                    with self._lock:
                        self._cache[msg.channel] = msg
        finally:
            socket.close()

    def snapshot(self) -> dict[str, Any]:
        """Latest frame per fingertip, keyed ``hand -> finger``, plus per-hand joints.

        Fingers that have not reported yet are absent. ``joints[hand]`` is the
        ``(22,)`` degree vector attached to that hand's newest frame, or ``None``
        when the publisher sends no trailer.
        """
        with self._lock:
            cached = list(self._cache.values())
        out: dict[str, Any] = {"left": {}, "right": {}, "joints": {"left": None, "right": None}}
        joint_ts = {"left": -1.0, "right": -1.0}
        for msg in cached:
            hand, finger = msg.hand_finger
            out[hand][finger] = msg
            if msg.joints is not None and msg.ts > joint_ts[hand]:
                joint_ts[hand] = msg.ts
                out["joints"][hand] = msg.joints
        return out

    def force_window(self) -> dict[str, dict[str, np.ndarray]]:
        """Rolling force history, ``hand -> finger -> (W, 6)`` oldest first.

        Always ``W`` rows: before the sampler has filled a deque the window is
        left-padded with its oldest row, or zeros when it is empty.
        """
        width = self._force_window_len
        with self._lock:
            hist = {channel: list(rows) for channel, rows in self._force_hist.items()}
        out: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
        for channel, rows in hist.items():
            hand, finger = wire.channel_to_hand_finger(channel)
            if len(rows) < width:
                pad_row = rows[0] if rows else np.zeros(wire.F6_LEN, dtype=np.float32)
                rows = [pad_row] * (width - len(rows)) + rows
            out[hand][finger] = np.stack(rows[-width:], axis=0).astype(np.float32, copy=False)
        return out

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._force_thread.join(timeout=1.0)
