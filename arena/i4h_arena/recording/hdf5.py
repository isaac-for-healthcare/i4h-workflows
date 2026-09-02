# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HDF5 episode capture, tagged by workflow node.

Because the recorder subscribes to engine
events, every frame knows which node produced it. That is what lets
``tools/mimic`` augment a single skill and ``tools/annotator`` label per skill
rather than per episode — neither is expressible against a flat episode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import h5py
import numpy as np

from i4h_common.episode import Segment, write_segments
from i4h_engine.events import EventKind, WorkflowEvent

logger = logging.getLogger("i4h_arena.recording")

#: Display-independent sensor output stored beside each camera image when a sensor offers it.
SIGNAL_OUTPUT = "attenuation"


class EpisodeRecorder:
    """Stream camera frames to a temporary group, then commit or discard it."""

    def __init__(self, path: str | Path, *, workflow: Any, cameras: tuple[str, ...] = ()) -> None:
        self.path = Path(path)
        self.workflow = workflow
        self.cameras = cameras
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(str(self.path), "a")
        self._data = self._file.require_group("data")
        self._data.attrs.setdefault("workflow", workflow.name)
        self._data.attrs.setdefault("mode", workflow.mode)
        self._data.attrs.setdefault("scene", workflow.scene)

        self._actions: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._attempt_group: h5py.Group | None = None
        self._camera_datasets: dict[str, h5py.Dataset] = {}
        self._frame_queue: Queue[tuple[str, np.ndarray] | None] = Queue(maxsize=32)
        self._writer_error: BaseException | None = None
        self._writer_thread = Thread(target=self._write_frames, name="i4h-hdf5-writer", daemon=True)
        self._writer_thread.start()
        self._segments: list[Segment] = []
        self._open_node: tuple[str, str, int] | None = None
        self._episode = 0
        self._attempt = 1

    # -- lifecycle -------------------------------------------------------
    def begin_episode(self, episode: int, attempt: int) -> None:
        self._drain_frames()
        self._episode = episode
        self._attempt = attempt
        self._actions.clear()
        self._states.clear()
        self._discard_attempt()
        self._attempt_group = self._data.create_group("_attempt")
        self._attempt_group.create_group("obs")
        self._camera_datasets.clear()
        self._segments.clear()
        self._open_node = None

    def restart_episode(self, *, node: str = "", task_id: str = "") -> None:
        """Discard pre-reset samples and continue recording from a clean scene."""
        self.begin_episode(self._episode, self._attempt)
        if node:
            self._open_node = (node, task_id, 0)

    def on_event(self, event: WorkflowEvent) -> None:
        """Turn node transitions into frame ranges."""
        if event.kind == EventKind.NODE_ENTERED:
            self._open_node = (event.node, event.task_id, len(self._actions))
        elif (
            event.kind in (EventKind.NODE_SUCCEEDED, EventKind.NODE_FAILED, EventKind.NODE_ABORTED)
            and self._open_node
            and self._open_node[0] == event.node
        ):
            node, task_id, start = self._open_node
            self._segments.append(Segment(node=node, task_id=task_id, start=start, end=len(self._actions)))
            self._open_node = None

    def on_step(self, action: np.ndarray, view: Any) -> None:
        # Env 0 only. The HDF5 schema is one trajectory per demo, and the engine
        # advances the frontier lock-step across the batch anyway (DESIGN.md §5),
        # so envs 1..N would be near-duplicates rather than extra demos.
        # Recording a vectorized rollout properly needs per-env engine state.
        self._actions.append(np.asarray(action, dtype=np.float32)[0])
        self._states.append(np.asarray(view.joints().pos, dtype=np.float32)[0])
        for camera in self.cameras:
            frame = view.camera(camera)
            if frame is not None:
                self._raise_writer_error()
                self._frame_queue.put((camera, np.asarray(frame.to_array())))
            # The viewable image carries the live display mapping, so an operator changing
            # polarity or window mid-episode would change the recording. Store the renderer's
            # own signal too, which no display control can reach.
            signal = self._sensor_signal(view, camera)
            if signal is not None:
                self._raise_writer_error()
                self._frame_queue.put((f"{camera}_{SIGNAL_OUTPUT}", signal))

    @staticmethod
    def _sensor_signal(view: Any, camera: str) -> np.ndarray | None:
        reader = getattr(view, "sensor_signal", None)
        if not callable(reader):
            return None
        values = reader(camera, SIGNAL_OUTPUT)
        return None if values is None else np.asarray(values)

    def end_episode(self, result: Any, *, keep: bool) -> None:
        self._drain_frames()
        if self._open_node:  # a node still active when the workflow ended
            node, task_id, start = self._open_node
            self._segments.append(Segment(node=node, task_id=task_id, start=start, end=len(self._actions)))
            self._open_node = None

        if not keep or not self._actions:
            logger.info("discarding episode %s attempt %s (%s)", result.index, result.attempt, result.status.value)
            self._discard_attempt()
            return

        name = f"demo_{len(self._existing_demos())}"
        if self._attempt_group is None:
            raise RuntimeError("begin_episode must be called before end_episode")
        demo = self._attempt_group
        demo.create_dataset("actions", data=np.stack(self._actions))
        obs = demo["obs"]
        obs.create_dataset("joint_pos", data=np.stack(self._states))

        demo.attrs["success"] = bool(result.succeeded)
        demo.attrs["num_samples"] = len(self._actions)
        demo.attrs["workflow"] = self.workflow.name
        demo.attrs["mode"] = self.workflow.mode
        demo.attrs["episode_index"] = result.index
        demo.attrs["attempt_index"] = result.attempt
        demo.attrs["status"] = result.status.value
        write_segments(demo, self._segments)

        self._data.move("_attempt", name)
        self._attempt_group = None
        self._camera_datasets.clear()
        self._data.attrs["total"] = len(self._existing_demos())
        self._file.flush()
        logger.info(
            "saved %s: %s frames, %s segments (%s)",
            name,
            len(self._actions),
            len(self._segments),
            result.status.value,
        )

    def _append_frame(self, camera: str, frame: np.ndarray) -> None:
        if self._attempt_group is None:
            raise RuntimeError("begin_episode must be called before on_step")
        dataset = self._camera_datasets.get(camera)
        if dataset is None:
            obs = self._attempt_group["obs"]
            dataset = obs.create_dataset(
                camera,
                data=frame[np.newaxis, ...],
                maxshape=(None, *frame.shape),
                chunks=(1, *frame.shape),
                compression="lzf",
            )
            self._camera_datasets[camera] = dataset
            return
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = frame

    def _write_frames(self) -> None:
        while True:
            item = self._frame_queue.get()
            if item is None:
                self._frame_queue.task_done()
                return
            try:
                if self._writer_error is None:
                    self._append_frame(*item)
            except BaseException as exc:  # propagate writer failures on the simulator thread
                self._writer_error = exc
            finally:
                self._frame_queue.task_done()

    def _drain_frames(self) -> None:
        self._frame_queue.join()
        self._raise_writer_error()

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError("camera recording writer failed") from self._writer_error

    def _discard_attempt(self) -> None:
        if "_attempt" in self._data:
            del self._data["_attempt"]
            self._file.flush()
        self._attempt_group = None
        self._camera_datasets.clear()

    def _existing_demos(self) -> list[str]:
        return [n for n in self._data if n.startswith("demo_")]

    def close(self) -> None:
        error: BaseException | None = None
        try:
            self._drain_frames()
        except BaseException as exc:
            error = exc
        try:
            self._discard_attempt()
        finally:
            self._frame_queue.put(None)
            self._writer_thread.join()
            try:
                self._file.flush()
            finally:
                self._file.close()
        if error is not None:
            raise error
