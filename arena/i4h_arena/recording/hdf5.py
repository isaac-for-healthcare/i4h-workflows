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

from i4h_common.episode import DIAGNOSTICS_GROUP, Segment, write_segments
from i4h_engine.events import EventKind, WorkflowEvent

logger = logging.getLogger("i4h_arena.recording")

#: Filters for camera datasets. ``lzf`` was chosen for speed, but it barely
#: dents a rendered X-ray: one 1024x1024 fluoroscopy frame still cost about
#: 3.7 MB, which is a 2 GB episode and puts a few hundred demonstrations beyond
#: any disk we have. Nothing here is on the simulation thread -- frames go
#: through a queue to a writer thread, and the renderer produces them far
#: slower than gzip consumes them -- so the trade was paying latency nobody was
#: waiting on to save space that ran out.
#:
#: ``shuffle`` is what makes the level worth having: a greyscale projection
#: written as three equal channels interleaves near-identical bytes, and
#: grouping them by position gives the deflate pass long runs to find. Level 4
#: rather than 9 because the last levels buy little on image data and cost
#: several times the CPU.
CAMERA_COMPRESSION = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

#: Longest edge kept in the recording. The fluoroscopy sensor renders 1024 so
#: the operator can see the wire while driving, but nothing downstream consumes
#: that: the GR00T catheter manifest asks for 256, and the converter resizes to
#: it before training. Storing the other fifteen sixteenths of the pixels costs
#: about 1.3 MB a frame -- 164 GB for two hundred episodes against 11 GB at 256
#: -- to be discarded later. Set to ``0`` to record whatever the sensor renders.
RECORDED_CAMERA_EDGE = 256


def downsample_frame(frame: np.ndarray, edge: int = RECORDED_CAMERA_EDGE) -> np.ndarray:
    """Box-average ``frame`` down until its short side is near ``edge``.

    Averaged rather than subsampled because the catheter is about a pixel wide
    at 1024: taking every fourth pixel drops the wire out of the frames it
    happens to fall between, which is the one thing the image is recorded for.
    Averaging dims it instead of losing it.

    Only exact integer factors are used, so the box divides the frame evenly
    and no edge pixel is weighted differently from the rest. A frame that does
    not divide, or is already small, is returned untouched -- this is a storage
    saving, and a shape it cannot halve cleanly is not worth resampling for.
    """
    if edge <= 0 or frame.ndim < 2:
        return frame
    rows, cols = frame.shape[:2]
    factor = min(rows // edge, cols // edge)
    if factor < 2 or rows % factor or cols % factor:
        return frame
    blocks = frame.reshape(rows // factor, factor, cols // factor, factor, *frame.shape[2:])
    reduced = blocks.mean(axis=(1, 3))
    # Rounded before the cast: numpy truncates toward zero on the way back to
    # an integer dtype, which would darken every frame by half a level.
    return (np.rint(reduced) if np.issubdtype(frame.dtype, np.integer) else reduced).astype(frame.dtype)


#: Display-independent sensor output stored beside each camera image when a sensor offers it.
SIGNAL_OUTPUT = "attenuation"


class EpisodeRecorder:
    """Stream camera frames to a temporary group, then commit or discard it."""

    def __init__(
        self,
        path: str | Path,
        *,
        workflow: Any,
        cameras: tuple[str, ...] = (),
        camera_edge: int = RECORDED_CAMERA_EDGE,
    ) -> None:
        self.path = Path(path)
        self.camera_edge = int(camera_edge)
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
        self._diagnostics: list[dict[str, np.ndarray]] = []
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
        self._diagnostics.clear()
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
        self._diagnostics.append(self._frame_diagnostics(view))
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

    @staticmethod
    def _frame_diagnostics(view: Any) -> dict[str, np.ndarray]:
        """Per-frame physics measurements, for scenes that offer any.

        Optional and duck-typed, like ``sensor_signal``: a scene with nothing to
        add omits the method and the recording simply has no ``diagnostics``
        group. Values are whatever the scene names them, so this stays free of
        any one embodiment's vocabulary.
        """
        reader = getattr(view, "diagnostics", None)
        if not callable(reader):
            return {}
        values = reader()
        if not values:
            return {}
        return {str(key): np.asarray(value, dtype=np.float32) for key, value in values.items()}

    def _stacked_diagnostics(self) -> dict[str, np.ndarray]:
        """One array per measurement, aligned frame for frame with ``actions``.

        A scene may start reporting a measurement partway through an episode --
        the live vessel gap only exists once a vessel has been built -- so the
        keys are unioned over the episode and absent frames are filled with
        ``nan``. Dropping the partial keys instead would lose the measurement
        entirely, and dropping the frames would break alignment with the
        actions, which is the one property that makes these worth recording.
        """
        names = {name for frame in self._diagnostics for name in frame}
        if not names:
            return {}
        stacked: dict[str, np.ndarray] = {}
        for name in sorted(names):
            shape = next(frame[name].shape for frame in self._diagnostics if name in frame)
            missing = np.full(shape, np.nan, dtype=np.float32)
            stacked[name] = np.stack([frame.get(name, missing) for frame in self._diagnostics])
        return stacked

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
        diagnostics = self._stacked_diagnostics()
        if diagnostics:
            # Beside ``obs`` rather than inside it: these are measurements of the
            # simulation, not observations a policy is trained against, and
            # everything that walks ``obs`` would otherwise have to learn to skip
            # them.
            group = demo.create_group(DIAGNOSTICS_GROUP)
            for measurement, values in diagnostics.items():
                group.create_dataset(measurement, data=values)

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
            "saved %s: %s frames, %s segments, %s diagnostics (%s)",
            name,
            len(self._actions),
            len(self._segments),
            len(diagnostics),
            result.status.value,
        )

    def _append_frame(self, camera: str, frame: np.ndarray) -> None:
        if self._attempt_group is None:
            raise RuntimeError("begin_episode must be called before on_step")
        # Here rather than at capture: this runs on the writer thread, so the
        # resample is paid alongside compression instead of on the step the
        # simulator is waiting to finish. It also catches the raw sensor
        # signal, which is queued separately but is the same size.
        frame = downsample_frame(frame, self.camera_edge)
        dataset = self._camera_datasets.get(camera)
        if dataset is None:
            obs = self._attempt_group["obs"]
            dataset = obs.create_dataset(
                camera,
                data=frame[np.newaxis, ...],
                maxshape=(None, *frame.shape),
                # One frame per chunk, so a reader pulling a single timestep
                # decompresses only that timestep.
                chunks=(1, *frame.shape),
                **CAMERA_COMPRESSION,
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
