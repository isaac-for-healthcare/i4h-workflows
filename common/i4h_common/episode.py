# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The HDF5 episode schema shared by the writer and every reader.

Layout::

    /data                              attrs: env_args, total, ...
      demo_0                           attrs: success, num_samples, workflow, ...
        actions            (T, A)      or obs/actions on compatible files
        obs/joint_pos      (T, D)
        obs/<camera>       (T, H, W, 3) uint8
        segments           (S,)

``segments`` is a structured array of
``(node, task_id, start, end)`` recording which workflow node was active for each
frame range. It is optional, and readers must tolerate its absence. With it,
``mimic`` can augment a single skill and ``annotator`` can label per skill
instead of per episode — neither is expressible against a flat episode.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DATA_GROUP = "data"
SEGMENTS = "segments"

#: h5py structured dtype for the segments dataset.
SEGMENT_DTYPE = np.dtype(
    [("node", h5py.string_dtype()), ("task_id", h5py.string_dtype()), ("start", "<i8"), ("end", "<i8")]
)


class EpisodeError(ValueError):
    """A recording does not match the schema."""


@dataclass(frozen=True, slots=True)
class Segment:
    """Frames ``[start, end)`` were produced while ``node`` was active."""

    node: str
    task_id: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def demo_names(file_or_group: h5py.File | h5py.Group) -> list[str]:
    """Demo group names in numeric order (``demo_0``, ``demo_1``, … ``demo_10``)."""
    root = file_or_group
    if isinstance(file_or_group, h5py.File):
        if DATA_GROUP not in file_or_group:
            raise EpisodeError(f"{file_or_group.filename}: missing /{DATA_GROUP} group")
        root = file_or_group[DATA_GROUP]
    names = [name for name in root if name.startswith("demo_")]
    return sorted(names, key=_demo_sort_key)


def _demo_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix("demo_")
    return (int(suffix), "") if suffix.isdigit() else (1 << 30, name)


def action_path(demo: h5py.Group) -> str:
    """Where actions live. Newer recordings use ``actions``; older ones ``obs/actions``."""
    for candidate in ("actions", "obs/actions"):
        if candidate in demo:
            return candidate
    raise EpisodeError(f"{demo.name}: expected 'actions' or 'obs/actions'")


def state_path(demo: h5py.Group) -> str | None:
    return "obs/joint_pos" if "obs/joint_pos" in demo else None


def camera_keys(demo: h5py.Group) -> list[str]:
    """Names of uint8 ``(T, H, W, 3)`` datasets under ``obs/``."""
    obs = demo.get("obs")
    if obs is None:
        return []
    return sorted(
        name
        for name, dataset in obs.items()
        if isinstance(dataset, h5py.Dataset) and dataset.dtype == np.uint8 and dataset.ndim == 4
    )


def write_segments(demo: h5py.Group, segments: Sequence[Segment]) -> None:
    """Attach node segmentation to a demo, replacing any existing dataset."""
    if SEGMENTS in demo:
        del demo[SEGMENTS]
    if not segments:
        return
    array = np.array(
        [(s.node, s.task_id, s.start, s.end) for s in segments],
        dtype=SEGMENT_DTYPE,
    )
    demo.create_dataset(SEGMENTS, data=array)


def read_segments(demo: h5py.Group) -> tuple[Segment, ...]:
    """Node segmentation, or ``()`` for recordings written without it."""
    dataset = demo.get(SEGMENTS)
    if dataset is None:
        return ()
    out: list[Segment] = []
    for row in dataset[()]:
        out.append(
            Segment(
                node=_as_str(row["node"]),
                task_id=_as_str(row["task_id"]),
                start=int(row["start"]),
                end=int(row["end"]),
            )
        )
    return tuple(out)


def _as_str(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass(frozen=True, slots=True)
class Episode:
    """Read-only accessor for one ``demo_N`` group."""

    name: str
    group: h5py.Group

    @property
    def success(self) -> bool:
        return bool(self.group.attrs.get("success", False))

    @property
    def num_samples(self) -> int:
        declared = self.group.attrs.get("num_samples")
        if declared is not None:
            return int(declared)
        return int(self.group[action_path(self.group)].shape[0])

    @property
    def actions(self) -> np.ndarray:
        return self.group[action_path(self.group)][()]

    @property
    def states(self) -> np.ndarray | None:
        path = state_path(self.group)
        return self.group[path][()] if path else None

    @property
    def cameras(self) -> list[str]:
        return camera_keys(self.group)

    def camera(self, name: str) -> np.ndarray:
        return self.group[f"obs/{name}"][()]

    @property
    def segments(self) -> tuple[Segment, ...]:
        return read_segments(self.group)

    def segment(self, node: str) -> Segment | None:
        """The first segment produced by ``node``, if the recording is tagged."""
        return next((s for s in self.segments if s.node == node), None)

    def validate(self) -> None:
        actions = self.group.get(action_path(self.group))
        if actions is None or actions.shape[0] == 0:
            raise EpisodeError(f"{self.name}: actions dataset is empty")
        states_path = state_path(self.group)
        if states_path and self.group[states_path].shape[0] != actions.shape[0]:
            raise EpisodeError(f"{self.name}: {states_path} length does not match actions")
        total = actions.shape[0]
        for segment in self.segments:
            if not 0 <= segment.start <= segment.end <= total:
                raise EpisodeError(
                    f"{self.name}: segment {segment.node} range {segment.start}:{segment.end} out of [0,{total}]"
                )


def open_episodes(path: str | Path, mode: str = "r") -> Iterator[tuple[h5py.File, list[Episode]]]:
    """Context-manager-ish helper: yields the open file plus its episodes.

    Usage::

        with h5py.File(path) as handle:
            for episode in episodes(handle):
                ...
    """
    handle = h5py.File(str(path), mode)
    try:
        yield handle, [Episode(name, handle[DATA_GROUP][name]) for name in demo_names(handle)]
    finally:
        handle.close()


def episodes(handle: h5py.File) -> list[Episode]:
    """Episodes in an already-open file."""
    data = handle[DATA_GROUP]
    return [Episode(name, data[name]) for name in demo_names(handle)]
