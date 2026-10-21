# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The HDF5 episode schema shared by the writer and every reader.

Layout::

    /                                  attrs: format_version
    /data                              attrs: env_args, total, ...
      demo_0                           attrs: success, num_samples, recording_version, workflow, ...
        actions            (T, A)      or obs/actions on compatible files
        obs/joint_pos      (T, D)
        obs/<camera>       (T, H, W, 3) uint8
        diagnostics/<name> (T,) or (T, K) float32
        segments           (S,)
        initial_state/<family>/<asset>/<field>   (1, ...)

Sample ``t`` is one simulation step: the action applied on that step, and the
joints, camera frames and diagnostics read after it. ``num_samples`` is the
number of samples, which is ``T``.

**The step that ends an episode through a termination term is not recorded.**
IsaacLab's ``ManagerBasedRLEnv.step`` resets a terminated environment before it
returns, so the joints and frames that step leaves behind already belong to the
next, re-randomized episode. A demo that ends this way -- every success whose
predicate is a termination term, such as the liver scan's -- therefore holds
one sample fewer than the steps the attempt took, and its last stored action is
the one *before* the action that triggered the termination. Replaying such a
demo applies every stored action and stops one action short of the state the
simulator graded; the replay itself still completes. Nothing after such a
reset is recorded either. The workflow may consume the retained termination
signal, including across remote-policy WAITING ticks. If it is still running
and requests another physics step, the attempt ends as ``aborted``, even when
``success`` fired. Aborted attempts are retained only with
``--record-failures``; empty attempts are always discarded. An explicit
operator scene reset restarts the recording.

Demos recorded under this rule carry ``recording_version`` (see
:data:`RECORDING_VERSION`). A demo without it was recorded before the rule: if
its episode ended through a termination, its last sample shows the next
episode (the arm back at its reset pose), so re-record it before grading its
final frames.

``segments`` is a structured array of
``(node, task_id, start, end)`` recording which workflow node was active for each
frame range. It is optional, and readers must tolerate its absence. With it,
``mimic`` can augment a single skill and ``annotator`` can label per skill
instead of per episode — neither is expressible against a flat episode.

``diagnostics`` holds per-frame measurements of the simulation rather than
observations, so a recording can be judged after the fact instead of only by
watching it. It is optional and scene-specific, and readers must tolerate its
absence. It sits outside ``obs`` so that everything walking observations is
unaffected by it. Frames a measurement was not reported for hold ``nan``.

``initial_state`` is the scene as the episode's reset left it, before its first
action: every articulation's root pose and joints, every rigid object's pose,
in IsaacLab's ``InteractiveScene.get_state(is_relative=True)`` layout with env 0
kept as a batch of one. Actions alone do not determine an episode when the
reset randomizes object poses, so replay restores this state instead of drawing
a new one. It is optional; recordings made before it existed replay from a
fresh reset.

``format_version`` on the file root is Isaac Lab's dataset format attribute
(:data:`ISAACLAB_FORMAT_VERSION_ATTR`). Isaac Lab's ``HDF5DatasetFileHandler``
reads a file without it as the legacy wxyz layout and rewrites every
``root_pose`` it loads to xyzw. The poses in ``initial_state`` come from Isaac
Lab 3 and are already xyzw, so the arena recorder stamps
:data:`ISAACLAB_FORMAT_VERSION` on each file it creates and the offline tools
carry the source file's root attributes into the files they write. A file the
recorder appends to keeps whatever it declared.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DATA_GROUP = "data"
SEGMENTS = "segments"
DIAGNOSTICS_GROUP = "diagnostics"
INITIAL_STATE_GROUP = "initial_state"

#: Demo attribute naming the recording rules a demo was written under.
RECORDING_VERSION_ATTR = "recording_version"
#: Rules the arena recorder writes today. ``1``: the step that ends an episode
#: through a termination term is not recorded (v0.9.0, NVBug 6941887). A demo
#: without the attribute reads as ``0``: recorded before any of these rules.
RECORDING_VERSION = 1

#: File-root attribute Isaac Lab's ``HDF5DatasetFileHandler`` reads to tell an
#: xyzw recording from a legacy wxyz one (absent or ``0``).
ISAACLAB_FORMAT_VERSION_ATTR = "format_version"
#: Isaac Lab's ``DATASET_FORMAT_VERSION`` for xyzw quaternions, which is what
#: ``InteractiveScene.get_state`` returns in Isaac Lab 3.
ISAACLAB_FORMAT_VERSION = 1

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


def recording_version(demo: h5py.Group) -> int:
    """The recording rules ``demo`` was written under; ``0`` when it predates them."""
    return int(demo.attrs.get(RECORDING_VERSION_ATTR, 0))


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


def diagnostic_keys(demo: h5py.Group) -> list[str]:
    """Names of per-frame diagnostic datasets, or ``[]`` when none were written."""
    group = demo.get(DIAGNOSTICS_GROUP)
    if group is None:
        return []
    return sorted(name for name, value in group.items() if isinstance(value, h5py.Dataset))


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


#: Nested ``{family: {asset: {field: array}}}`` scene state; see the module docstring.
StateTree = Mapping[str, Any]


def write_initial_state(demo: h5py.Group, state: StateTree | None) -> None:
    """Attach the scene state an episode started from, replacing any existing one."""
    if INITIAL_STATE_GROUP in demo:
        del demo[INITIAL_STATE_GROUP]
    if not state:
        return
    _write_tree(demo.create_group(INITIAL_STATE_GROUP), state)


def _write_tree(group: h5py.Group, tree: StateTree) -> None:
    for key, value in tree.items():
        if isinstance(value, Mapping):
            _write_tree(group.create_group(key), value)
        else:
            group.create_dataset(key, data=np.asarray(value))


def read_initial_state(demo: h5py.Group) -> dict[str, Any] | None:
    """The scene state an episode started from, or ``None`` for recordings without it."""
    group = demo.get(INITIAL_STATE_GROUP)
    return None if group is None else _read_tree(group)


def _read_tree(group: h5py.Group) -> dict[str, Any]:
    return {key: _read_tree(value) if isinstance(value, h5py.Group) else value[()] for key, value in group.items()}


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
    def recording_version(self) -> int:
        """The recording rules this demo was written under; ``0`` for demos that predate them."""
        return recording_version(self.group)

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
    def diagnostics(self) -> list[str]:
        return diagnostic_keys(self.group)

    def diagnostic(self, name: str) -> np.ndarray:
        """One per-frame measurement. Absent frames hold ``nan``."""
        try:
            return self.group[f"{DIAGNOSTICS_GROUP}/{name}"][()]
        except KeyError as exc:
            raise EpisodeError(f"{self.name}: no diagnostic {name!r}; recorded {self.diagnostics}") from exc

    @property
    def segments(self) -> tuple[Segment, ...]:
        return read_segments(self.group)

    @property
    def initial_state(self) -> dict[str, Any] | None:
        return read_initial_state(self.group)

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
        for name in self.diagnostics:
            length = self.group[f"{DIAGNOSTICS_GROUP}/{name}"].shape[0]
            if length != total:
                raise EpisodeError(f"{self.name}: diagnostic {name} has {length} frames, actions have {total}")
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
