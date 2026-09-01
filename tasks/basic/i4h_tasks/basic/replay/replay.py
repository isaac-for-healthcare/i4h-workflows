# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay a recorded episode as a workflow node.

Replay is a task that sources actions from HDF5 instead of from a controller
or policy, which means a replayed segment can be spliced
into a live workflow, e.g. replay the approach, then hand off to a policy for the
grasp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from i4h_common.episode import Episode, action_path, demo_names
from i4h_common.world import apply_action
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


class Replay(Task):
    """Feed recorded actions to the robot, one frame per tick."""

    advance_on_success = True

    @dataclass
    class Outputs:
        frames: int = 0
        completed: bool = False

    def __init__(
        self,
        dataset: str | Path,
        *,
        episode: int = 0,
        node: str | None = None,
        loop: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.dataset = Path(dataset)
        self.episode = episode
        #: Replay only the frames recorded while this workflow node was active.
        #: Requires a recording carrying segments (see i4h_common.episode).
        self.node = node
        self.loop = loop
        self._actions: np.ndarray | None = None
        self._index = 0

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._actions = self._load()
        self._index = 0

    def _load(self) -> np.ndarray:
        if not self.dataset.is_file():
            raise FileNotFoundError(f"no recording at {self.dataset}")
        with h5py.File(str(self.dataset), "r") as handle:
            names = demo_names(handle)
            if not names:
                raise ValueError(f"{self.dataset} has no demo_* groups")
            if self.episode >= len(names):
                raise IndexError(f"{self.dataset} has {len(names)} episodes; asked for index {self.episode}")
            group = handle["data"][names[self.episode]]
            actions = np.asarray(group[action_path(group)][()], dtype=np.float32)
            if self.node:
                segment = Episode(names[self.episode], group).segment(self.node)
                if segment is None:
                    raise KeyError(
                        f"{self.dataset}:{names[self.episode]} has no segment for node {self.node!r}; "
                        f"the recording may predate node tagging"
                    )
                actions = actions[segment.start : segment.end]
        return actions

    def tick(self, ctx: TickContext) -> Status:
        assert self._actions is not None
        if self._index >= len(self._actions):
            if not self.loop:
                return Status.SUCCESS
            self._index = 0
        frame = self._actions[self._index]
        self._index += 1
        # A recording's actions mean whatever the scene that produced them meant;
        # route through apply_action so an ee_pose scene replays as poses.
        apply_action(ctx.act, np.tile(frame, (ctx.num_envs, 1)))
        return Status.SUCCESS if not self.loop and self._index >= len(self._actions) else Status.RUNNING

    def on_exit(self, ctx: TickContext) -> Outputs:
        total = len(self._actions) if self._actions is not None else 0
        return self.Outputs(frames=self._index, completed=self._index >= total)

    def describe(self) -> str:
        scope = f" node={self.node}" if self.node else ""
        return f"replay {self.dataset.name}[{self.episode}]{scope}"
