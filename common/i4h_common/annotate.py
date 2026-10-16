# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a vision model grades a workflow's recordings.

What success looks like on camera is a property of the workflow, the same layer
as its ``success`` predicate: it holds for every recording of the workflow,
whichever policy, controller or operator produced it. A workflow therefore
declares it as ``Workflow(annotate=Annotate(...))``; ``scripts/e2e/run.sh``
passes it to ``tools/annotator``, and lint checks its cameras against the
cameras the workflow's scene records.

This module is the vocabulary both sides share, so it imports nothing beyond
the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass

#: How ``tools/annotator`` reads the frames: by the end state (a placement), or
#: by whether the action happens across the attempt (a scan or sweep).
JUDGES = ("terminal", "trajectory")


@dataclass(frozen=True, slots=True)
class Annotate:
    """Grading settings for a workflow's recordings. Every field is optional.

    ``criterion``
        The visible success condition. Needed when the policy's language prompt
        names no state a camera can see; empty grades against that prompt.
    ``cameras``
        Recorded cameras to show, sampled at the same instants. Empty shows the
        first camera by name.
    ``judge``
        One of :data:`JUDGES`; empty uses the annotator's default, ``terminal``.
    ``frames``
        Frames sampled per camera; ``0`` uses the annotator's default.
    """

    criterion: str = ""
    cameras: tuple[str, ...] = ()
    judge: str = ""
    frames: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "criterion", " ".join(str(self.criterion).split()))
        if isinstance(self.cameras, str):
            raise TypeError("annotate cameras must be a sequence of camera names, not one string")
        cameras = tuple(str(name).strip() for name in self.cameras)
        if any(not name for name in cameras):
            raise ValueError("annotate camera names must not be empty")
        if len(set(cameras)) != len(cameras):
            raise ValueError(f"annotate cameras {list(cameras)} name a camera twice")
        object.__setattr__(self, "cameras", cameras)
        if self.judge and self.judge not in JUDGES:
            raise ValueError(f"annotate judge {self.judge!r} is not one of {list(JUDGES)}")
        if isinstance(self.frames, bool) or not isinstance(self.frames, int) or self.frames < 0:
            raise ValueError(f"annotate frames must be a frame count, or 0 for the default; got {self.frames!r}")
