# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The authored form of a task declaration.

One type, written by hand, for both kinds of task:

* an **in-process** skill declares it as class attributes on its ``Task``
  subclass — the class is the declaration, so the two cannot disagree;
* a **remote** policy declares it as an entry in its stack's ``catalog.py``.

Discovery turns either into a :class:`~i4h_common.manifest.TaskSpec` by adding what
the *location* already knows — project, name, runtime. Nothing is restated.

It lives in ``common`` because that is the only package both sides depend on: a
policy backend needs ``common`` for :class:`~i4h_common.server.PolicyServer` and
nothing else, and giving it ``engine`` would assert a relationship that
does not exist — a backend never runs the engine, it answers observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskDef:
    """What a human writes to declare a task.

    Everything here is a real decision. Ports come from ``Inputs``/``Outputs``,
    the backend entry point from convention, and ``dof``/``action_space`` from
    the scene — none of them appear.
    """

    #: One line, shown by ``run.sh list --tasks``. Required for every task.
    summary: str = ""
    #: Optional language instruction when it adds detail beyond ``summary``.
    prompt: str = ""

    # -- what the scene must provide -------------------------------------
    #: Remote tasks name the arm they drive, so lint can match them to a scene.
    embodiment: str = ""
    #: Cameras this checkpoint was trained on.
    cameras: tuple[str, ...] = ()
    #: Anything else the scene must offer, e.g. ``{"action_space": "ee_pose"}``.
    requires: dict[str, Any] = field(default_factory=dict)

    # -- optional planning metadata --------------------------------------
    precondition: dict[str, Any] = field(default_factory=dict)
    postcondition: dict[str, Any] = field(default_factory=dict)

    # -- backend-only: never crosses the wire ----------------------------
    #: Checkpoint configuration, read by this stack's ``server.py``.
    model: dict[str, Any] = field(default_factory=dict)
    #: Observation shaping, read by the same server.
    observation: dict[str, Any] = field(default_factory=dict)
    #: Finetuning hyper-parameters. **Its presence marks the task trainable** —
    #: a released checkpoint nobody retrains simply omits it.
    train: dict[str, Any] = field(default_factory=dict)

    #: Override the conventional ``i4h_tasks.<project>.server:main`` entry point.
    backend_entry: str = ""

    @property
    def trainable(self) -> bool:
        return bool(self.train)

    def merged_requires(self) -> dict[str, Any]:
        """``requires`` with the embodiment and cameras folded in."""
        out: dict[str, Any] = {}
        if self.embodiment:
            out["embodiment"] = self.embodiment
        if self.cameras:
            out["cameras"] = list(self.cameras)
        out.update(self.requires)
        return out


class TaskDefError(ValueError):
    """A task manifest is malformed."""


def load_taskdef(path: Path) -> tuple[str, str | None, TaskDef]:
    """Read one task manifest. Returns ``(runtime, impl, definition)``.

    ``impl`` decides the runtime: a class to point at means the task runs
    in-process, no class means it is served over the bus. Declaring both would
    let them contradict each other.
    """
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise TaskDefError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TaskDefError(f"{path}: expected a mapping")

    known = {f.name for f in fields(TaskDef)} | {"impl", "name"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise TaskDefError(f"{path}: unknown keys {unknown}")

    declared = raw.get("name")
    if declared is not None and str(declared) != path.stem:
        raise TaskDefError(f"{path}: declares name {declared!r} but the file is named {path.stem!r}")

    impl = raw.get("impl")
    runtime = "inprocess" if impl else "remote"
    if not raw.get("summary"):
        raise TaskDefError(f"{path}: every task must define a summary")
    if runtime == "remote" and not raw.get("embodiment"):
        raise TaskDefError(
            f"{path}: declares no impl, so it is served over the bus — and a remote task must name "
            f"an embodiment for lint to match it to a scene"
        )
    if raw.get("prompt") and str(raw["prompt"]).strip().casefold() == str(raw["summary"]).strip().casefold():
        raise TaskDefError(f"{path}: prompt duplicates summary; omit prompt unless it adds detail")

    return (
        runtime,
        (str(impl) if impl else None),
        TaskDef(
            summary=str(raw.get("summary", "")),
            prompt=str(raw.get("prompt", "")),
            embodiment=str(raw.get("embodiment", "")),
            cameras=tuple(raw.get("cameras", ()) or ()),
            requires=dict(raw.get("requires", {}) or {}),
            precondition=dict(raw.get("precondition", {}) or {}),
            postcondition=dict(raw.get("postcondition", {}) or {}),
            model=dict(raw.get("model", {}) or {}),
            observation=dict(raw.get("observation", {}) or {}),
            train=dict(raw.get("train", {}) or {}),
            backend_entry=str(raw.get("backend_entry", "")),
        ),
    )
