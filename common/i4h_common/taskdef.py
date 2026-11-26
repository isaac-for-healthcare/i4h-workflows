# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A policy task's declaration: ``tasks/<stack>/i4h_tasks/<stack>/manifest/<task>.yaml``.

Each policy stack runs in its own environment, so its tasks are declared as data
that the stack's ``serve`` and ``train`` modules and the light tooling all read
the same way. It lives in ``common`` because that is the only package they share.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskDef:
    """What a policy task declares."""

    #: One line describing the task. Required.
    summary: str = ""
    #: Optional language instruction when it adds detail beyond ``summary``.
    prompt: str = ""
    #: The embodiment the checkpoint drives.
    embodiment: str = ""
    #: Cameras this checkpoint was trained on.
    cameras: tuple[str, ...] = ()
    #: Anything else the scene must offer, e.g. ``{"action_space": "ee_pose"}``.
    requires: dict[str, Any] = field(default_factory=dict)
    #: Checkpoint configuration, read by the stack's ``serve`` module.
    model: dict[str, Any] = field(default_factory=dict)
    #: Observation shaping, read by the same module and by dataset conversion.
    observation: dict[str, Any] = field(default_factory=dict)
    #: Finetuning hyper-parameters. **Its presence marks the task trainable** --
    #: a released checkpoint nobody retrains simply omits it.
    train: dict[str, Any] = field(default_factory=dict)

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


def load_taskdef(path: Path) -> TaskDef:
    """Read one task manifest."""
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise TaskDefError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TaskDefError(f"{path}: expected a mapping")

    unknown = sorted(set(raw) - {f.name for f in fields(TaskDef)})
    if unknown:
        raise TaskDefError(f"{path}: unknown keys {unknown}")
    if not raw.get("summary"):
        raise TaskDefError(f"{path}: every task must define a summary")
    if not raw.get("embodiment"):
        raise TaskDefError(f"{path}: a policy task must name the embodiment its checkpoint drives")
    if raw.get("prompt") and str(raw["prompt"]).strip().casefold() == str(raw["summary"]).strip().casefold():
        raise TaskDefError(f"{path}: prompt duplicates summary; omit prompt unless it adds detail")

    return TaskDef(
        summary=str(raw["summary"]),
        prompt=str(raw.get("prompt", "")),
        embodiment=str(raw["embodiment"]),
        cameras=tuple(raw.get("cameras", ()) or ()),
        requires=dict(raw.get("requires", {}) or {}),
        model=dict(raw.get("model", {}) or {}),
        observation=dict(raw.get("observation", {}) or {}),
        train=dict(raw.get("train", {}) or {}),
    )
