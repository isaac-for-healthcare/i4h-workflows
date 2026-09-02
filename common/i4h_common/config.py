# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot descriptors — the third and last thing that must be config.

``tools/dataset`` needs LeRobot column names, ``tasks/teleop`` needs the leader
calibration, and ``arena`` needs the home pose. None of those venvs can import
the others, so these cross a boundary and are therefore data.

What is *not* here: ``dof``, ``action_space``, ``gripper`` and ``control_hz``.
The scene declares those. The same arm mounted under a different controller has
a different action width, and simulated at a different decimation has a
different control rate — neither is a property of the arm.

The descriptors live beside the embodiments that define the robots
(``arena/i4h_arena/embodiments/manifest/``); this module globs for that directory
rather than hardcoding it, so ``common`` stays ignorant of arena's layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path

import yaml

from i4h_common.paths import workflow_root


@dataclass(frozen=True, slots=True)
class RobotConfig:
    """Everything about a robot that more than one venv needs to agree on."""

    name: str
    joint_names: tuple[str, ...] = ()
    home_joint_pos_rad: tuple[float, ...] = ()
    #: Optional override of the articulation's own home pose. Normally empty:
    #: IsaacLab sets `default_joint_pos` from the embodiment cfg, and the scene
    #: view falls back to it, so only a robot that must start somewhere *else*
    #: needs this.
    #: LeRobot column names, in action order.
    action_names: tuple[str, ...] = ()
    state_names: tuple[str, ...] = ()
    #: ``[(group, start, stop), ...]`` splits for policy state/action heads.
    state_split: tuple[tuple[str, int, int], ...] = ()
    action_split: tuple[tuple[str, int, int], ...] = ()
    teleop_devices: tuple[str, ...] = ()
    gripper_open: float = 0.0
    gripper_closed: float = 0.0
    joint_limits: tuple[tuple[float, float], ...] = ()
    #: Calibrated ranges used to translate between Isaac joint radians and
    #: LeRobot's checkpoint-facing joint coordinates.
    isaaclab_joint_pos_limit_range: tuple[tuple[float, float], ...] = ()
    lerobot_joint_pos_limit_range: tuple[tuple[float, float], ...] = ()
    extra: dict[str, object] = field(default_factory=dict)

    def split_width(self, group: str) -> int:
        """Width of an ``action_split`` group, e.g. how many arm joints.

        Derived rather than declared: the split already states it, and a second
        field would be one more thing to keep in agreement.
        """
        for name, start, stop in self.action_split:
            if name == group:
                return int(stop) - int(start)
        raise KeyError(f"{self.name}: no action_split group {group!r}; have {[n for n, _s, _e in self.action_split]}")

    def joint_index(self, name: str) -> int:
        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise KeyError(f"{self.name}: unknown joint {name!r}") from exc


#: Where robot descriptors may live, relative to the workflow root.
ROBOT_GLOBS = ("*/*/embodiments/manifest", "*/embodiments/manifest")


@lru_cache(maxsize=1)
def _robots_dir() -> Path | None:
    for pattern in ROBOT_GLOBS:
        for directory in sorted(workflow_root().glob(pattern)):
            if directory.is_dir() and ".venv" not in directory.parts:
                return directory
    return None


def available_robots() -> tuple[str, ...]:
    directory = _robots_dir()
    if directory is None:
        return ()
    return tuple(sorted(path.stem for path in directory.glob("*.yaml")))


@cache
def get_robot_config(name: str) -> RobotConfig:
    """Load ``arena/i4h_arena/embodiments/manifest/<name>.yaml``."""
    directory = _robots_dir()
    path = (directory / f"{name}.yaml") if directory else Path(f"{name}.yaml")
    if not path.is_file():
        raise KeyError(f"no robot descriptor at {path}; have {list(available_robots())}")
    with path.open("rb") as handle:
        raw = yaml.safe_load(handle) or {}

    def _splits(key: str) -> tuple[tuple[str, int, int], ...]:
        return tuple((str(g), int(a), int(b)) for g, a, b in raw.get(key, ()))

    known = {
        "name",
        "joint_names",
        "home_joint_pos_rad",
        "action_names",
        "state_names",
        "state_split",
        "action_split",
        "teleop_devices",
        "gripper_open",
        "gripper_closed",
        "joint_limits",
        "isaaclab_joint_pos_limit_range",
        "lerobot_joint_pos_limit_range",
    }

    def _ranges(key: str) -> tuple[tuple[float, float], ...]:
        return tuple((float(low), float(high)) for low, high in raw.get(key, ()))

    return RobotConfig(
        name=str(raw.get("name", name)),
        joint_names=tuple(raw.get("joint_names", ())),
        home_joint_pos_rad=tuple(float(v) for v in raw.get("home_joint_pos_rad", ())),
        action_names=tuple(raw.get("action_names", ())),
        state_names=tuple(raw.get("state_names", ())),
        state_split=_splits("state_split"),
        action_split=_splits("action_split"),
        teleop_devices=tuple(raw.get("teleop_devices", ())),
        gripper_open=float(raw.get("gripper_open", 0.0)),
        gripper_closed=float(raw.get("gripper_closed", 0.0)),
        joint_limits=_ranges("joint_limits"),
        isaaclab_joint_pos_limit_range=_ranges("isaaclab_joint_pos_limit_range"),
        lerobot_joint_pos_limit_range=_ranges("lerobot_joint_pos_limit_range"),
        extra={k: v for k, v in raw.items() if k not in known},
    )
