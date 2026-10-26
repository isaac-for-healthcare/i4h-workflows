# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorder terms the i4h embodiments hand to Isaac Lab's RecorderManager.

Arena's policy runner records with them into a requested dataset, and Isaac Lab's
``record_demos.py`` records demonstrations with them.
"""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from i4h_arena.tensor_utils import to_torch


def action_state_recorders(trajectory_terms: Any | None = None) -> Any:
    """Isaac Lab's action/state recorder terms, plus Arena's trajectory terms when it asked for them."""
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: PLC0415
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg  # noqa: PLC0415
    from isaaclab_arena.utils.configclass import combine_configclass_instances  # noqa: PLC0415

    terms = ActionStateRecorderManagerCfg()
    if trajectory_terms is None:
        return terms
    return combine_configclass_instances("RecorderCfg", terms, trajectory_terms, bases=(RecorderManagerBaseCfg,))


class JointPositionRecorder(RecorderTerm):
    """``obs/joint_pos``: the robot's joint positions, where the dataset converter reads its state.

    The articulations named in ``assets`` are concatenated in order. Within one,
    joints follow ``joint_names`` when it is given, otherwise the articulation's own
    order. An embodiment whose policy observations already hold ``joint_pos``
    records it through Isaac Lab's observation term instead.
    """

    def __init__(self, cfg: "JointPositionRecorderCfg", env: Any) -> None:
        super().__init__(cfg, env)
        self._indices: list[list[int]] | None = None

    def record_pre_step(self) -> tuple[str, torch.Tensor]:
        if self._indices is None:
            self._indices = []
            for asset in self.cfg.assets:
                names = list(self._env.scene[asset].data.joint_names)
                order = self.cfg.joint_names or names
                self._indices.append([names.index(name) for name in order])
        parts = [
            to_torch(self._env.scene[asset].data.joint_pos)[:, indices]
            for asset, indices in zip(self.cfg.assets, self._indices, strict=True)
        ]
        return "obs/joint_pos", torch.cat(parts, dim=-1).clone()


@configclass
class JointPositionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = JointPositionRecorder
    assets: tuple[str, ...] = ("robot",)
    joint_names: tuple[str, ...] = ()


@configclass
class _JointPositionTermsCfg:
    record_joint_pos: JointPositionRecorderCfg = JointPositionRecorderCfg()


def with_joint_positions(recorders: Any, *, assets: tuple[str, ...] = ("robot",), joint_names=()) -> Any:
    """``recorders`` plus ``obs/joint_pos``, for an embodiment whose observations do not record it."""
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg  # noqa: PLC0415
    from isaaclab_arena.utils.configclass import combine_configclass_instances  # noqa: PLC0415

    terms = _JointPositionTermsCfg()
    terms.record_joint_pos.assets = tuple(assets)
    terms.record_joint_pos.joint_names = tuple(joint_names)
    return combine_configclass_instances("RecorderCfg", recorders, terms, bases=(RecorderManagerBaseCfg,))
