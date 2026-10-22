# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorder terms the i4h embodiments hand to Isaac Lab's RecorderManager.

Arena's policy runner records with them into a requested dataset, and Isaac Lab's
``record_demos.py`` records demonstrations with them.
"""

from __future__ import annotations

from typing import Any


def action_state_recorders(trajectory_terms: Any | None = None) -> Any:
    """Isaac Lab's action/state recorder terms, plus Arena's trajectory terms when it asked for them."""
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: PLC0415
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg  # noqa: PLC0415
    from isaaclab_arena.utils.configclass import combine_configclass_instances  # noqa: PLC0415

    terms = ActionStateRecorderManagerCfg()
    if trajectory_terms is None:
        return terms
    return combine_configclass_instances("RecorderCfg", terms, trajectory_terms, bases=(RecorderManagerBaseCfg,))
