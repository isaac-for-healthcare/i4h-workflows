# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorder terms the i4h embodiments hand to Isaac Lab's RecorderManager.

Workflow runs record through ``run.sh --record`` and clear these terms, so they
only matter to Isaac Lab and Arena tools that build the environment themselves,
such as ``record_demos.py``.
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
