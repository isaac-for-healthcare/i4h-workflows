# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G1 locomanip modality config for GR00T N1.6, shared by the camera variants.

The state and action slices match ``modality.json`` as ``i4h-dataset convert
--robot g1`` writes it. Only the video keys differ between variants, and each
one must name a camera the recording holds: GR00T's episode loader asserts
that every requested video exists in the dataset's ``info.json``.
"""

from __future__ import annotations

from collections.abc import Sequence

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ACTION_KEYS = (
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "base_height_command",
    "navigate_command",
)


def g1_locomanip_config(video_keys: Sequence[str]) -> dict[str, ModalityConfig]:
    """Return the G1 locomanip modality config with the given video keys."""
    absolute = ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=list(video_keys),
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=list(ACTION_KEYS),
            action_configs=[absolute for _ in ACTION_KEYS],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    }


def register_g1_locomanip(video_keys: Sequence[str]) -> None:
    register_modality_config(g1_locomanip_config(video_keys), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
