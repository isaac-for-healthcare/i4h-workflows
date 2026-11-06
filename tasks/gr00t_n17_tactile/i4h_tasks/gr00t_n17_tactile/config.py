# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register the H2 + SharpaWave modalities, including the two tactile categories.

Same mechanism as ``tasks/gr00t_n17/config.py``: importing this module registers
the modality config under ``EmbodimentTag.NEW_EMBODIMENT`` as a side effect, and
both ``server.py`` and ``train.py`` resolve the manifest's ``modality_config``
name to this file. The ``tactile`` and ``tactile_force`` categories are read by
the data loader and processor of the GR00T patch series (``patches/README.md``);
the vanilla N1.7 loader ignores them.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

from i4h_tasks.gr00t_n17_tactile.tactile.interface import FINGERTIPS

#: 32-step chunk of relative joint targets, one ActionConfig per group.
ACTION_HORIZON = 32

#: The causal force window the T-Rex encoder consumes ends at the current
#: frame; the deform maps and the force window both extend 16 frames into the
#: future so that delay augmentation can shift either stream by up to
#: ``force_delay_max`` frames. Both need the loader's ``allow_padding``.
FORCE_WINDOW = 16
DELAY_MAX = 16

_RELATIVE_JOINTS = ActionConfig(
    rep=ActionRepresentation.RELATIVE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

H2_SHARPA_TACTILE_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["high", "left_wrist_view", "right_wrist_view"]),
    # The hand joint groups serve twice: as state, and as the joint angles the
    # fingertip forward kinematics reads to place every fingertip's tactile features.
    "state": ModalityConfig(delta_indices=[0], modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"]),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
        action_configs=[_RELATIVE_JOINTS] * 4,
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
    # One 240x240 single-channel deformation map per fingertip, in the canonical
    # order the per-finger positional embedding and the kinematics assume.
    "tactile": ModalityConfig(
        delta_indices=list(range(DELAY_MAX + 1)),
        modality_keys=[f"{fingertip}_deform" for fingertip in FINGERTIPS],
    ),
    # Ten fingertips x six axes per frame as one numeric column. A separate
    # category because it is a parquet column that needs a history window,
    # neither a video (``tactile``) nor a slice of ``observation.state``.
    "tactile_force": ModalityConfig(
        delta_indices=list(range(-(FORCE_WINDOW - 1), DELAY_MAX + 1)),
        modality_keys=["force"],
    ),
}

register_modality_config(H2_SHARPA_TACTILE_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
