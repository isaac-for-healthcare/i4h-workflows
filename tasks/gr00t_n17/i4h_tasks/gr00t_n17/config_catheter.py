# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register the catheter navigation modalities used by the N1.7 backend.

A sibling of ``config.py`` rather than an addition to it. Only one modality
config can be registered per embodiment tag, and ``NEW_EMBODIMENT`` is the only
tag available to embodiments GR00T does not ship, so SO-ARM and the catheter
cannot both be registered in one process. ``launch_finetune.py`` loads exactly
one of these files by path, chosen by ``train.modality_config`` in the task
manifest, so the registration a run gets is the one its task asked for.

The group names are not free: they have to match the ``state_split`` and
``action_split`` groups in ``arena/i4h_arena/embodiments/manifest/catheter.yaml``,
because those are what dataset conversion writes into the dataset's
``modality.json``, and the video key has to match the camera the scene
publishes. A mismatch here does not raise -- the loader simply finds no such
key -- so ``tests/test_catheter_modality.py`` pins them against the manifest.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

#: Both catheter groups are absolute, which is the opposite of the arm case
#: next door and worth saying why. ``RELATIVE`` means an action is an offset
#: from the current state, and GR00T computes a second set of relative
#: statistics for those keys. These actions are velocity commands -- insertion
#: metres per second, rotation radians per second -- that the environment
#: integrates itself. The commanded value stands alone rather than describing a
#: displacement from where the catheter currently is, which is the same
#: situation as SO-ARM's gripper signal.
CATHETER_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["fluoroscopy"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["catheter", "carm"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["catheter", "carm"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(CATHETER_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
