# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path

from isaaclab_arena_gr00t.config.dataset_config import Gr00tDatasetConfig


@dataclass
class ExtendedDatasetConfig(Gr00tDatasetConfig):
    """Extended Gr00tDatasetConfig with semantic segmentation support.

    This class extends the original config to support semantic segmentation streams
    without modifying the IsaacLab-Arena source files.
    """

    camera_mappings: dict[str, str] = field(
        default=None,
        metadata={
            "description": "Dictionary mapping HDF5 camera names to semantic segmentation names. "
            "E.g. {'robot_head_cam_rgb': 'observation.images.ego_view', "
            "'robot_head_cam_semantic_segmentation': 'observation.images.seg_view'}"
        },
    )
    use_rheo_converter: bool = field(
        default=False,
        metadata={
            "description": (
                "If True, use rheo HDF5 layout: obs/robot_joint_state, "
                "obs/robot_dex3_joint_state, processed_actions (WBC+PINK output)."
            )
        },
    )
    rheo_action_key: str = field(
        default="processed_actions",
        metadata={
            "description": (
                "HDF5 key for action. Use 'processed_actions' (43-D, WBC+PINK joint "
                "targets, same as executed) for imitation/replay."
            )
        },
    )
    rheo_28d_state_action: bool = field(
        default=False,
        metadata={
            "description": (
                "If True (with use_rheo_converter), output 28-D state/action: "
                "left_arm(7)+right_arm(7)+left_hand(7)+right_hand(7)."
            )
        },
    )
    rheo_camera_mappings_obs: dict[str, str] = field(
        default=None,
        metadata={
            "description": (
                "When use_rheo_converter: map HDF5 obs key to LeRobot video key. "
                "E.g. {'front_camera': 'observation.images.cam_room', "
                "'left_wrist_camera': 'observation.images.cam_left_wrist', "
                "'right_wrist_camera': 'observation.images.cam_right_wrist'}."
            )
        },
    )

    def __post_init__(self):
        """Extended post-initialization with multi-camera support."""
        if getattr(self, "use_rheo_converter", False) and getattr(self, "rheo_28d_state_action", False):
            _modality_28d = Path(__file__).parent / "modality_assemble_trocar.json"
            if _modality_28d.exists():
                self.modality_template_path = _modality_28d
            if getattr(self, "task_description_lerobot", "").startswith("annotation.human.action"):
                self.task_description_lerobot = "annotation.human.task_description"
        super().__post_init__()

        if self.camera_mappings is None:
            self.camera_mappings = {self.pov_cam_name_sim: self.video_name_lerobot}
