# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import numpy as np
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from policy_runner.gr00tn1_policy.utils import DATA_CONFIG_MAP

class GR00TN1PolicyRunner:
    """
    Policy runner for GR00TN1 policy, based on the openpi library.

    Args:
        ckpt_path: Path to the checkpoint file.
        repo_id: Repository ID of the original training dataset.
        task_description: Task description. Default is "Perform a liver ultrasound."

    """

    def __init__(
        self,
        ckpt_path,
        data_config,
        embodiment_tag,
        task_description="Perform a liver ultrasound.",
        device="cuda" if torch.cuda.is_available() else "cpu",
        action_key="action.panda_hand",
    ):
        print(f"Loading model from {ckpt_path}...")
        data_config = DATA_CONFIG_MAP[data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        self.model: BasePolicy = Gr00tPolicy(
            model_path=ckpt_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=embodiment_tag,
            device=device,
        )
        self.task_description = task_description
        self.action_key = action_key

    def infer(self, room_img, wrist_img, current_state) -> torch.Tensor:
        # Prepare input data with batch dimension for model
        data_point = {
            "video.room": np.expand_dims(room_img, axis=0),
            "video.wrist": np.expand_dims(wrist_img, axis=0),
            "state.panda_hand": np.expand_dims(np.array(current_state), axis=0),
            "annotation.human.task_description": self.task_description,
        }

        return self.model.get_action(data_point)[self.action_key]
