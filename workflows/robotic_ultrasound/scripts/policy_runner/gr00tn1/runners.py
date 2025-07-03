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

import numpy as np
import torch
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from policy_runner.gr00tn1.utils import DATA_CONFIG_MAP


class GR00TN1PolicyRunner:
    """
    Policy runner for GR00T N1 policy.

    Args:
        ckpt_path: Path to the checkpoint file.
        data_config: Data configuration.
        embodiment_tag: Embodiment tag.
        task_description: Task description. Default is "Perform a liver ultrasound."
        device: Device to run the model on. Default is "cuda" if available, otherwise "cpu".
        action_key: Key to extract from the model output. Default is "action.panda_hand".
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
            denoising_steps=4,
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

    def infer_g1(self, room_img, current_state) -> torch.Tensor:
        # Prepare input data with batch dimension for model
        data_point = {
            "video.rs_view": np.expand_dims(room_img, axis=0),
            "state.left_arm": np.expand_dims(np.array(current_state[15:22]), axis=0),
            "state.left_hand": np.expand_dims(np.array(current_state[22:29]), axis=0),
            "state.right_arm": np.expand_dims(np.array(current_state[29:36]), axis=0),
            "state.right_hand": np.expand_dims(np.array(current_state[36:43]), axis=0),
            "annotation.human.task_description": [self.task_description],
        }
        print(data_point["video.rs_view"].shape)
        print(data_point["state.left_arm"].shape)
        print(data_point["state.left_hand"].shape)
        print(data_point["state.right_arm"].shape)
        print(data_point["state.right_hand"].shape)
        print(type(data_point["annotation.human.task_description"]))
        action_chunk = self.model.get_action(data_point)
        # for key in action_chunk.keys():
        #     print(key, "***********")
        #     print(action_chunk[key].shape, "***********")
        #     print(action_chunk[key].dtype, "***********")
        #     print(action_chunk[key], "***********")
        # concat the action chunk for all the keys
        concat_pred_action = np.concatenate(
            [np.atleast_1d(action_chunk[key]) for key in action_chunk.keys()],
            axis=1,
        )
        print(concat_pred_action.shape, '********')
        return concat_pred_action
