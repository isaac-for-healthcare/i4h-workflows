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
from openpi.policies import policy_config
from openpi_client import image_tools
from policy_runner.pi0.config import get_config


class PI0PolicyRunner:
    """
    Policy runner for PI0 policy, based on the openpi library.

    Args:
        ckpt_path: Path to the checkpoint file.
        repo_id: Repository ID of the original training dataset.
        task_description: Task description. Default is "Perform a liver ultrasound."

    """

    def __init__(
        self,
        ckpt_path,
        repo_id,
        task_description="Perform a liver ultrasound.",
    ):
        config = get_config(name="robotic_ultrasound", repo_id=repo_id)
        print(f"Loading model from {ckpt_path}...")
        self.model = policy_config.create_trained_policy(config, ckpt_path)
        # Prompt for the model
        self.task_description = task_description

    def infer(self, room_img, wrist_img, current_state) -> torch.Tensor:
        room_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(room_img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))

        element = {
            "observation/image": room_img,
            "observation/wrist_image": wrist_img,
            "observation/state": current_state,
            "prompt": self.task_description,
        }
        # Query model to get action
        return self.model.infer(element)["actions"]
