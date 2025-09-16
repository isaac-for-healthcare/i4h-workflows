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
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from policy_runner.gr00tn1_5.trt.trt_model_forward import setup_tensorrt_engines


class GR00TN1_5_PolicyRunner:
    """
    Policy runner for GR00T N1.5 policy.

    Args:
        ckpt_path: Path to the checkpoint file.
        data_config: Data configuration.
        embodiment_tag: Embodiment tag.
        task_description: Task description. Default is "Perform a liver ultrasound."
        device: Device to run the model on. Only supports "cuda" for now.
    """

    def __init__(
        self,
        ckpt_path,
        data_config,
        embodiment_tag,
        task_description="Grip the scissors and put it into the tray",
        device="cuda",
        trt_engine_path=None,
        trt=False,
    ):
        print(f"Loading model from {ckpt_path}...")
        data_config = DATA_CONFIG_MAP[data_config]
        data_config.video_keys = ["video.room", "video.wrist"]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        if not torch.cuda.is_available():
            raise RuntimeError("Deployment of GR00T N1.5 requires NVIDIA GPU with CUDA 12.0+")

        self.model: BasePolicy = Gr00tPolicy(
            model_path=ckpt_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=embodiment_tag,
            device=device,
        )
        self.task_description = task_description

        if trt:
            setup_tensorrt_engines(self.model, trt_engine_path)

    def infer(self, room_img, wrist_img, current_state) -> torch.Tensor:
        # Prepare input data with batch dimension for model
        data_point = {
            "video.room": np.expand_dims(room_img, axis=0),
            "video.wrist": np.expand_dims(wrist_img, axis=0),
            "state.single_arm": np.expand_dims(np.array(current_state[:5]), axis=0),
            "state.gripper": np.expand_dims(np.array(current_state[5:6]), axis=0),
            "annotation.human.task_description": self.task_description,
        }

        # Get action once (truly efficient)
        action = self.model.get_action(data_point)

        single_arm = action.get("action.single_arm", action.get("state.single_arm", action.get("single_arm", None)))
        gripper = action.get("action.gripper", action.get("state.gripper", action.get("gripper", None)))

        if single_arm is None or gripper is None:
            raise ValueError(f"Could not find valid action format. Available keys: {list(action.keys())}")

        # Convert to tensors if they're numpy arrays
        if isinstance(single_arm, np.ndarray):
            single_arm = torch.from_numpy(single_arm)
        if isinstance(gripper, np.ndarray):
            gripper = torch.from_numpy(gripper)

        # Handle sequence dimensions properly
        if single_arm.dim() == 2 and gripper.dim() == 1:
            # Sequence: arm is [seq_len, 5], gripper is [seq_len] -> reshape gripper to [seq_len, 1]
            gripper = gripper.unsqueeze(-1)
        elif single_arm.dim() > 2 or gripper.dim() > 2:
            # Remove batch dimension if present
            single_arm = single_arm.squeeze(0)
            gripper = gripper.squeeze(0)
            if single_arm.dim() == 2 and gripper.dim() == 1:
                gripper = gripper.unsqueeze(-1)

        result = torch.cat([single_arm, gripper], dim=-1)

        return result
