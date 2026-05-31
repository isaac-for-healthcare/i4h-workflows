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

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


class GR00TN1_7_PolicyRunner:
    """Policy runner for GR00T N1.7 policy.

    Supports both PyTorch and TensorRT inference modes. When
    ``trt_engine_path`` is provided, the model's forward passes are
    monkey-patched with TRT engines via ``setup_tensorrt_engines()``.

    Args:
        ckpt_path: Path to the N1.7 checkpoint directory (or HuggingFace id).
        embodiment_tag: Embodiment tag string (resolved case-insensitively).
        task_description: Natural language instruction for the task.
        device: CUDA device string. Only ``"cuda"`` family is supported.
        trt_engine_path: Directory containing TRT engine files. If ``None``,
            uses PyTorch inference.
        trt_mode: TRT acceleration scope. One of ``"n17_full_pipeline"``
            (ViT + LLM + action head), ``"vit_llm_only"``,
            ``"action_head"``, or ``"dit_only"``.
    """

    def __init__(
        self,
        ckpt_path: str,
        embodiment_tag: str = "new_embodiment",
        task_description: str = "Grip the scissors and put it into the tray",
        device: str = "cuda",
        trt_engine_path: str | None = None,
        trt_mode: str = "n17_full_pipeline",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Deployment of GR00T N1.7 requires NVIDIA GPU with CUDA 12.0+")

        print(f"Loading GR00T N1.7 model from {ckpt_path} ...")
        self.model = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.resolve(embodiment_tag),
            model_path=ckpt_path,
            device=device,
        )
        self.task_description = task_description
        self._language_key = self.model.language_key
        print(f"GR00T N1.7 model loaded successfully (language_key={self._language_key!r}).")

        if trt_engine_path is not None:
            self._setup_trt(self.model, trt_engine_path, trt_mode)

    @staticmethod
    def _setup_trt(policy, trt_engine_path: str, mode: str) -> None:
        # runners.py -> gr00tn1_7 -> policy -> scripts -> so_arm_starter
        #            -> workflows -> i4h-workflows-internal
        groot_root = Path(__file__).resolve().parents[5] / "third_party" / "Isaac-GR00T"
        trt_fwd = groot_root / "scripts" / "deployment" / "trt_model_forward.py"

        spec = importlib.util.spec_from_file_location("trt_model_forward", trt_fwd)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot find {trt_fwd}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        print(f"Setting up TensorRT engines from {trt_engine_path} (mode={mode}) ...")
        mod.setup_tensorrt_engines(policy, trt_engine_path, mode=mode)
        print("TensorRT engines loaded successfully.")

    def infer(
        self,
        room_img: np.ndarray,
        wrist_img: np.ndarray,
        current_state: np.ndarray,
    ) -> torch.Tensor:
        """Run a single inference step.

        Args:
            room_img: RGB image from the room camera, uint8, shape ``(H, W, 3)``.
            wrist_img: RGB image from the wrist camera, uint8, shape ``(H, W, 3)``.
            current_state: Joint state array of length >= 6
                           (first 5 = single_arm, index 5 = gripper).

        Returns:
            Tensor of shape ``(chunk_length, 6)`` — concatenated
            ``[single_arm (5), gripper (1)]`` for each step in the action
            chunk.
        """
        obs = {
            "video": {
                "room": room_img[np.newaxis, np.newaxis].astype(np.uint8),
                "wrist": wrist_img[np.newaxis, np.newaxis].astype(np.uint8),
            },
            "state": {
                "single_arm": np.array(current_state[:5], dtype=np.float32)[np.newaxis, np.newaxis],
                "gripper": np.array(current_state[5:6], dtype=np.float32)[np.newaxis, np.newaxis],
            },
            "language": {
                self._language_key: [[self.task_description]],
            },
        }

        action_dict, _info = self.model.get_action(obs)

        single_arm = action_dict.get("single_arm")
        gripper = action_dict.get("gripper")

        if single_arm is None or gripper is None:
            raise ValueError(f"Could not find expected action keys. " f"Available keys: {list(action_dict.keys())}")

        if isinstance(single_arm, np.ndarray):
            single_arm = torch.from_numpy(single_arm)
        if isinstance(gripper, np.ndarray):
            gripper = torch.from_numpy(gripper)

        # Remove batch dimension: (1, T, D) -> (T, D)
        if single_arm.dim() == 3:
            single_arm = single_arm.squeeze(0)
        if gripper.dim() == 3:
            gripper = gripper.squeeze(0)

        if gripper.dim() == 1:
            gripper = gripper.unsqueeze(-1)

        return torch.cat([single_arm, gripper], dim=-1)
