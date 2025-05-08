# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import einops
import imageio
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from einops import rearrange

from cosmos_transfer1.auxiliary.guardrail.common.io_utils import save_video
from cosmos_transfer1.checkpoints import (
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH,
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
)
from cosmos_transfer1.diffusion.inference.inference_utils import (
    compute_num_latent_frames,
    get_upscale_size,
    read_and_resize_input,
    resize_video,
    split_video_into_patches,
    load_spatial_temporal_weights,
)
from cosmos_transfer1.diffusion.config.transfer.augmentors import BilateralOnlyBlurAugmentorConfig
# from cosmos_transfer1.diffusion.datasets.augmentors.control_input import get_augmentor_for_eval
from simulation.environments.cosmos_transfer1.utils.control_input import get_augmentor_for_eval
from cosmos_transfer1.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos_transfer1.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.config_helper import get_config_module, override
from cosmos_transfer1.utils.io import load_from_fileobj

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase

DEFAULT_AUGMENT_SIGMA = 0.001
NUM_MAX_FRAMES = 5000
VIDEO_RES_SIZE_INFO = {
    "1,1": (960, 960),
    "4,3": (960, 704),
    "3,4": (704, 960),
    "16,9": (1280, 704),
    "9,16": (704, 1280),
}

def get_ctrl_batch(
    model, data_batch, num_video_frames, input_video_path, control_inputs, blur_strength, canny_threshold
):
    """Prepare complete input batch for video generation including latent dimensions.

    Args:
        model: Diffusion model instance

    Returns:
        - data_batch (dict): Complete model input batch
    """
    state_shape = model.state_shape

    H, W = (
        state_shape[-2] * model.tokenizer.spatial_compression_factor,
        state_shape[-1] * model.tokenizer.spatial_compression_factor,
    )

    # Initialize control input dictionary
    control_input_dict = {k: v for k, v in data_batch.items()}
    num_total_frames = NUM_MAX_FRAMES
    if input_video_path:
        input_frames, fps, aspect_ratio = read_and_resize_input(
            input_video_path, num_total_frames=num_total_frames, interpolation=cv2.INTER_AREA
        )
        _, num_total_frames, H, W = input_frames.shape
        control_input_dict["video"] = input_frames.numpy()  # CTHW
        data_batch["input_video"] = input_frames.bfloat16()[None] / 255 * 2 - 1  # BCTHW
    else:
        data_batch["input_video"] = None
    target_w, target_h = W, H

    control_weights = []
    for hint_key, control_info in control_inputs.items():
        if "input_control" in control_info:
            in_file = control_info["input_control"]
            interpolation = cv2.INTER_NEAREST if hint_key == "seg" else cv2.INTER_LINEAR
            log.info(f"reading control input {in_file} for hint {hint_key}")
            control_input_dict[f"control_input_{hint_key}"], fps, aspect_ratio = read_and_resize_input(
                in_file, num_total_frames=num_total_frames, interpolation=interpolation
            )  # CTHW
            num_total_frames = min(num_total_frames, control_input_dict[f"control_input_{hint_key}"].shape[1])
            target_h, target_w = H, W = control_input_dict[f"control_input_{hint_key}"].shape[2:]
        if hint_key == "upscale":
            orig_size = (W, H)
            target_w, target_h = get_upscale_size(orig_size, aspect_ratio, upscale_factor=3)
            input_resized = resize_video(
                input_frames[None].numpy(),
                target_h,
                target_w,
                interpolation=cv2.INTER_LINEAR,
            )  # BCTHW
            control_input_dict["control_input_upscale"] = split_video_into_patches(
                torch.from_numpy(input_resized), H, W
            )
            data_batch["input_video"] = control_input_dict["control_input_upscale"].bfloat16() / 255 * 2 - 1
        control_weights.append(control_info["control_weight"])

    # Trim all control videos and input video to be the same length.
    log.info(f"Making all control and input videos to be length of {num_total_frames} frames.")
    if len(control_inputs) > 1:
        for hint_key in control_inputs.keys():
            cur_key = f"control_input_{hint_key}"
            if cur_key in control_input_dict:
                control_input_dict[cur_key] = control_input_dict[cur_key][:, :num_total_frames]
    if input_video_path:
        control_input_dict["video"] = control_input_dict["video"][:, :num_total_frames]
        data_batch["input_video"] = data_batch["input_video"][:, :, :num_total_frames]

    hint_key = "control_input_" + "_".join(control_inputs.keys())
    add_control_input = get_augmentor_for_eval(
        input_key="video",
        output_key=hint_key,
        preset_blur_strength=blur_strength,
        preset_canny_threshold=canny_threshold,
        blur_config=BilateralOnlyBlurAugmentorConfig[blur_strength],
    )

    if len(control_input_dict):
        control_input = add_control_input(control_input_dict)[hint_key]
        if control_input.ndim == 4:
            control_input = control_input[None]
        control_input = control_input.bfloat16() / 255 * 2 - 1
        control_weights = load_spatial_temporal_weights(
            control_weights, B=1, T=num_video_frames, H=target_h, W=target_w, patch_h=H, patch_w=W
        )
        data_batch["control_weight"] = control_weights

        if len(control_inputs) > 1:  # Multicontrol enabled
            data_batch["hint_key"] = "control_input_multi"
            data_batch["control_input_multi"] = control_input
        else:  # Single-control case
            data_batch["hint_key"] = hint_key
            data_batch[hint_key] = control_input

    data_batch["target_h"], data_batch["target_w"] = target_h // 8, target_w // 8
    data_batch["video"] = torch.zeros((1, 3, 121, H, W), dtype=torch.uint8).cuda()
    data_batch["image_size"] = torch.tensor([[H, W, H, W]] * 1, dtype=torch.bfloat16).cuda()
    data_batch["padding_mask"] = torch.zeros((1, 1, H, W), dtype=torch.bfloat16).cuda()

    return data_batch

def generate_world_from_control(
    model: DiffusionV2WModel,
    state_shape: list[int],
    is_negative_prompt: bool,
    data_batch: dict,
    guidance: float,
    num_steps: int,
    seed: int,
    condition_latent: torch.Tensor,
    num_input_frames: int,
    sigma_max: float,
    x_sigma_max=None,
    x0_spatial_condtion=None,
) -> Tuple[np.array, list, list]:
    """Generate video using a conditioning video/image input.

    Args:
        model (DiffusionV2WModel): The diffusion model instance
        state_shape (list[int]): Shape of the latent state [C,T,H,W]
        is_negative_prompt (bool): Whether negative prompt is provided
        data_batch (dict): Batch containing model inputs including text embeddings
        guidance (float): Classifier-free guidance scale for sampling
        num_steps (int): Number of diffusion sampling steps
        seed (int): Random seed for generation
        condition_latent (torch.Tensor): Latent tensor from conditioning video/image file
        num_input_frames (int): Number of input frames

    Returns:
        np.array: Generated video frames in shape [T,H,W,C], range [0,255]
    """
    assert not model.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i, "not supported"
    augment_sigma = DEFAULT_AUGMENT_SIGMA

    b, c, t, h, w = condition_latent.shape
    if condition_latent.shape[2] < state_shape[1]:
        # Padding condition latent to state shape
        condition_latent = torch.cat(
            [
                condition_latent,
                condition_latent.new_zeros(b, c, state_shape[1] - t, h, w),
            ],
            dim=2,
        ).contiguous()
    num_of_latent_condition = compute_num_latent_frames(model, num_input_frames)

    sample = model.generate_samples_from_batch(
        data_batch,
        guidance=guidance,
        state_shape=[c, t, h, w],
        num_steps=num_steps,
        is_negative_prompt=is_negative_prompt,
        seed=seed,
        condition_latent=condition_latent,
        num_condition_t=num_of_latent_condition,
        condition_video_augment_sigma_in_inference=augment_sigma,
        x_sigma_max=x_sigma_max,
        sigma_max=sigma_max,
        target_h=data_batch["target_h"],
        target_w=data_batch["target_w"],
        patch_h=h,
        patch_w=w,
        x0_spatial_condtion=x0_spatial_condtion,
    )
    return sample