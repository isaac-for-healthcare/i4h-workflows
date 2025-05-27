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

from cosmos_transfer1.diffusion.conditioner import VideoExtendCondition
from cosmos_transfer1.diffusion.config.base.conditioner import VideoCondBoolConfig
from cosmos_transfer1.diffusion.diffusion.functional.batch_ops import batch_mul
from cosmos_transfer1.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos_transfer1.diffusion.model.model_v2w import DiffusionV2WModel, VideoDenoisePrediction
from cosmos_transfer1.diffusion.module.parallel import split_inputs_cp
from cosmos_transfer1.utils import log
from megatron.core import parallel_state
from torch import Tensor


class DiffusionV2WModelWithGuidance(DiffusionV2WModel, DiffusionT2WModel):
    def __init__(self, config):
        super().__init__(config)

    def denoise(
        self,
        noise_x: Tensor,
        sigma: Tensor,
        condition: VideoExtendCondition,
        condition_video_augment_sigma_in_inference: float = 0.001,
        seed: int = 1,
        x0_spatial_condtion: dict = None,
    ) -> VideoDenoisePrediction:
        """Denoises input tensor using conditional video generation.

        Args:
            noise_x (Tensor): Noisy input tensor.
            sigma (Tensor): Noise level.
            condition (VideoExtendCondition): Condition for denoising.
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            seed (int): Random seed for reproducibility
            x0_spatial_condtion (dict): Dictionary of spatial condition for x0

        Returns:
            VideoDenoisePrediction containing:
            - x0: Denoised prediction
            - eps: Noise prediction
            - logvar: Log variance of noise prediction
            - xt: Input before c_in multiplication
            - x0_pred_replaced: x0 prediction with condition regions replaced by ground truth
        """

        assert condition.gt_latent is not None, "find None gt_latent in condition, likely didn't call"
        "self.add_condition_video_indicator_and_video_input_mask when preparing the condition"
        "or this is a image batch but condition.data_type is wrong, get {noise_x.shape}"
        gt_latent = condition.gt_latent
        cfg_video_cond_bool: VideoCondBoolConfig = self.config.conditioner.video_cond_bool

        condition_latent = gt_latent

        # Augment the latent with different sigma value, and add the augment_sigma to the condition object if needed
        condition, augment_latent = self.augment_conditional_latent_frames(
            condition, cfg_video_cond_bool, condition_latent, condition_video_augment_sigma_in_inference, sigma, seed
        )
        condition_video_indicator = condition.condition_video_indicator  # [B, 1, T, 1, 1]

        if parallel_state.get_context_parallel_world_size() > 1:
            cp_group = parallel_state.get_context_parallel_group()
            condition_video_indicator = split_inputs_cp(condition_video_indicator, seq_dim=2, cp_group=cp_group)
            augment_latent = split_inputs_cp(augment_latent, seq_dim=2, cp_group=cp_group)
            gt_latent = split_inputs_cp(gt_latent, seq_dim=2, cp_group=cp_group)

        if x0_spatial_condtion is not None:
            if sigma > x0_spatial_condtion["sigma_threshold"]:
                log.info("x0_spatial_condtion is not None, use it to override the new_noise_xt")
                log.info(f"x0_spatial_condtion x0: {x0_spatial_condtion['x0'].shape}")
                log.info(f"x0_spatial_condtion x_sigma_mask: {x0_spatial_condtion['x_sigma_mask'].shape}")

                if x0_spatial_condtion["x_sigma_mask"].shape[1] > 16:
                    x_sigma_mask_bk = x0_spatial_condtion["x_sigma_mask"][:, :16]
                    x_sigma_mask = x0_spatial_condtion["x_sigma_mask"][:, 16:]
                    if sigma > x0_spatial_condtion["sigma_threshold"] * 2.5:
                        # inject the feature from the warped background
                        noise_x = x0_spatial_condtion["x0"] * x_sigma_mask_bk + noise_x * (1 - x_sigma_mask_bk)
                        noise = x0_spatial_condtion["noise"]
                        noise_x = noise_x + (noise * sigma * x_sigma_mask_bk)
                else:
                    x_sigma_mask = x0_spatial_condtion["x_sigma_mask"]
                noise_x = x0_spatial_condtion["x0"] * x_sigma_mask + noise_x * (1 - x_sigma_mask)
                noise = x0_spatial_condtion["noise"]
                noise_x = noise_x + (noise * sigma * x_sigma_mask)
            else:
                log.info("current sigma is too small, no need to use x0_spatial_condtion")

        # Compose the model input with condition region (augment_latent) and generation region (noise_x)
        new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * noise_x

        # Call the abse model
        denoise_pred = super().denoise(new_noise_xt, sigma, condition)

        x0_pred_replaced = condition_video_indicator * gt_latent + (1 - condition_video_indicator) * denoise_pred.x0

        x0_pred = x0_pred_replaced

        return VideoDenoisePrediction(
            x0=x0_pred,
            eps=batch_mul(noise_x - x0_pred, 1.0 / sigma),
            logvar=denoise_pred.logvar,
            xt=new_noise_xt,
            x0_pred_replaced=x0_pred_replaced,
        )
