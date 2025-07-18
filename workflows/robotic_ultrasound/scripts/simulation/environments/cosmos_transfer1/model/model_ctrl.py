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

from typing import Callable, Dict, Optional, Tuple, Union

import torch
from cosmos_transfer1.diffusion.inference.inference_utils import merge_patches_into_video, split_video_into_patches
from cosmos_transfer1.diffusion.model.model_ctrl import VideoDiffusionModelWithCtrl
from cosmos_transfer1.diffusion.model.model_t2w import broadcast_condition
from cosmos_transfer1.diffusion.module.parallel import broadcast, cat_outputs_cp, split_inputs_cp
from cosmos_transfer1.utils import log, misc
from einops import rearrange
from megatron.core import parallel_state
from simulation.environments.cosmos_transfer1.model.model_v2w import DiffusionV2WModelWithGuidance
from torch import Tensor


class VideoDiffusionModelWithCtrlAndGuidance(VideoDiffusionModelWithCtrl, DiffusionV2WModelWithGuidance):
    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        seed: int = 1,
        target_h: int = 88,
        target_w: int = 160,
        patch_h: int = 88,
        patch_w: int = 160,
        x0_spatial_condtion: dict = None,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain
        conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising
        operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned
        states.

        Args:
            data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should
                align with the expectations of the `self.conditioner`
            guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative
                to the unconditioned state in the output. Defaults to 1.5.
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W as condition to generate video.
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region
                and config.conditioner.video_cond_bool.condition_location == "first_n"
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            target_h (int): final stitched latent height
            target_w (int): final stitched latent width
            patch_h (int): latent patch height for each network inference
            patch_w (int): latent patch width for each network inference
            x0_spatial_condtion (dict): Dictionary of spatial condition for x0

        Returns:
            Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and
            return x0 prediction.

        The returned function is suitable for use in scenarios where a denoised state is required based on both
        conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        # data_batch should be the one processed by self.get_data_and_condition
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
            # Add conditions for long video generation.

        if condition_latent is None:
            condition_latent = torch.zeros(data_batch["latent_hint"].shape, **self.tensor_kwargs)
            num_condition_t = 0
            condition_video_augment_sigma_in_inference = 1000

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent[:1], condition, num_condition_t
        )

        uncondition.video_cond_bool = True  # Not do cfg on condition frames
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent[:1], uncondition, num_condition_t
        )

        # Add extra conditions for ctrlnet.
        latent_hint = data_batch["latent_hint"]
        hint_key = data_batch["hint_key"]
        setattr(condition, hint_key, latent_hint)
        if "use_none_hint" in data_batch and data_batch["use_none_hint"]:
            setattr(uncondition, hint_key, None)
        else:
            setattr(uncondition, hint_key, latent_hint)

        to_cp = self.net.is_context_parallel_enabled

        if x0_spatial_condtion is not None:
            generator = torch.Generator(device=self.tensor_kwargs["device"])
            generator.manual_seed(seed)
            noise = torch.randn(*x0_spatial_condtion["x0"].shape, **self.tensor_kwargs, generator=generator)
            x0_spatial_condtion["noise"] = noise

        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

            cp_group = parallel_state.get_context_parallel_group()
            latent_hint = getattr(condition, hint_key)
            log.info(f"latent_hint shape_before_cp_split: {latent_hint.shape}")
            seq_dim = 3 if latent_hint.ndim == 6 else 2
            latent_hint = split_inputs_cp(latent_hint, seq_dim=seq_dim, cp_group=cp_group)
            log.info(f"latent_hint shape_after_cp_split: {latent_hint.shape}")
            setattr(condition, hint_key, latent_hint)
            if getattr(uncondition, hint_key) is not None:
                setattr(uncondition, hint_key, latent_hint)

            # the control_weight in cp inference
            control_weight = getattr(condition, "control_weight")
            if isinstance(control_weight, torch.Tensor) and control_weight.ndim == 6:
                # spatial_temporal_weights
                raise NotImplementedError("Spatial temporal weights in cp inference is not implemented yet.")

            if x0_spatial_condtion is not None:
                x0 = x0_spatial_condtion["x0"]
                x_sigma_mask = x0_spatial_condtion["x_sigma_mask"]

                noise = broadcast(noise, to_tp=False, to_cp=to_cp)
                noise = split_inputs_cp(noise, seq_dim=2, cp_group=cp_group)

                log.info(f"x_sigma_mask shape_before_cp_split: {x_sigma_mask.shape}")
                x0 = broadcast(x0, to_tp=False, to_cp=to_cp)
                x_sigma_mask = broadcast(x_sigma_mask, to_tp=False, to_cp=to_cp)
                x0 = split_inputs_cp(x0, seq_dim=2, cp_group=cp_group)
                log.info(f"x_sigma_mask shape_after_cp_split: {x_sigma_mask.shape}")
                x_sigma_mask = split_inputs_cp(x_sigma_mask, seq_dim=2, cp_group=cp_group)

                x0_spatial_condtion = {
                    "x0": x0,
                    "x_sigma_mask": x_sigma_mask,
                    "sigma_threshold": x0_spatial_condtion["sigma_threshold"],
                    "noise": noise,
                }

        setattr(condition, "base_model", self.model.base_model)
        setattr(uncondition, "base_model", self.model.base_model)
        if hasattr(self, "hint_encoders"):
            self.model.net.hint_encoders = self.hint_encoders

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor):
            w, h = target_w, target_h
            n_img_w = (w - 1) // patch_w + 1
            n_img_h = (h - 1) // patch_h + 1

            overlap_size_w = overlap_size_h = 0
            if n_img_w > 1:
                overlap_size_w = (n_img_w * patch_w - w) // (n_img_w - 1)
                assert n_img_w * patch_w - overlap_size_w * (n_img_w - 1) == w
            if n_img_h > 1:
                overlap_size_h = (n_img_h * patch_h - h) // (n_img_h - 1)
                assert n_img_h * patch_h - overlap_size_h * (n_img_h - 1) == h

            batch_images = noise_x
            batch_sigma = sigma
            output = []
            for idx, cur_images in enumerate(batch_images):
                noise_x = cur_images.unsqueeze(0)
                sigma = batch_sigma[idx : idx + 1]
                condition.gt_latent = condition_latent[idx : idx + 1]
                uncondition.gt_latent = condition_latent[idx : idx + 1]
                setattr(condition, hint_key, latent_hint[idx : idx + 1])
                if getattr(uncondition, hint_key) is not None:
                    setattr(uncondition, hint_key, latent_hint[idx : idx + 1])

                cond_x0 = self.denoise(
                    noise_x,
                    sigma,
                    condition,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                    seed=seed,
                    x0_spatial_condtion=x0_spatial_condtion,
                ).x0_pred_replaced
                uncond_x0 = self.denoise(
                    noise_x,
                    sigma,
                    uncondition,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                    seed=seed,
                    x0_spatial_condtion=x0_spatial_condtion,
                ).x0_pred_replaced
                x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
                output.append(x0)
            output = rearrange(torch.stack(output), "(n t) b ... -> (b n t) ...", n=n_img_h, t=n_img_w)
            final_output = merge_patches_into_video(output, overlap_size_h, overlap_size_w, n_img_h, n_img_w)
            final_output = split_video_into_patches(final_output, patch_h, patch_w)
            return final_output

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Union[torch.Tensor, None] = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
        target_h: int = 88,
        target_w: int = 160,
        patch_h: int = 88,
        patch_w: int = 160,
        x0_spatial_condtion: dict = None,
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image
        or video samples. Different from the base model, this function support condition latent as input, it will create
        a different x0_fn if condition latent is given. If this feature is stabilized, we could consider to move this
        function to the base model.

        Args:
            data_batch (dict): Complete model input batch
            guidance (float): Guidance factor for the diffusion process
            seed (int): Random seed for reproducibility
            state_shape (Tuple | None): Shape of the generated samples
            n_sample (int | None): Number of samples to generate
            is_negative_prompt (bool): Whether to use negative prompt
            num_steps (int): Number of steps for the diffusion process
            condition_latent (Optional[torch.Tensor]): latent tensor in shape B,C,T,H,W as condition to generate video
            num_condition_t (Optional[int]): number of condition latent T, if None, will use the whole first half
            condition_video_augment_sigma_in_inference (Optional[float]): sigma for condition video augmentation in
                inference
            x_sigma_max (Optional[torch.Tensor]): latent after applying noise with maximum sigma
            sigma_max (Optional[float]): maximum sigma applied to in_clean_image for image-to-image/video-to-video
            target_h (Optional[int]): final stitched latent height
            target_w (Optional[int]): final stitched latent width
            patch_h (Optional[int]): latent patch height for each network inference
            patch_w (Optional[int]): latent patch width for each network inference
            x0_spatial_condtion (Optional[dict]): Dictionary of spatial condition for x0

        Returns:
            Tensor: Generated samples in shape (n_sample, *state_shape)
        """
        assert patch_h <= target_h and patch_w <= target_w
        if n_sample is None:
            input_key = self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            log.debug(f"Default Video state shape is used. {self.state_shape}")
            state_shape = self.state_shape
        x0_fn = self.get_x0_fn_from_batch(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            seed=seed,
            target_h=target_h,
            target_w=target_w,
            patch_h=patch_h,
            patch_w=patch_w,
            x0_spatial_condtion=x0_spatial_condtion,
        )

        if sigma_max is None:
            sigma_max = self.sde.sigma_max

        if x_sigma_max is None:
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * sigma_max
            )

        if self.net.is_context_parallel_enabled:
            log.info(f"x_sigma_max shape_before_cp_split: {x_sigma_max.shape}")
            x_sigma_max = broadcast(x_sigma_max, to_tp=False, to_cp=True)
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
            log.info(f"x_sigma_max shape_after_cp_split: {x_sigma_max.shape}")

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=sigma_max)

        if self.net.is_context_parallel_enabled:
            log.info(f"samples shape_before_cp_cat: {samples.shape}")
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
            log.info(f"samples shape_after_cp_cat: {samples.shape}")

        return samples
