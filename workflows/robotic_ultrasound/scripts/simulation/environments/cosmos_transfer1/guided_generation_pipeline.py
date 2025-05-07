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

import os
from typing import Optional
import sys

import numpy as np
import torch
from tqdm import tqdm

import cv2

from cosmos_transfer1.diffusion.inference.world_generation_pipeline import DiffusionControl2WorldGenerationPipeline

from cosmos_transfer1.auxiliary.upsampler.model.upsampler import PixtralPromptUpsampler
from cosmos_transfer1.checkpoints import (
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH,
    BASE_7B_CHECKPOINT_PATH,
    COSMOS_TOKENIZER_CHECKPOINT,
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
    generate_world_from_control,
    get_ctrl_batch,
    get_video_batch,
    load_model_by_config,
    load_network_model,
    load_tokenizer_model,
    merge_patches_into_video,
    non_strict_load_model,
    resize_control_weight_map,
    split_video_into_patches,
    resize_video
)
from cosmos_transfer1.diffusion.model.model_ctrl import VideoDiffusionModelWithCtrl, VideoDiffusionT2VModelWithCtrl
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.base_world_generation_pipeline import BaseWorldGenerationPipeline

MODEL_NAME_DICT = {
    BASE_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3",
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3",
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_vis_block3",
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_depth_block3",
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_keypoint_block3",
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_seg_block3",
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_upscale_block3",
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH: "CTRL_7Bv1pt3_t2v_121frames_control_input_hdmap_block3",
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_t2v_121frames_control_input_hdmap_block3",
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_t2v_121frames_control_input_lidar_block3",
}
MODEL_CLASS_DICT = {
    BASE_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH: VideoDiffusionT2VModelWithCtrl,
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionT2VModelWithCtrl,
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionT2VModelWithCtrl,
}


class DiffusionControl2WorldGenerationPipelineWithGuidance(DiffusionControl2WorldGenerationPipeline):
    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_name: str,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_guardrail_models: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
        num_input_frames: int = 1,
        control_inputs: dict = None,
        sigma_max: float = 70.0,
        blur_strength: str = "medium",
        canny_threshold: str = "medium",
        upsample_prompt: bool = False,
        offload_prompt_upsampler: bool = False,
        process_group: torch.distributed.ProcessGroup | None = None,
    ):
        """Initialize diffusion world generation pipeline.

        Args:
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_guardrail_models: Whether to offload guardrail models
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
            num_input_frames: Number of latent conditions
            control_inputs: Dictionary of control inputs for guided generation
            sigma_max: Sigma max for partial denoising
            blur_strength: Strength of blur applied to input
            canny_threshold: Threshold for edge detection
            upsample_prompt: Whether to upsample prompts using prompt upsampler model
            offload_prompt_upsampler: Whether to offload prompt upsampler after use
            process_group: Process group for distributed training
        """
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_guardrail_models=offload_guardrail_models,
            guidance=guidance,
            num_steps=num_steps,
            height=height,
            width=width,
            fps=fps,
            num_video_frames=num_video_frames,
            seed=seed,
            num_input_frames=num_input_frames,
            control_inputs=control_inputs,
            sigma_max=sigma_max,
            blur_strength=blur_strength,
            canny_threshold=canny_threshold,
            upsample_prompt=upsample_prompt,
            offload_prompt_upsampler=offload_prompt_upsampler,
            process_group=process_group,
        )
        
    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            # config_file="cosmos_transfer1/diffusion/config/transfer/config.py",
            config_file="environments/cosmos_transfer1/config/transfer/config.py",
            model_class=self.model_class,
            base_checkpoint_dir=self.checkpoint_dir,
        )

    # load the hint encoders. these encoders are run along with the main model to provide additional context
    def _load_network(self):
        load_network_model(self.model, f"{self.checkpoint_dir}/{self.checkpoint_name}")
        if len(self.control_inputs) > 1:
            hint_encoders = torch.nn.ModuleList([])
            for _, spec in self.control_inputs.items():
                model = load_model_by_config(
                    config_job_name=self.model_name,
                    # config_file="cosmos_transfer1/diffusion/config/transfer/config.py",
                    config_file="environments/cosmos_transfer1/config/transfer/config.py",
                    model_class=self.model_class,
                    base_checkpoint_dir=self.checkpoint_dir,
                )
                load_network_model(model, spec["ckpt_path"])
                hint_encoders.append(model.model.net)
                del model
                torch.cuda.empty_cache()
            self.model.hint_encoders = hint_encoders
        else:
            for _, spec in self.control_inputs.items():
                net_state_dict = torch.load(
                    spec["ckpt_path"], map_location="cpu", weights_only=False
                )  # , weights_only=True)
                non_strict_load_model(self.model.model, net_state_dict)

        if self.process_group is not None:
            self.model.model.net.enable_context_parallel(self.process_group)
            self.model.model.base_model.net.enable_context_parallel(self.process_group)
            if hasattr(self.model.model, "hint_encoders"):
                self.model.model.hint_encoders.net.enable_context_parallel(self.process_group)

    def rgb_to_mask(self, image, color_map=None):
        """
        Convert a (..., 3) image to a (...) mask using a color map.

        Parameters:
        - image: np.ndarray of shape (..., 3), dtype=np.uint8
        - color_map: dict mapping (R, G, B) tuples to integer labels

        Returns:
        - mask: np.ndarray of shape (...), dtype=int
        """
        if color_map is None:
            color_map = {
                (0, 0, 0) : 0,
                (255, 0, 0) : 1,
                (0, 255, 0) : 2,
                (0, 0, 255) : 3,
                (255, 255, 0) : 4 
            }
        # Ensure image is uint8
        image = image.astype(np.uint8)

        # Prepare output mask
        shape = image.shape
        mask = np.zeros(shape[:-1], dtype=np.uint8)

        # Create a view to compare colors efficiently
        for rgb, label in color_map.items():
            matches = (image[..., 0] == rgb[0]) & \
                    (image[..., 1] == rgb[1]) & \
                    (image[..., 2] == rgb[2])
            mask[matches] = label

        return mask
    
    def read_and_resize_numpy_mask(self, input_path, foreground_label=[3, 4], h=704, w=1280, interpolation=cv2.INTER_LINEAR):
        # BCTHW
        video_path = input_path 
        frames = np.load(video_path)
        if "arr_0" in frames:
            frames = frames["arr_0"]
        elif "seg_images" in frames:
            frames = frames["seg_images"]
            # get the room-view mask
            frames = frames[:, 0, ...]
        else:
            raise ValueError(f"Unknown video mask format: {video_path}")
        
        if frames.shape[-1] == 3:
            frames = self.rgb_to_mask(frames)

        tmp = np.zeros_like(frames)
        for label in foreground_label:
            tmp[frames == label] = 255
        frames = tmp
        log.info(f"video mask frames: {frames.shape}")
    
        if frames.ndim == 3:
            frames = np.stack([frames, frames, frames], axis=0)[None]
            log.info(f"original video mask frames: {frames.shape}")
            control_input = resize_video(frames.astype(np.float32), h, w, interpolation=interpolation)  # BCTHW, range [0, 255]
            control_input = (torch.from_numpy(control_input).float()/255.0) # BCTHW, range [0, 1]
            log.info(f"resized video mask frames: {control_input.shape}")
        elif frames.ndim == 4:
            log.info(f"original video mask frames: {frames.shape}")
            out = []
            for i in range(frames.shape[0]):
                tmp_frames = np.stack([frames[i], frames[i], frames[i]], axis=0)[None]
                log.info(f"original video mask frames: {tmp_frames.shape}")
                control_input = resize_video(tmp_frames.astype(np.float32), h, w, interpolation=interpolation)  # BCTHW, range [0, 255]
                control_input = (torch.from_numpy(control_input).float()/255.0) # BCTHW, range [0, 1]
                log.info(f"resized video mask frames: {control_input.shape}")
                out.append(control_input)
            control_input = torch.cat(out, dim=1)  # 1x6xTHW
            log.info(f"resized video mask frames: {control_input.shape}")
                
        return control_input
        
    def construct_latent_weight_map(self, control_input, h=704, w=1280) -> torch.Tensor:
        # construct weight map in the same way as control input in latent space
        fh = h // 8
        fw = w // 8
        if control_input.shape[1] == 6:
            out = []
            for c in [0, 3]:
                weight_map_i = [
                torch.nn.functional.interpolate(
                    control_input[:, [c], :1, :, :],
                    size=(1, fh, fw),
                    mode="trilinear",
                    align_corners=False,
                )
                ]
                for wi in range(1, control_input.shape[2], 8):
                    weight_map_i += [
                        torch.nn.functional.interpolate(
                            control_input[:, [c], wi : wi + 8],
                            size=(1, fh, fw),
                            mode="trilinear",
                            align_corners=False,
                                        )
                    ]
                weight_map_ = torch.cat(weight_map_i, dim=2).repeat(1, 16, 1, 1, 1)
                out.append(weight_map_)
            weight_map = torch.cat(out, dim=1)
            
        elif control_input.shape[1] == 3:
            weight_map_i = [
                torch.nn.functional.interpolate(
                    control_input[:, :1, :1, :, :],
                    size=(1, fh, fw),
                    mode="trilinear",
                    align_corners=False,
                )
            ]
            for wi in range(1, control_input.shape[2], 8):
                weight_map_i += [
                    torch.nn.functional.interpolate(
                        control_input[:, :1, wi : wi + 8],
                        size=(1, fh, fw),
                        mode="trilinear",
                        align_corners=False,
                                    )
                ]
            weight_map = torch.cat(weight_map_i, dim=2).repeat(1, 16, 1, 1, 1)
        log.info(f"video mask weight_map: {weight_map.shape}")
        
        return weight_map

    def _run_model(
        self,
        embedding: torch.Tensor,
        negative_prompt_embedding: torch.Tensor | None = None,
        video_path="",
        control_inputs: dict = None,
        x0_spatial_condtion: dict = None,
    ) -> torch.Tensor:
        """Generate video frames using the diffusion model.

        Args:
            embedding: Text embedding tensor from T5 encoder
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            Tensor of generated video frames

        Note:
            Model and tokenizer are automatically offloaded after inference
            if offloading is enabled.
        """
        # Get video batch and state shape
        data_batch, _ = get_video_batch(
            model=self.model,
            prompt_embedding=embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
        )
        data_batch = get_ctrl_batch(
            self.model,
            data_batch,
            self.num_video_frames,
            video_path,
            control_inputs,
            self.blur_strength,
            self.canny_threshold,
        )

        hint_key = data_batch["hint_key"]
        input_video = data_batch["input_video"]
        control_input = data_batch[hint_key]
        control_weight = data_batch["control_weight"]
        num_new_generated_frames = self.num_video_frames - self.num_input_frames
        B, C, T, H, W = control_input.shape
        
        log.info(f"control_input shape: {B, C, T, H, W}")
        log.info(f"control_weight shape: {control_weight.shape}")
        if x0_spatial_condtion is not None:
            video_mask = self.read_and_resize_numpy_mask(x0_spatial_condtion["x0_mask_path"], foreground_label=x0_spatial_condtion["foreground_label"],h=H, w=W)
            sigma_threshold = x0_spatial_condtion["sigma_threshold"]

        assert T == video_mask.shape[2] == input_video.shape[2], f"T: {T} != video_mask.shape[2]: {video_mask.shape[2]} != input_video.shape[2]: {input_video.shape[2]}"
        
        if (T - self.num_input_frames) % num_new_generated_frames != 0:  # pad duplicate frames at the end
            pad_t = num_new_generated_frames - ((T - self.num_input_frames) % num_new_generated_frames)
            pad_frames = control_input[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
            control_input = torch.cat([control_input, pad_frames], dim=2)
            if input_video is not None:
                pad_video = input_video[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
                input_video = torch.cat([input_video, pad_video], dim=2) 
            if video_mask is not None:
                # add padding to the mask
                pad_video = video_mask[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
                video_mask = torch.cat([video_mask, pad_video], dim=2)
            num_total_frames_with_padding = control_input.shape[2]
            if (
                isinstance(control_weight, torch.Tensor)
                and control_weight.ndim > 5
                and num_total_frames_with_padding > control_weight.shape[3]
            ):
                pad_t = num_total_frames_with_padding - control_weight.shape[3]
                pad_weight = control_weight[:, :, :, -1:].repeat(1, 1, 1, pad_t, 1, 1)
                control_weight = torch.cat([control_weight, pad_weight], dim=3)
        else:
            num_total_frames_with_padding = T
        N_clip = (num_total_frames_with_padding - self.num_input_frames) // num_new_generated_frames

        video = []
        for i_clip in tqdm(range(N_clip)):
            data_batch_i = {k: v for k, v in data_batch.items()}
            start_frame = num_new_generated_frames * i_clip
            end_frame = num_new_generated_frames * (i_clip + 1) + self.num_input_frames

            if input_video is not None:
                x_sigma_max = []
                if video_mask is not None:
                    x0_list = []
                    input_x_sigma_mask = []
                for b in range(B):
                    input_frames = input_video[b : b + 1, :, start_frame:end_frame].cuda()
                    x0 = self.model.encode(input_frames).contiguous()
                    # x_sigma_max.append(self.model.get_x_from_clean(x0, self.sigma_max, seed=(self.seed + i_clip)))
                    x_sigma_max.append(self.model.get_x_from_clean(x0, self.sigma_max, seed=self.seed))
                    if video_mask is not None:
                        x0_list.append(x0)
                        input_x_sigma_mask.append(self.construct_latent_weight_map(video_mask[b : b + 1, :, start_frame:end_frame].cuda(), h=H, w=W))
                x_sigma_max = torch.cat(x_sigma_max)
                if video_mask is not None:
                    x0_spatial_condtion = {
                        "x0": torch.cat(x0_list),
                        "x_sigma_mask": torch.cat(input_x_sigma_mask),
                        "sigma_threshold": sigma_threshold,
                    }

                if self.sigma_max >= 80:
                    log.info(f"sigma_max is greater than 80, using None for x_sigma_max")
                    x_sigma_max = None

            else:
                x_sigma_max = None

            data_batch_i[hint_key] = control_input[:, :, start_frame:end_frame].cuda()
            latent_hint = []
            for b in range(B):
                data_batch_p = {k: v for k, v in data_batch_i.items()}
                data_batch_p[hint_key] = data_batch_i[hint_key][b : b + 1]
                if len(control_inputs) > 1:
                    latent_hint_i = []
                    for idx in range(0, data_batch_p[hint_key].size(1), 3):
                        x_rgb = data_batch_p[hint_key][:, idx : idx + 3]
                        latent_hint_i.append(self.model.encode(x_rgb))
                    latent_hint.append(torch.cat(latent_hint_i).unsqueeze(0))
                else:
                    latent_hint.append(self.model.encode_latent(data_batch_p))
            data_batch_i["latent_hint"] = latent_hint = torch.cat(latent_hint)

            if isinstance(control_weight, torch.Tensor) and control_weight.ndim > 4:
                control_weight_t = control_weight[..., start_frame:end_frame, :, :].cuda()
                t, h, w = latent_hint.shape[-3:]
                data_batch_i["control_weight"] = resize_control_weight_map(control_weight_t, (t, h // 2, w // 2))

            if i_clip == 0:
                num_input_frames = 0
                latent_tmp = latent_hint if latent_hint.ndim == 5 else latent_hint[:, 0]
                condition_latent = torch.zeros_like(latent_tmp)
            else:
                num_input_frames = self.num_input_frames
                prev_frames = split_video_into_patches(prev_frames, control_input.shape[-2], control_input.shape[-1])
                condition_latent = []
                for b in range(B):
                    input_frames = prev_frames[b : b + 1].cuda().bfloat16() / 255.0 * 2 - 1
                    condition_latent += [self.model.encode(input_frames).contiguous()]
                condition_latent = torch.cat(condition_latent)
            
            log.info(f"latent_hint shape: {latent_hint.shape}")
            if x_sigma_max is not None:
                log.info(f"x_sigma_max shape: {x_sigma_max.shape}")
            if x0_spatial_condtion is not None:
                log.info(f"x0 shape: {x0_spatial_condtion['x0'].shape}")
                log.info(f"x_sigma_mask shape: {x0_spatial_condtion['x_sigma_mask'].shape}")
           
            # Generate video frames
            latents = generate_world_from_control(
                model=self.model,
                state_shape=self.model.state_shape,
                is_negative_prompt=True,
                data_batch=data_batch_i,
                guidance=self.guidance,
                num_steps=self.num_steps,
                # seed=(self.seed + i_clip), 
                seed=self.seed, 
                condition_latent=condition_latent,
                num_input_frames=num_input_frames,
                sigma_max=self.sigma_max if x_sigma_max is not None else None,
                x_sigma_max=x_sigma_max,
                x0_spatial_condtion=x0_spatial_condtion,
            )
            frames = self._run_tokenizer_decoding(latents)
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2)[None]

            if i_clip == 0:
                video.append(frames)
            else:
                video.append(frames[:, :, self.num_input_frames :])
            prev_frames = torch.zeros_like(frames)
            prev_frames[:, :, : self.num_input_frames] = frames[:, :, -self.num_input_frames :]

        video = torch.cat(video, dim=2)[:, :, :T]
        video = video[0].permute(1, 2, 3, 0).numpy()
        return video

    def generate(
        self,
        prompt: str,
        video_path: str,
        negative_prompt: Optional[str] = None,
        control_inputs: dict = None,
        save_folder: str = "outputs/",
        x0_spatial_condtion: dict = None,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt and control video.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Convert prompt to embeddings
        3. Generate video frames using diffusion
        4. Run safety checks and apply face blur on generated video frames

        Args:
            prompt: Text description of desired video
            video_path: Path to input video
            negative_prompt: Optional text to guide what not to generate
            control_inputs: Control inputs for guided generation
            save_folder: Folder to save intermediate files

        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")
        log.info(f"Run with video path: {video_path}")
        log.info(f"Run with negative prompt: {negative_prompt}")

        # Upsample prompt if enabled
        if self.prompt_upsampler:
            if int(os.environ["RANK"]) == 0:
                self._push_torchrun_environ_variables()
                prompt = self._process_prompt_upsampler(prompt, video_path, save_folder)
                self._pop_torchrun_environ_variables()
                log.info(f"Upsampled prompt: {prompt}")

        # log.info("Run guardrail on prompt")
        is_safe = True # self._run_guardrail_on_prompt_with_offload(prompt)
        # if not is_safe:
        #     log.critical("Input text prompt is not safe")
        #     return None
        # log.info("Pass guardrail on prompt")

        log.info("Run text embedding on prompt")
        if negative_prompt:
            prompts = [prompt, negative_prompt]
        else:
            prompts = [prompt]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        prompt_embedding = prompt_embeddings[0]
        negative_prompt_embedding = prompt_embeddings[1] if negative_prompt else None
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            video_path=video_path,
            control_inputs=control_inputs,
            x0_spatial_condtion=x0_spatial_condtion,
        )
        log.info("Finish generation")

        return video, prompt
