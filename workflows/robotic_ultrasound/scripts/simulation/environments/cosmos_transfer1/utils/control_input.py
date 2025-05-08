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

import random
from functools import partial
from typing import Any, Optional

import cv2
import matplotlib.colors as mcolors
import numpy as np
import pycocotools.mask
import torch
import torchvision.transforms.functional as transforms_F

from cosmos_transfer1.diffusion.config.transfer.blurs import (
    AnisotropicDiffusionConfig,
    BilateralFilterConfig,
    BlurAugmentorConfig,
    GaussianBlurConfig,
    GuidedFilterConfig,
    LaplacianOfGaussianConfig,
    MedianBlurConfig,
)
from cosmos_transfer1.diffusion.datasets.augmentors.guided_filter import FastGuidedFilter
from cosmos_transfer1.diffusion.datasets.augmentors.human_keypoint_utils import (
    coco_wholebody_133_skeleton,
    convert_coco_to_openpose,
    openpose134_skeleton,
)
from cosmos_transfer1.utils import log
from cosmos_transfer1.diffusion.inference.inference_utils import detect_aspect_ratio
from cosmos_transfer1.diffusion.datasets.augmentors.control_input import (
    AddControlInputUpscale,
    AddControlInputDepth,
    AddControlInputKeypoint,
    AddControlInputHDMAP,
    AddControlInputLIDAR,
    AddControlInputBlurDownUp,
    AddControlInputSeg,
    AddControlInputEdge,
    IMAGE_RES_SIZE_INFO,
    VIDEO_RES_SIZE_INFO,
    Augmentor,
)

def resize_frames(frames, is_image, data_dict):
    # Resize the frames to target size before computing control signals to save compute.
    need_reshape = len(frames.shape) < 4
    if need_reshape:  # HWC -> CTHW
        frames = frames.transpose((2, 0, 1))[:, None]
    H, W = frames.shape[2], frames.shape[3]

    aspect_ratio = detect_aspect_ratio((W, H))
    RES_SIZE_INFO = IMAGE_RES_SIZE_INFO if is_image else VIDEO_RES_SIZE_INFO
    new_W, new_H = RES_SIZE_INFO["720"][aspect_ratio]
    # new_W, new_H = 512, 512
    scaling_ratio = min((new_W / W), (new_H / H))
    if scaling_ratio < 1:
        W, H = int(scaling_ratio * W + 0.5), int(scaling_ratio * H + 0.5)
        frames = [
            cv2.resize(_image_np, (W, H), interpolation=cv2.INTER_AREA) for _image_np in frames.transpose((1, 2, 3, 0))
        ]
        frames = np.stack(frames).transpose((3, 0, 1, 2))
    if need_reshape:  # CTHW -> HWC
        frames = frames[:, 0].transpose((1, 2, 0))
    return frames

# x
class AddControlInputBlurDownUpI4H(AddControlInputBlurDownUp,Augmentor):
    """
    Main class for adding blurred input to the data dictionary.
    self.output_keys[0] indicates the types of blur added to the input.
    For example, control_input_gaussian_guided indicates that both Gaussian and Guided filters are applied
    """

    def __init__(
        self,
        input_keys: list,  # [key_load, key_img]
        output_keys: Optional[list] = [
            "control_input_gaussian_guided_bilateral_median_log"
        ],  # eg ["control_input_gaussian_guided"]
        args: Optional[dict] = None,  # not used
        use_random: bool = True,  # whether to use random parameters
        blur_config: BlurAugmentorConfig = BlurAugmentorConfig(),
        downup_preset: str | int = "medium",  # preset strength for downup factor
        min_downup_factor: int = 4,  # minimum downup factor
        max_downup_factor: int = 16,  # maximum downup factor
        downsize_before_blur: bool = False,  # whether to downsize before applying blur and then upsize or downup after blur
    ) -> None:
        super().__init__(
            input_keys,
            output_keys,
            args,
            use_random,
            blur_config,
            downup_preset,
            min_downup_factor,
            max_downup_factor,
            downsize_before_blur
        )

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_vis" in data_dict:
            # already processed
            return data_dict
        key_img = self.input_keys[1]
        key_out = self.output_keys[0]
        frames, is_image = self._load_frame(data_dict)

        # Resize the frames to target size before blurring.
        frames = resize_frames(frames, is_image, data_dict)
        H, W = frames.shape[2], frames.shape[3]

        if self.use_random:
            scale_factor = random.randint(self.min_downup_factor, self.max_downup_factor + 1)
        else:
            scale_factor = self.downup_preset
        if self.downsize_before_blur:
            frames = [
                cv2.resize(_image_np, (W // scale_factor, H // scale_factor), interpolation=cv2.INTER_AREA)
                for _image_np in frames.transpose((1, 2, 3, 0))
            ]
            frames = np.stack(frames).transpose((3, 0, 1, 2))
        frames = self.blur(frames)
        if self.downsize_before_blur:
            frames = [
                cv2.resize(_image_np, (W, H), interpolation=cv2.INTER_LINEAR)
                for _image_np in frames.transpose((1, 2, 3, 0))
            ]
            frames = np.stack(frames).transpose((3, 0, 1, 2))
        if is_image:
            frames = frames[:, 0]
        # turn into tensor
        controlnet_img = torch.from_numpy(frames)
        if not self.downsize_before_blur:
            # Resize image
            controlnet_img = transforms_F.resize(
                controlnet_img,
                size=(int(H / scale_factor), int(W / scale_factor)),
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
            controlnet_img = transforms_F.resize(
                controlnet_img,
                size=(H, W),
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
        data_dict[key_out] = controlnet_img
        return data_dict

#x
class AddControlInputEdgeI4H(AddControlInputEdge,Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_edge"],
        args: Optional[dict] = None,
        use_random: bool = True,
        preset_canny_threshold="medium",
        **kwargs,
    ) -> None:
        super().__init__(
            input_keys,
            output_keys,
            args,
            use_random,
            preset_canny_threshold,
            **kwargs
        )

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_edge" in data_dict:
            # already processed
            return data_dict
        key_img = self.input_keys[1]
        key_out = self.output_keys[0]
        frames = data_dict[key_img]
        # Get lower and upper threshold for canny edge detection.
        if self.use_random:  # always on for training, always off for inference
            t_lower = np.random.randint(20, 100)  # Get a random lower thre within [0, 255]
            t_diff = np.random.randint(50, 150)  # Get a random diff between lower and upper
            t_upper = min(255, t_lower + t_diff)  # The upper thre is lower added by the diff
        else:
            if self.preset_strength == "none" or self.preset_strength == "very_low":
                t_lower, t_upper = 20, 50
            elif self.preset_strength == "low":
                t_lower, t_upper = 50, 100
            elif self.preset_strength == "medium":
                t_lower, t_upper = 100, 200
            elif self.preset_strength == "high":
                t_lower, t_upper = 200, 300
            elif self.preset_strength == "very_high":
                t_lower, t_upper = 300, 400
            else:
                raise ValueError(f"Preset {self.preset_strength} not recognized.")
        frames = np.array(frames)
        is_image = len(frames.shape) < 4

        # Resize the frames to target size before computing canny edges.
        frames = resize_frames(frames, is_image, data_dict)

        # Compute the canny edge map by the two thresholds.
        if is_image:
            edge_maps = cv2.Canny(frames, t_lower, t_upper)[None, None]
        else:
            edge_maps = [cv2.Canny(img, t_lower, t_upper) for img in frames.transpose((1, 2, 3, 0))]
            edge_maps = np.stack(edge_maps)[None]
        edge_maps = torch.from_numpy(edge_maps).expand(3, -1, -1, -1)
        if is_image:
            edge_maps = edge_maps[:, 0]
        data_dict[key_out] = edge_maps
        return data_dict

class AddControlInput(Augmentor):
    """
    For backward compatibility. The previously trained models use legacy_process
    """

    def __init__(
        self,
        input_keys: list,
        output_keys=["control_input_vis"],
        args=None,
        blur_config: BlurAugmentorConfig = BlurAugmentorConfig(),
        use_random=True,
        preset_blur_strength="medium",
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)

        self.process = AddControlInputBlurDownUpI4H(
            input_keys,
            output_keys,
            args,
            blur_config=blur_config,
            downup_preset=preset_blur_strength,  # preset strength for downup factor
            use_random=use_random,
        )

    def __call__(self, data_dict: dict) -> dict:
        return self.process(data_dict)


class AddControlInputComb(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = None,
        blur_config: BlurAugmentorConfig = None,
        args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        assert "comb" in args
        self.comb = {}
        for class_name in args["comb"]:
            if class_name in [AddControlInput, AddControlInputBlurDownUpI4H]:
                aug = class_name(input_keys=input_keys, args=args, blur_config=blur_config, **kwargs)
            else:
                aug = class_name(input_keys=input_keys, args=args, **kwargs)

            key = aug.output_keys[0]
            self.comb[key] = aug

    def __call__(self, data_dict: dict) -> dict:
        all_comb = []
        for k, v in self.comb.items():
            data_dict = v(data_dict)
            all_comb.append(data_dict.pop(k))
            if all_comb[-1].dim() == 4:
                all_comb[-1] = all_comb[-1].squeeze(1)
        all_comb = torch.cat(all_comb, dim=0)
        data_dict[self.output_keys[0]] = all_comb
        return data_dict

#x
def get_augmentor_for_eval(
    input_key: str,
    output_key: str,
    blur_config: BlurAugmentorConfig = BlurAugmentorConfig(),
    preset_blur_strength: str = "medium",
    preset_canny_threshold: str = "medium",
    blur_type: str = "gaussian,guided,bilateral,median,log,anisotropic",  # do we still need this value?
) -> AddControlInputComb:
    comb = []
    output_keys = output_key.replace("control_input_", "").split("_")
    for key in output_keys:
        if "edge" in key:
            comb.append(partial(AddControlInputEdgeI4H, output_keys=["control_input_edge"]))
        elif "upscale" in key:
            comb.append(partial(AddControlInputUpscale, output_keys=["control_input_upscale"]))
        elif "depth" in key:
            comb.append(partial(AddControlInputDepth, output_keys=["control_input_depth"]))
        elif "seg" in key:
            comb.append(partial(AddControlInputSeg, output_keys=["control_input_seg"]))
        elif "vis" in key:
            comb.append(AddControlInput)
        elif "keypoint" in key:
            comb.append(partial(AddControlInputKeypoint, output_keys=["control_input_keypoint"]))
        elif "hdmap" in key:
            comb.append(partial(AddControlInputHDMAP, output_keys=["control_input_hdmap"]))
        elif "lidar" in key:
            comb.append(partial(AddControlInputLIDAR, output_keys=["control_input_lidar"]))
    process = AddControlInputComb(
        input_keys=["", input_key],
        output_keys=[output_key],
        args={"comb": comb},
        blur_config=blur_config,
        use_random=False,
        preset_blur_strength=preset_blur_strength,
        preset_canny_threshold=preset_canny_threshold,
    )
    return process
