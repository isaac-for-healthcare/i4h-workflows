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
import tempfile
from typing import Dict, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
from cosmos_transfer1.diffusion.config.transfer.augmentors import BilateralOnlyBlurAugmentorConfig
from cosmos_transfer1.diffusion.inference.inference_utils import (
    DEFAULT_AUGMENT_SIGMA,
    NUM_MAX_FRAMES,
    VIDEO_RES_SIZE_INFO,
    compute_num_latent_frames,
    detect_aspect_ratio,
    get_upscale_size,
    load_spatial_temporal_weights,
    non_strict_load_model,
    read_video_or_image_into_frames_BCTHW,
    resize_video,
    skip_init_linear,
    split_video_into_patches,
)
from cosmos_transfer1.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos_transfer1.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.io import load_from_fileobj
from simulation.environments.cosmos_transfer1.utils.control_input import get_augmentor_for_eval
from tqdm import tqdm

DEBUG_GENERATION = os.environ.get("DEBUG_GENERATION", "0") == "1"


def load_network_model(model: DiffusionT2WModel, ckpt_path: str):
    """
    Load network model from checkpoint
    Args:
        model: model to load
        ckpt_path: path to checkpoint
    """
    with skip_init_linear():
        model.set_up_model()
    if not DEBUG_GENERATION:
        net_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        non_strict_load_model(model.model, net_state_dict)
    model.cuda()


def preprocess_h5_file(h5_file_path):
    """
    Preprocess h5 file to mp4 videos
    Args:
        h5_file_path: path to h5 file
    Returns:
        data: dict containing the following keys:
            - room_video_path: path to room video
            - room_depth_path: path to room depth video
            - room_seg_path: path to room seg video
            - wrist_video_path: path to wrist video
            - wrist_depth_path: path to wrist depth video
            - wrist_seg_path: path to wrist seg video
            - room_camera_para_path: path to room camera parameters
            - wrist_camera_para_path: path to wrist camera parameters
            - seg_depth_images_npz: path to seg depth images
    """
    # preprocess h5 file to mp4 videos
    mp4_save_dir = tempfile.mkdtemp(prefix="cosmos-transfer1-h5-to-mp4_")
    print(f"preprocessing h5 file to mp4 videos: {mp4_save_dir}")
    with h5py.File(h5_file_path, "r") as f:
        rgb_images = f["data/demo_0/observations/rgb_images"]  # (263, 2, 224, 224, 3)
        depth_images = f["data/demo_0/observations/depth_images"]  # (263, 2, 224, 224, 1)
        seg_images = f["data/demo_0/observations/seg_images"]  # (263, 2, 224, 224, 4) - Only last channel needed
        # Video parameters
        num_frames, num_videos, height, width, _ = seg_images.shape
        fps = 30  # Frames per second
        video_code = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
        # Create video writers
        writers = {
            "rgb": [
                cv2.VideoWriter(f"{mp4_save_dir}/rgb_video_{i}.mp4", video_code, fps, (width, height))
                for i in range(num_videos)
            ],
            "depth": [
                cv2.VideoWriter(f"{mp4_save_dir}/depth_video_{i}.mp4", video_code, fps, (width, height), isColor=False)
                for i in range(num_videos)
            ],
            "seg": [
                cv2.VideoWriter(
                    f"{mp4_save_dir}/seg_mask_video_{i}.mp4", video_code, fps, (width, height), isColor=True
                )
                for i in range(num_videos)
            ],
        }
        for i in range(num_frames):
            for vid_idx in range(num_videos):
                # RGB Video
                rgb_frame = rgb_images[i, vid_idx].astype(np.uint8)[:, :, ::-1]
                writers["rgb"][vid_idx].write(rgb_frame)
                # Depth Video
                depth_frame = depth_images[i, vid_idx, :, :, 0]
                output = 1.0 / (depth_frame + 1e-6)
                depth_min = output.min()
                depth_max = output.max()
                max_val = (2**8) - 1  # Maximum value for uint16
                if depth_max - depth_min > np.finfo("float").eps:
                    out_array = max_val * (output - depth_min) / (depth_max - depth_min)
                else:
                    out_array = np.zeros_like(output)
                formatted = out_array.astype("uint8")
                writers["depth"][vid_idx].write(formatted)
                seg_mask_frame = seg_images[i, vid_idx, :, :, :].astype(np.uint8)
                writers["seg"][vid_idx].write(seg_mask_frame)
        # Release all video writers
        for category in writers:
            for writer in writers[category]:
                writer.release()
        # save camera parameters
        room_camera_intrinsic_matrices = f[
            "data/demo_0/observations/room_camera_intrinsic_matrices"
        ]  # (n_frames, 3, 3)
        room_camera_pos = f["data/demo_0/observations/room_camera_pos"]  # (n_frames, 3)
        room_camera_quat = f["data/demo_0/observations/room_camera_quat_w_ros"]  # (n_frames, 4)
        save_dict = {
            "room_camera_intrinsic_matrices": room_camera_intrinsic_matrices,
            "room_camera_pos": room_camera_pos,
            "room_camera_quat": room_camera_quat,
        }
        room_camera_para_path = f"{mp4_save_dir}/room_camera_para.npz"
        np.savez(room_camera_para_path, **save_dict)
        wrist_camera_intrinsic_matrices = f[
            "data/demo_0/observations/wrist_camera_intrinsic_matrices"
        ]  # (n_frames, 3, 3)
        wrist_camera_pos = f["data/demo_0/observations/wrist_camera_pos"]  # (n_frames, 3)
        wrist_camera_quat = f["data/demo_0/observations/wrist_camera_quat_w_ros"]  # (n_frames, 4)
        save_dict = {
            "wrist_camera_intrinsic_matrices": wrist_camera_intrinsic_matrices,
            "wrist_camera_pos": wrist_camera_pos,
            "wrist_camera_quat": wrist_camera_quat,
        }
        wrist_camera_para_path = f"{mp4_save_dir}/wrist_camera_para.npz"
        np.savez(wrist_camera_para_path, **save_dict)
        save_dict = {
            "depth_images": f["data/demo_0/observations/depth_images"],
            "seg_images": f["data/demo_0/observations/seg_images"],
        }
        seg_depth_images_npz = f"{mp4_save_dir}/seg_depth_images.npz"
        np.savez(seg_depth_images_npz, **save_dict)
        data = {
            "room_video_path": f"{mp4_save_dir}/rgb_video_0.mp4",
            "room_depth_path": f"{mp4_save_dir}/depth_video_0.mp4",
            "room_seg_path": f"{mp4_save_dir}/seg_mask_video_0.mp4",
            "wrist_video_path": f"{mp4_save_dir}/rgb_video_1.mp4",
            "wrist_depth_path": f"{mp4_save_dir}/depth_video_1.mp4",
            "wrist_seg_path": f"{mp4_save_dir}/seg_mask_video_1.mp4",
            "room_camera_para_path": room_camera_para_path,
            "wrist_camera_para_path": wrist_camera_para_path,
            "seg_depth_images_npz": seg_depth_images_npz,
            "h5_file_path": h5_file_path,
        }
    return data


def update_h5_file(source_h5_path, output_hdf5_path, video_room_view, video_wrist_view):
    """
    Update h5 file with generated videos
    """
    with h5py.File(source_h5_path, "r") as f:
        rgb_images = f["data/demo_0/observations/rgb_images"]
        # Create a new HDF5 file
        with h5py.File(output_hdf5_path, "w") as dst:
            # Copy all groups and datasets, except the part we want to replace
            def copy_attributes(name, obj):
                if isinstance(obj, h5py.Group):
                    if name not in dst:
                        dst.create_group(name)
                    # Copy group attributes
                    for key, value in obj.attrs.items():
                        dst[name].attrs[key] = value

                elif isinstance(obj, h5py.Dataset):
                    if name != "data/demo_0/observations/rgb_images":
                        # Copy the dataset
                        dst.create_dataset(name, data=obj[:])
                        # Copy dataset attributes
                        for key, value in obj.attrs.items():
                            dst[name].attrs[key] = value

            # Process all items in the source file
            f.visititems(copy_attributes)

            # Create a new RGB dataset
            num_frames = video_room_view.shape[0]

            # Create a dataset with the same shape as the original
            dst_rgb = dst.create_dataset(
                "data/demo_0/observations/rgb_images", shape=rgb_images.shape, dtype=rgb_images.dtype
            )

            # Copy attributes
            for key, value in rgb_images.attrs.items():
                dst_rgb.attrs[key] = value

            # Replace frames in the dataset
            for i in tqdm(range(num_frames)):
                # Process camera 0 (top view)
                if i < len(video_room_view):
                    # Ensure correct shape
                    frame_rgb0 = video_room_view[i]
                    if frame_rgb0.shape != rgb_images.shape[2:]:
                        frame_rgb0 = cv2.resize(frame_rgb0, (rgb_images.shape[3], rgb_images.shape[2]))
                    dst_rgb[i, 0] = frame_rgb0

                # Process camera 1 (bottom view)
                if i < len(video_wrist_view):
                    # Ensure correct shape
                    frame_rgb1 = video_wrist_view[i]
                    if frame_rgb1.shape != rgb_images.shape[2:]:
                        frame_rgb1 = cv2.resize(frame_rgb1, (rgb_images.shape[3], rgb_images.shape[2]))
                    dst_rgb[i, 1] = frame_rgb1

            # If video frame count is less than original data, copy remaining frames
            if num_frames < rgb_images.shape[0]:
                dst_rgb[num_frames:] = rgb_images[num_frames:]


def concat_videos(video1_path: str, video2_path: str, output_path: str, direction: str = "horizontal"):
    """Concatenate two videos horizontally or vertically
    Args:
        video1_path: path to first video
        video2_path: path to second video
        output_path: path to output video
        direction: direction to concatenate videos
    """
    # Open the video files
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))

    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))

    if direction == "horizontal":
        # Use the minimum height and sum of widths for the output video
        output_height = min(height1, height2)
        output_width = width1 + width2
    else:  # vertical
        # Use the minimum width and sum of heights for the output video
        output_width = min(width1, width2)
        output_height = height1 + height2

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, min(fps1, fps2), (output_width, output_height))

    while True:
        # Read frames from both videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Break if either video is finished
        if not ret1 or not ret2:
            break

        if direction == "horizontal":
            # Resize frames to match the output height
            frame1 = cv2.resize(frame1, (width1, output_height))
            frame2 = cv2.resize(frame2, (width2, output_height))
            # Concatenate frames horizontally
            combined_frame = np.hstack((frame1, frame2))
        else:  # vertical
            # Resize frames to match the output width
            frame1 = cv2.resize(frame1, (output_width, height1))
            frame2 = cv2.resize(frame2, (output_width, height2))
            # Concatenate frames vertically
            combined_frame = np.vstack((frame1, frame2))

        # Write the combined frame
        out.write(combined_frame)

    # Release everything
    cap1.release()
    cap2.release()
    out.release()


def read_video_or_image_into_frames(
    input_path: str,
    input_path_format: str = "mp4",
    also_return_fps: bool = True,
) -> Tuple[np.ndarray, int, float]:
    """Read video or image file and convert to tensor format.

    Args:
        input_path (str): Path to input video/image file
        input_path_format (str): Format of input file (default: "mp4")
        also_return_fps (bool): Whether to return fps along with frames

    Returns:
        numpy.ndarray | tuple: Video tensor in shape [T,H,W,C], optionally with fps if requested
    """
    loaded_data = load_from_fileobj(input_path, format=input_path_format)
    frames, meta_data = loaded_data
    if input_path.endswith(".png") or input_path.endswith(".jpg") or input_path.endswith(".jpeg"):
        frames = np.array(frames[0])  # HWC, [0,255]
        if frames.shape[-1] > 3:  # RGBA, set the transparent to white
            # Separate the RGB and Alpha channels
            rgb_channels = frames[..., :3]
            alpha_channel = frames[..., 3] / 255.0  # Normalize alpha channel to [0, 1]

            # Create a white background
            white_bg = np.ones_like(rgb_channels) * 255  # White background in RGB

            # Blend the RGB channels with the white background based on the alpha channel
            frames = (rgb_channels * alpha_channel[..., None] + white_bg * (1 - alpha_channel[..., None])).astype(
                np.uint8
            )
        frames = [frames]
        fps = 0
    else:
        fps = int(meta_data.get("fps"))

    input_tensor = np.stack(frames, axis=0)
    if also_return_fps:
        return input_tensor, fps
    return input_tensor


def rgb_to_mask(image: np.ndarray, color_map: Optional[Dict[Tuple[int, int, int], int]] = None) -> np.ndarray:
    """
    Convert a (..., 3) image to a (...) mask using a color map.

    Args:
        image: np.ndarray of shape (..., 3), dtype=np.uint8
        color_map: dict mapping (R, G, B) tuples to integer labels

    Returns:
        mask: np.ndarray of shape (...), dtype=int
    """
    if color_map is None:
        color_map = {(0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2, (0, 0, 255): 3, (255, 255, 0): 4}
    # Ensure image is uint8
    image = image.astype(np.uint8)

    # Prepare output mask
    shape = image.shape
    mask = np.zeros(shape[:-1], dtype=np.uint8)

    # Create a view to compare colors efficiently
    for rgb, label in color_map.items():
        matches = (image[..., 0] == rgb[0]) & (image[..., 1] == rgb[1]) & (image[..., 2] == rgb[2])
        mask[matches] = label

    return mask


def read_and_resize_input(
    input_control_path: str, num_total_frames: int, interpolation: int
) -> Tuple[np.ndarray, int, float]:
    """Read and resize input control video/image.

    Args:
        input_control_path: path to input control video/image
        num_total_frames: number of total frames
        interpolation: interpolation method

    Returns:
        control_input: control input tensor
        fps: fps of the control input
        aspect_ratio: aspect ratio of the control input
    """
    control_input, fps = read_video_or_image_into_frames_BCTHW(
        input_control_path,
        normalize=False,  # s.t. output range is [0, 255]
        max_frames=num_total_frames,
        also_return_fps=True,
    )  # BCTHW
    aspect_ratio = detect_aspect_ratio((control_input.shape[-1], control_input.shape[-2]))
    w, h = VIDEO_RES_SIZE_INFO[aspect_ratio]
    if DEBUG_GENERATION:
        w, h = (128, 128)
        log.info(f"Running in debug mode, resizing control input to {h}x{w} and ignore aspect ratio.")
    control_input = resize_video(control_input, h, w, interpolation=interpolation)  # BCTHW, range [0, 255]
    control_input = torch.from_numpy(control_input[0])  # CTHW, range [0, 255]
    return control_input, fps, aspect_ratio


def get_ctrl_batch(
    model, data_batch, num_video_frames, input_video_path, control_inputs, blur_strength, canny_threshold
):
    """Prepare complete input batch for video generation including latent dimensions.

    Args:
        model: Diffusion model instance
        data_batch: Complete model input batch
        num_video_frames: number of video frames
        input_video_path: path to input video
        control_inputs: control inputs
        blur_strength: blur strength
        canny_threshold: canny threshold

    Returns:
        data_batch (dict): Complete model input batch
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
    x_sigma_max: Optional[torch.Tensor] = None,
    x0_spatial_condtion: Optional[dict] = None,
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
        sigma_max (float): Maximum sigma value for the diffusion process
        x_sigma_max (Optional[torch.Tensor]): latent after applying noise with maximum sigma
        x0_spatial_condtion (Optional[dict]): Dictionary of spatial condition for x0

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
