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

import argparse
import json
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import sys
import tempfile
from io import BytesIO

import cv2
import h5py
import numpy as np
import torch
import torch.distributed as dist
from cosmos_transfer1.checkpoints import BASE_7B_CHECKPOINT_PATH
from cosmos_transfer1.diffusion.inference.inference_utils import load_controlnet_specs, validate_controlnet_specs
from cosmos_transfer1.diffusion.inference.preprocessors import Preprocessors
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.io import save_video
from megatron.core import parallel_state
from simulation.environments.cosmos_transfer1.guided_generation_pipeline import (
    DiffusionControl2WorldGenerationPipelineWithGuidance,
)
from simulation.environments.cosmos_transfer1.utils.inference_utils import (
    concat_videos,
    read_video_or_image_into_frames,
    preprocess_h5_file,
    update_h5_file,
)
from simulation.environments.cosmos_transfer1.utils.warper_transfered_video_gpu import (
    main as warper_transfered_video_main,
)

torch.enable_grad(False)
torch.serialization.add_safe_globals([BytesIO])


def initialize_distributed() -> tuple:
    """
    Initialize distributed training environment.

    Returns:
        tuple: local_rank, global_rank, world_size, and device.
    """
    # Initialize the process group with NCCL backend
    dist.init_process_group(backend="nccl", init_method="env://")

    # Get global rank and world size from the process group
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get the number of GPUs available on the current node
    local_size = torch.cuda.device_count()

    # Calculate local rank (GPU index within the node)
    local_rank = global_rank % local_size

    # Set the device to the correct local GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(f"Initialized rank {global_rank} (local rank {local_rank}) with world size {world_size}")

    return local_rank, global_rank, world_size, device


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control to world generation demo script", conflict_handler="resolve")

    # Add transfer specific arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a "
        "lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget "
        "4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
        help="prompt which the sampled video condition on",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents "
        "a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. "
        "The geometries are very primitive. The images are very pixelated and of poor CG quality. "
        "There are many subtitles in the footage. Overall, the video is unrealistic at all.",
        help="negative prompt which the sampled video condition on",
    )
    parser.add_argument(
        "--input_video_path",
        type=str,
        default="",
        help="Optional input RGB video path",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=1,
        help="Number of conditional frames for long video generation",
        choices=[1, 9, 17],
    )
    parser.add_argument("--sigma_max", type=float, default=80.0, help="sigma_max for partial denoising")
    parser.add_argument(
        "--blur_strength",
        type=str,
        default="medium",
        choices=["very_low", "low", "medium", "high", "very_high"],
        help="blur strength.",
    )
    parser.add_argument(
        "--canny_threshold",
        type=str,
        default="medium",
        choices=["very_low", "low", "medium", "high", "very_high"],
        help="blur strength of canny threshold applied to input. Lower means less blur or more detected edges, "
        "which means higher fidelity to input.",
    )
    parser.add_argument(
        "--controlnet_specs",
        type=str,
        help="Path to JSON file specifying multicontrolnet configurations",
        required=True,
    )
    parser.add_argument(
        "--is_av_sample", action="store_true", help="Whether the model is an driving post-training model"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Base directory containing model checkpoints"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="Cosmos-Tokenize1-CV8x8x8-720p",
        help="Tokenizer weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--video_save_name",
        type=str,
        default="output",
        help="Output filename for generating a single video",
    )
    parser.add_argument(
        "--video_save_folder",
        type=str,
        default="outputs/",
        help="Output folder for generating a batch of videos",
    )
    parser.add_argument(
        "--batch_input_path",
        type=str,
        help="Path to a JSONL file of input prompts for generating a batch of videos",
    )
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion sampling steps")
    parser.add_argument("--guidance", type=float, default=5, help="Classifier-free guidance scale value")
    parser.add_argument("--fps", type=int, default=24, help="FPS of the output video")
    parser.add_argument("--height", type=int, default=704, help="Height of video to sample")
    parser.add_argument("--width", type=int, default=1280, help="Width of video to sample")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--num_gpus", type=int, default=1, choices=[1], help="Number of GPUs used to run inference in parallel."
    )
    parser.add_argument(
        "--offload_diffusion_transformer",
        action="store_true",
        help="Offload DiT after inference",
    )
    parser.add_argument(
        "--offload_text_encoder_model",
        action="store_true",
        help="Offload text encoder model after inference",
    )
    parser.add_argument(
        "--offload_guardrail_models",
        action="store_true",
        help="Offload guardrail models after inference",
    )
    parser.add_argument(
        "--upsample_prompt",
        action="store_true",
        help="Upsample prompt using Pixtral upsampler model",
    )
    parser.add_argument(
        "--offload_prompt_upsampler",
        action="store_true",
        help="Offload prompt upsampler model after inference",
    )
    parser.add_argument(
        "--source_data_dir",
        type=str,
        default=None,
        help="path to source data directory for batch inference.",
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default=None,
        help="path to output data directory for batch inference.",
    )
    parser.add_argument(
        "--save_name_offset",
        type=int,
        default=0,
        help="offset for the video save name.",
    )
    parser.add_argument(
        "--foreground_label",
        type=str,
        default="3,4",
        help="comma separated list of foreground labels to be used for the mask.",
    )
    parser.add_argument(
        "--sigma_threshold",
        type=float,
        default=1.2866e00,
        help="This controls how many guidance steps are performed during generation. Smaller values mean more steps, "
        "larger values mean less steps.",
    )

    cmd_args = parser.parse_args()

    # Load and parse JSON input
    control_inputs, json_args = load_controlnet_specs(cmd_args)

    log.info(f"control_inputs: {json.dumps(control_inputs, indent=4)}")
    log.info(f"args in json: {json.dumps(json_args, indent=4)}")

    # if parameters not set on command line, use the ones from the controlnet_specs
    # if both not set use command line defaults
    for key in json_args:
        if f"--{key}" not in sys.argv:
            setattr(cmd_args, key, json_args[key])

    log.info(f"final args: {json.dumps(vars(cmd_args), indent=4)}")
    return cmd_args, control_inputs


def inference(cfg, pipeline, control_inputs, data, device_rank):
    """Run control-to-world generation demo.

    This function handles the main control-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """

    # override input_control by video files in provided data
    control_inputs["depth"]["input_control"] = data["room_depth_path"]
    control_inputs["seg"]["input_control"] = data["room_seg_path"]
    cfg.input_video_path = data["room_video_path"]
    cfg.x0_mask_path = data["seg_depth_images_npz"]

    source_data_dir = args.source_data_dir
    output_data_dir = args.output_data_dir
    if output_data_dir is None:
        output_data_dir = source_data_dir

    misc.set_random_seed(cfg.seed)
    pipeline.seed = cfg.seed

    preprocessors = Preprocessors()

    # Single prompt case
    if cfg.prompt.endswith(".json"):
        with open(cfg.prompt, "r") as f:
            prompts = json.load(f)
        log.info(f"randomly select one prompt from the json file: {cfg.prompt}")
        prompt_list = []
        for _, v in prompts.items():
            prompt_list.append(v)
        random.shuffle(prompt_list)
        prompt = prompt_list[0]
        if isinstance(prompt, dict):
            top_view_prompt = prompt["top_view_prompt"]
            bottom_view_prompt = prompt["bottom_view_prompt"]

    prompts = [{"prompt": top_view_prompt, "visual_input": cfg.input_video_path}]

    # create video save folder and temporary folder
    cfg.video_save_folder = output_data_dir
    os.makedirs(cfg.video_save_folder, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="cosmos-transfer1_")

    if cfg.x0_mask_path is not None:
        log.info(f"Using x0 mask path: {cfg.x0_mask_path} for guided video generation.")
        x0_spatial_condtion = {
            "x0_mask_path": cfg.x0_mask_path,
            "foreground_label": [int(label) for label in cfg.foreground_label.split(",")],
            "sigma_threshold": cfg.sigma_threshold,
        }

    for _, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        current_video_path = input_dict.get("visual_input", None)

        file_name_index = int(os.path.basename(data["h5_file_path"]).replace(".hdf5", "").split("_")[-1])
        video_save_path = os.path.join(
            cfg.video_save_folder, f"data_{args.save_name_offset + file_name_index}_room_view.mp4"
        )
        prompt_save_path = os.path.join(
            cfg.video_save_folder, f"data_{args.save_name_offset + file_name_index}_room_view.txt"
        )

        if os.path.exists(video_save_path):
            log.info(f"Video already exists: {video_save_path}, skipping room-view generation.")
            video_room_view = read_video_or_image_into_frames(video_save_path, also_return_fps=False)
            continue

        # if control inputs are not provided, run respective preprocessor (for seg and depth)
        preprocessors(current_video_path, current_prompt, control_inputs, temp_dir)

        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            video_path=current_video_path,
            negative_prompt=cfg.negative_prompt,
            control_inputs=control_inputs,
            save_folder=cfg.video_save_folder,
            x0_spatial_condtion=x0_spatial_condtion,
        )
        if generated_output is None:
            log.critical("Guardrail blocked generation.")
            continue
        video_room_view, prompt = generated_output

        # Save video
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        save_video(
            video=video_room_view,
            fps=cfg.fps,
            H=cfg.height,
            W=cfg.width,
            video_save_quality=5,
            video_save_path=video_save_path,
        )
        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            f.write(prompt.encode("utf-8"))
        log.info(f"Saved video to {video_save_path}", rank0_only=False)
        log.info(f"Saved prompt to {prompt_save_path}", rank0_only=False)

    # Warp transferred video
    room_camera_params_path = data["room_camera_para_path"]
    wrist_camera_params_path = data["wrist_camera_para_path"]
    transfered_video_path = video_save_path
    wrist_img_video_path = data["wrist_video_path"]
    seg_depth_images_path = data["seg_depth_images_npz"]
    concat_video_second_view = True

    warped_video_path, roi_masks_path = warper_transfered_video_main(
        room_camera_params_path,
        wrist_camera_params_path,
        transfered_video_path,
        wrist_img_video_path,
        seg_depth_images_path,
        temp_dir,
        device=f"gpu{device_rank}",
        return_concat_video=concat_video_second_view,
        fill_missing_pixels=True,
    )
    torch.cuda.empty_cache()

    # generate second view
    if concat_video_second_view:
        postfix = " The bottom video is facing the same table in the top video without reflection."
        prompts = [{"prompt": bottom_view_prompt + postfix, "visual_input": warped_video_path}]
        concat_videos(
            data["room_depth_path"],
            data["wrist_depth_path"],
            os.path.join(temp_dir, "depth_video_concatv.mp4"),
            "vertical",
        )
        control_inputs["depth"]["input_control"] = os.path.join(temp_dir, "depth_video_concatv.mp4")
        concat_videos(
            data["room_seg_path"],
            data["wrist_seg_path"],
            os.path.join(temp_dir, "seg_mask_video_concatv.mp4"),
            "vertical",
        )
        control_inputs["seg"]["input_control"] = os.path.join(temp_dir, "seg_mask_video_concatv.mp4")
    else:
        prompts = [{"prompt": bottom_view_prompt, "visual_input": warped_video_path}]
        control_inputs["depth"]["input_control"] = data["wrist_depth_path"]
        control_inputs["seg"]["input_control"] = data["wrist_seg_path"]
    cfg.input_video_path = warped_video_path
    cfg.x0_mask_path = roi_masks_path

    if cfg.x0_mask_path is not None:
        log.info(f"Using x0 mask path: {cfg.x0_mask_path} for guided video generation.")
        x0_spatial_condtion = {
            "x0_mask_path": cfg.x0_mask_path,
            "foreground_label": [1],
            "sigma_threshold": cfg.sigma_threshold,
        }
    # update video save folder for second view
    cfg.video_save_folder = output_data_dir

    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        current_video_path = input_dict.get("visual_input", None)

        # if control inputs are not provided, run respective preprocessor (for seg and depth)
        preprocessors(current_video_path, current_prompt, control_inputs, temp_dir)

        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            video_path=current_video_path,
            negative_prompt=cfg.negative_prompt,
            control_inputs=control_inputs,
            save_folder=cfg.video_save_folder,
            x0_spatial_condtion=x0_spatial_condtion,
        )
        if generated_output is None:
            log.critical("Guardrail blocked generation.")
            continue
        video_wrist_view, prompt = generated_output

        video_wrist_view = (
            video_wrist_view[:, video_wrist_view.shape[1] // 2 :, ...] if concat_video_second_view else video_wrist_view
        )

        video_save_path = os.path.join(
            cfg.video_save_folder, f"data_{args.save_name_offset + file_name_index}_wrist_view.mp4"
        )
        prompt_save_path = os.path.join(
            cfg.video_save_folder, f"data_{args.save_name_offset + file_name_index}_wrist_view.txt"
        )

        # Save video
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        save_video(
            video=video_wrist_view,
            fps=cfg.fps,
            H=cfg.height,
            W=cfg.width,
            video_save_quality=5,
            video_save_path=video_save_path,
        )

        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            f.write(prompt.encode("utf-8"))
        log.info(f"Saved video to {video_save_path}", rank0_only=False)
        log.info(f"Saved prompt to {prompt_save_path}", rank0_only=False)
        torch.cuda.empty_cache()

    return video_room_view, video_wrist_view  

if __name__ == "__main__":
    args, control_inputs = parse_arguments()
    random.seed(args.seed)

    local_rank, global_rank, world_size, device = initialize_distributed()
    assert args.num_gpus == 1, "Multi-GPU CP inference is not supported for batch inference."
    # set context_parallel_size to 1 to disable context parallel for batch inference
    parallel_state.initialize_model_parallel(context_parallel_size=1)

    source_data_dir = args.source_data_dir

    h5_file_paths = sorted(glob.glob(os.path.join(source_data_dir, "*.hdf5")))
    log.info(f"total h5 files: {len(h5_file_paths)}")

    control_inputs = validate_controlnet_specs(args, control_inputs)
    # Initialize transfer generation model pipeline
    pipeline = DiffusionControl2WorldGenerationPipelineWithGuidance(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=BASE_7B_CHECKPOINT_PATH,
        offload_network=args.offload_diffusion_transformer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_guardrail_models=args.offload_guardrail_models,
        guidance=args.guidance,
        num_steps=args.num_steps,
        fps=args.fps,
        num_input_frames=args.num_input_frames,
        control_inputs=control_inputs,
        sigma_max=args.sigma_max,
        blur_strength=args.blur_strength,
        canny_threshold=args.canny_threshold,
        upsample_prompt=args.upsample_prompt,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
    )

    for i, h5_file_path in enumerate(h5_file_paths):
        # skip if not the current process
        if i % world_size != global_rank:
            continue
        # check if output videos exist
        source_data_dir = args.source_data_dir
        output_data_dir = args.output_data_dir
        if output_data_dir is None:
            output_data_dir = source_data_dir

        file_name_index = int(os.path.basename(h5_file_path).replace(".hdf5", "").split("_")[-1])
        output_hdf5_path = os.path.join(output_data_dir, f"data_{args.save_name_offset + file_name_index}.hdf5")
        if os.path.exists(output_hdf5_path):
            log.info(f"Output video already exists: {output_hdf5_path}, skipping generation.")
            continue

        data = preprocess_h5_file(h5_file_path)
        log.info(
            f"processing data point {i+1}/{len(h5_file_paths)}: {os.path.basename(h5_file_path)}", rank0_only=False
        )
        args.seed = random.randint(0, 1000000) + global_rank
        video_room_view, video_wrist_view = inference(args, pipeline, control_inputs, data, local_rank)

        update_h5_file(h5_file_path, output_hdf5_path, video_room_view, video_wrist_view)

    if world_size > 1:
        dist.destroy_process_group()
