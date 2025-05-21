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

import os
import shutil
import tempfile
import unittest
from argparse import Namespace

import h5py
import numpy as np
import torch
from cosmos_transfer1.checkpoints import BASE_7B_CHECKPOINT_PATH
from cosmos_transfer1.diffusion.inference.inference_utils import validate_controlnet_specs
from helpers import get_md5_checksum, requires_cosmos_transfer1
from huggingface_hub import snapshot_download
from simulation.environments.cosmos_transfer1.guided_generation_pipeline import (
    DiffusionControl2WorldGenerationPipelineWithGuidance,
)
from simulation.environments.cosmos_transfer1.transfer import inference
from simulation.environments.cosmos_transfer1.utils.inference_utils import preprocess_h5_file, update_h5_file

MD5_CHECKSUM_LOOKUP = {
    "google-t5/t5-11b/pytorch_model.bin": "f890878d8a162e0045a25196e27089a3",
    "google-t5/t5-11b/tf_model.h5": "e081fc8bd5de5a6a9540568241ab8973",
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p/autoencoder.jit": "7f658580d5cf617ee1a1da85b1f51f0d",
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p/decoder.jit": "ff21a63ed817ffdbe4b6841111ec79a8",
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p/encoder.jit": "f5834d03645c379bc0f8ad14b9bc0299",
}


def download_checkpoint(checkpoint: str, output_dir: str) -> None:
    """Download a single checkpoint from HuggingFace Hub."""
    try:
        # Parse the checkpoint path to get repo_id and filename
        checkpoint, revision = checkpoint.split(":") if ":" in checkpoint else (checkpoint, None)
        checkpoint_dir = os.path.join(output_dir, checkpoint)
        if get_md5_checksum(output_dir, checkpoint, MD5_CHECKSUM_LOOKUP):
            print(f"Checkpoint {checkpoint_dir} EXISTS, skipping download... ")
            return
        else:
            print(f"Downloading {checkpoint} to {checkpoint_dir}")
        # Create the output directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Downloading {checkpoint}...")
        # Download the files
        snapshot_download(repo_id=checkpoint, local_dir=checkpoint_dir, revision=revision)
        print(f"Successfully downloaded {checkpoint}")

    except Exception as e:
        print(f"Error downloading {checkpoint}: {str(e)}")


@requires_cosmos_transfer1
class TestCosmosTransfer1Integration(unittest.TestCase):
    """Test cases for the test cosmos-transfer1 integration."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        # Create a temporary h5 file for test data
        self.temp_h5_path = os.path.join(self.temp_dir, "data_0.hdf5")
        self.dummy_video = (np.ones((100, 2, 224, 224, 3)) * 255).astype(np.uint8)
        dummy_seg_images = (np.zeros((100, 2, 224, 224, 3))).astype(np.uint8)
        # create a dummy seg image with 5 different classes
        dummy_seg_images[:, :, :50, :50, :] = np.array([255, 0, 0])
        dummy_seg_images[:, :, 50:100, 50:100, :] = np.array([0, 255, 0])
        dummy_seg_images[:, :, 100:150, 100:150, :] = np.array([0, 0, 255])
        dummy_seg_images[:, :, 150:200, 150:200, :] = np.array([255, 255, 0])
        dummy_camera_intrinsic_matrices = np.array(
            [[110.965096, 0.0, 112.0], [0.0, 176.24133, 112.0], [0.0, 0.0, 1.0]]
        )[None].repeat(100, axis=0)
        dummy_camera_pos = np.array([0.4364283, -0.02340021, 0.45660958])[None].repeat(100, axis=0)
        dummy_camera_quat_w_ros = np.array([-0.11901075, -0.69067025, 0.7013921, 0.1298467])[None].repeat(100, axis=0)
        with h5py.File(self.temp_h5_path, "w") as hf:
            grp = hf.create_group("/data/demo_0/observations/")
            grp.create_dataset("rgb_images", data=self.dummy_video)
            grp.create_dataset("depth_images", data=np.ones((100, 2, 224, 224, 1)).astype(np.float32))
            grp.create_dataset("seg_images", data=dummy_seg_images)
            grp.create_dataset("room_camera_intrinsic_matrices", data=dummy_camera_intrinsic_matrices)
            grp.create_dataset("room_camera_pos", data=dummy_camera_pos)
            grp.create_dataset("room_camera_quat_w_ros", data=dummy_camera_quat_w_ros)
            grp.create_dataset("wrist_camera_intrinsic_matrices", data=dummy_camera_intrinsic_matrices)
            grp.create_dataset("wrist_camera_pos", data=dummy_camera_pos)
            grp.create_dataset("wrist_camera_quat_w_ros", data=dummy_camera_quat_w_ros)

    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_preprocess_h5_file(self):
        """Test preprocess h5 file."""
        data = preprocess_h5_file(self.temp_h5_path)
        for _, value in data.items():
            # check if the processed file exists
            self.assertEqual(os.path.exists(value), True)

    def test_update_h5_file(self):
        """Test updating the h5 file."""
        output_hdf5_path = os.path.join(self.temp_dir, "test_output.hdf5")
        dummy_output_video = np.zeros((100, 224, 224, 3)).astype(np.uint8)
        update_h5_file(self.temp_h5_path, output_hdf5_path, dummy_output_video, dummy_output_video)
        # check if the output file exists
        self.assertEqual(os.path.exists(output_hdf5_path), True)

    def test_cosmos_transfer1_inference(self):
        """Test the inference of cosmos transfer1."""
        # Initialize args
        args = Namespace()
        args.source_data_dir = self.temp_dir
        args.output_data_dir = self.temp_dir
        args.checkpoint_dir = "/tmp/cosmos-transfer1-checkpoints"
        args.sigma_max = 80
        args.input_video_path = "dummy_input_video_path"
        args.seed = 0
        args.offload_diffusion_transformer = True
        args.offload_text_encoder_model = True
        args.offload_guardrail_models = True
        args.guidance = 5
        args.num_steps = 35
        args.fps = 30
        args.num_input_frames = 1
        args.blur_strength = "medium"
        args.canny_threshold = "medium"
        args.upsample_prompt = False
        args.offload_prompt_upsampler = True
        args.x0_mask_path = "dummy_x0_mask_path"
        args.foreground_label = "3,4"
        args.sigma_threshold = 1.2866
        args.height = 224
        args.width = 224
        args.negative_prompt = "dummy_negative_prompt"
        args.video_save_folder = self.temp_dir
        args.save_name_offset = 0
        args.prompt = "dummy_prompt"
        args.concat_video_second_view = True
        args.fill_missing_pixels = False
        args.model_config_file = (
            "workflows/robotic_ultrasound/scripts/simulation/environments/cosmos_transfer1/config/transfer/config.py"
        )
        control_inputs = {
            "edge": {"control_weight": 0.5},
            "depth": {"control_weight": 0.75, "input_control": "placeholder_not_needed.mp4"},
            "seg": {"control_weight": 0.75, "input_control": "placeholder_not_needed.mp4"},
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # download checkpoints
        for checkpoint in ["nvidia/Cosmos-Tokenize1-CV8x8x8-720p", "google-t5/t5-11b"]:
            download_checkpoint(checkpoint, args.checkpoint_dir)
        control_inputs = validate_controlnet_specs(args, control_inputs)
        data = preprocess_h5_file(self.temp_h5_path)
        with torch.cuda.device(0):
            # Initialize pipeline
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
                model_config_file=args.model_config_file,
            )
            # run inference
            video_room_view, video_wrist_view = inference(args, pipeline, control_inputs, data, 0)
        # check if the output video shape is correct, generated videos are 64x64 during testing
        self.assertEqual(video_room_view.shape, (100, 64, 64, 3))
        # check if the second view is half the size of the first view
        self.assertEqual(video_wrist_view.shape, (100, 32, 64, 3))


if __name__ == "__main__":
    unittest.main()
