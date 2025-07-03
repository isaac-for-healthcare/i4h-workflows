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

# ===== CRITICAL: Set environment variables BEFORE any imports =====
# This prevents async checkpointing and threading issues that cause SIGABRT
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["PYTHONFAULTHANDLER"] = "1"
os.environ["WANDB_MODE"] = "offline"

import shutil
import threading
import time
import unittest

import h5py
import numpy as np
import torch
from openpi import train
from policy_runner.pi0.config import get_config
from policy_runner.pi0.utils import compute_normalization_stats
from training.convert_hdf5_to_lerobot import Pi0FeatureDict, create_lerobot_dataset
from training.convert_hdf5_to_lerobot import main as convert_hdf5_to_lerobot


class TestBase(unittest.TestCase):
    """Base class for training tests with common setup and teardown methods."""

    TEST_REPO_ID = "i4h/test_data"

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Determine cache location
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/lerobot")
        self.test_data_dir = os.path.join(self.cache_dir, self.TEST_REPO_ID)

        # Setup temporary directories
        self.current_dir = os.getcwd()
        self.tmp_assets_dir = os.path.join(self.current_dir, "assets")
        self.tmp_checkpoints_dir = os.path.join(self.current_dir, "checkpoints")
        self.tmp_wandb_dir = os.path.join(self.current_dir, "wandb")
        self.hdf5_data_dir = os.path.join(self.current_dir, "test_data")

        # Create the test_data directory if it doesn't exist
        os.makedirs(self.hdf5_data_dir, exist_ok=True)

        # Create a basic config for testing
        self.test_config = get_config(
            name="robotic_ultrasound_lora", repo_id=self.TEST_REPO_ID, exp_name="test_experiment"
        )
        
        # ===== CRITICAL FIX: Disable async checkpointing for tests =====
        # This prevents the C++ thread crash that causes SIGABRT
        if hasattr(self.test_config, 'enable_async_checkpointing'):
            self.test_config.enable_async_checkpointing = False
        
        # Try to disable at the checkpoint manager level if possible
        if hasattr(self.test_config, 'checkpoint_manager_options'):
            if hasattr(self.test_config.checkpoint_manager_options, 'enable_async_checkpointing'):
                self.test_config.checkpoint_manager_options.enable_async_checkpointing = False
        
        self.test_prompt = "test_prompt"

        # Flag to allow for cleanup
        self.should_cleanup = False

    def tearDown(self):
        """Clean up after each test method."""
        # ===== CRITICAL: Join training threads first =====
        if hasattr(self, 'training_thread') and self.training_thread.is_alive():
            self.training_thread.join(timeout=10)
            if self.training_thread.is_alive():
                print("Warning: Training thread did not stop cleanly")

        # ===== CRITICAL: Shutdown JAX runtime to prevent C++ crashes =====
        if JAX_AVAILABLE:
            try:
                jax.clear_backends()  # Clear all JAX backends
                # Note: jax.shutdown() might not exist in all versions
                if hasattr(jax, 'shutdown'):
                    jax.shutdown()
            except Exception as e:
                print(f"Warning: JAX cleanup failed: {e}")

        if self.should_cleanup:
            # Remove test data directory
            if os.path.exists(self.test_data_dir):
                shutil.rmtree(self.test_data_dir)

            # Remove wandb directory if it exists
            if os.path.exists(self.tmp_wandb_dir):
                shutil.rmtree(self.tmp_wandb_dir)

            # Remove any checkpoints in current directory
            if os.path.exists(self.tmp_checkpoints_dir):
                shutil.rmtree(self.tmp_checkpoints_dir)

            # Remove any assets in current directory
            if os.path.exists(self.tmp_assets_dir):
                shutil.rmtree(self.tmp_assets_dir)

        # Always clean up the test_data directory
        if os.path.exists(self.hdf5_data_dir):
            shutil.rmtree(self.hdf5_data_dir)


class TestConvertHdf5ToLeRobot(TestBase):
    """Test the conversion of HDF5 data to LeRobot format."""

    def setUp(self):
        """Set up test fixtures, including creating a dummy HDF5 file."""
        super().setUp()

        # Create a dummy HDF5 file with the expected structure
        self._create_dummy_hdf5_file()
        self.feature_builder = Pi0FeatureDict()

    def _create_dummy_hdf5_file(self):
        """Create a dummy HDF5 file with 25 steps for testing."""
        num_steps = 50

        # Create the test_data directory if it doesn't exist
        os.makedirs(self.hdf5_data_dir, exist_ok=True)

        # Create a dummy HDF5 file
        hdf5_path = os.path.join(self.hdf5_data_dir, "data_0.hdf5")

        with h5py.File(hdf5_path, "w") as f:
            # Create the root group
            root_name = "data/demo_0"
            root_group = f.create_group(root_name)

            # Create action dataset (num_steps, 6)
            root_group.create_dataset("action", data=np.random.rand(num_steps, 6).astype(np.float32))

            # Create abs_joint_pos dataset (num_steps, 7)
            root_group.create_dataset("abs_joint_pos", data=np.random.rand(num_steps, 7).astype(np.float32))

            # Create observations group
            obs_group = root_group.create_group("observations")

            # Create RGB dataset (num_steps, 2, height, width, 3)
            # Using small 32x32 images to keep the file size small
            rgb_data = np.random.randint(0, 256, size=(num_steps, 2, 32, 32, 3), dtype=np.uint8)
            obs_group.create_dataset("rgb_images", data=rgb_data)
            obs_group.create_dataset("depth_images", data=rgb_data)
            obs_group.create_dataset("seg_images", data=rgb_data)

    def test_convert_hdf5_to_lerobot(self):
        """Test that HDF5 data can be converted to LeRobot format successfully."""
        convert_hdf5_to_lerobot(self.hdf5_data_dir, self.TEST_REPO_ID, self.test_prompt, self.feature_builder)
        meta_data_dir = os.path.join(self.test_data_dir, "meta")
        data_dir = os.path.join(self.test_data_dir, "data")
        self.assertTrue(os.path.exists(meta_data_dir), f"Meta data directory not created at {meta_data_dir}")
        self.assertTrue(os.path.exists(data_dir), f"Data directory not created at {data_dir}")

    def test_convert_hdf5_to_lerobot_with_seg(self):
        """Test that HDF5 data can be converted to LeRobot format successfully."""
        convert_hdf5_to_lerobot(
            self.hdf5_data_dir, self.TEST_REPO_ID, self.test_prompt, self.feature_builder, include_seg=True
        )
        meta_data_dir = os.path.join(self.test_data_dir, "meta")
        data_dir = os.path.join(self.test_data_dir, "data")

        self.assertTrue(os.path.exists(meta_data_dir), f"Meta data directory not created at {meta_data_dir}")
        self.assertTrue(os.path.exists(data_dir), f"Data directory not created at {data_dir}")

    def test_convert_hdf5_to_lerobot_data_dir_error_handling(self):
        """Test that data directory error can be handled."""
        # check get expected Exception
        fake_data_dir = os.path.join(self.current_dir, "fake_data_dir")
        error_test_repo_id = "i4h/error_test_data"
        with self.assertRaises(Exception) as context:
            convert_hdf5_to_lerobot(fake_data_dir, error_test_repo_id, self.test_prompt, self.feature_builder)
        self.assertTrue(f"Data directory {fake_data_dir} does not exist." == str(context.exception))

        # check get expected no hdf5 files warning message
        fake_empty_data_dir = os.path.join(self.current_dir, "fake_empty_data_dir")
        os.makedirs(fake_empty_data_dir, exist_ok=True)
        with self.assertWarns(Warning) as context:
            convert_hdf5_to_lerobot(fake_empty_data_dir, error_test_repo_id, self.test_prompt, self.feature_builder)
        self.assertTrue(f"No HDF5 files found in {fake_empty_data_dir}" == str(context.warning))

        # check get expected repo_id warning message
        hdf5_path = os.path.join(self.hdf5_data_dir, "data_0.hdf5")
        wrong_name_hdf5_path = os.path.join(fake_empty_data_dir, "wrong_name_0.hdf5")
        shutil.copy(hdf5_path, wrong_name_hdf5_path)
        with self.assertWarns(Warning) as context:
            convert_hdf5_to_lerobot(fake_empty_data_dir, error_test_repo_id, self.test_prompt, self.feature_builder)
        self.assertTrue(f"File {wrong_name_hdf5_path} does not match the expected pattern." == str(context.warning))

        # clean up
        shutil.rmtree(fake_empty_data_dir)
        shutil.rmtree(os.path.join(self.cache_dir, error_test_repo_id))

    def test_create_lerobot_dataset(self):
        """Test that LeRobot dataset can be created successfully."""
        output_path = os.path.join(self.current_dir, "test_dataset_output")
        # check get expected Exception
        os.makedirs(output_path, exist_ok=True)
        with self.assertRaises(Exception) as context:
            create_lerobot_dataset(
                output_path=output_path,
                features=self.feature_builder.features,
            )
        self.assertTrue(f"Output path {output_path} already exists." == str(context.exception))
        # clean up
        shutil.rmtree(output_path)

        # check normal creation
        fps = 30
        image_shape = (224, 224, 3)
        state_shape = (7,)
        actions_shape = (6,)
        test_dataset = create_lerobot_dataset(
            output_path=output_path,
            features=self.feature_builder.features,
            fps=fps,
        )

        self.assertEqual(test_dataset.fps, fps)
        self.assertEqual(test_dataset.features["image"]["shape"], image_shape)
        self.assertEqual(test_dataset.features["state"]["shape"], state_shape)
        self.assertEqual(test_dataset.features["actions"]["shape"], actions_shape)
        self.assertTrue(os.path.exists(os.path.join(output_path, "meta")))

        # clean up
        shutil.rmtree(output_path)


class TestNormalizationStats(TestBase):
    """Test the computation of normalization statistics."""

    def test_compute_normalization_stats(self):
        """Test that normalization statistics can be computed successfully."""
        # Compute normalization statistics
        # get number of GPUs
        num_gpus = torch.cuda.device_count()
        compute_normalization_stats(self.test_config, batch_size=num_gpus)

        # Check that the stats file was created
        output_path = self.test_config.assets_dirs / self.TEST_REPO_ID
        stats_file = output_path / "norm_stats.json"

        self.assertTrue(os.path.exists(stats_file), f"Stats file not created at {stats_file}")


class TestTraining(TestBase):
    """Test the training process."""

    def test_training_runs_for_one_minute(self):
        """Test that training can run for at least one minute without errors."""
        # Set the flag to True so that the test data is cleaned up after the final test
        self.should_cleanup = True
        # First ensure normalization stats exist
        # Check that the stats file was created
        output_path = self.test_config.assets_dirs / self.TEST_REPO_ID
        stats_file = output_path / "norm_stats.json"

        self.assertTrue(os.path.exists(stats_file), f"Stats file not created at {stats_file}")

        # Start training in a separate thread so we can stop it after a minute
        self.training_thread = threading.Thread(target=self._run_training)
        self.training_thread.daemon = True

        # Start timer
        start_time = time.time()
        self.training_thread.start()

        # Let training run for at least one minute
        time.sleep(60)

        # Check that training ran for at least one minute
        elapsed_time = time.time() - start_time
        self.assertGreaterEqual(elapsed_time, 60, "Training should run for at least one minute")

        # Training should still be running
        self.assertTrue(self.training_thread.is_alive(), "Training thread should still be running")

    def _run_training(self):
        """Run the training process."""
        try:
            train.main(self.test_config)
        except Exception as e:
            self.fail(f"Training failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
