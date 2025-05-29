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
import threading
import time
import unittest

import h5py
import numpy as np
from training.convert_hdf5_to_lerobot import GR00TN1FeatureDict
from training.convert_hdf5_to_lerobot import main as convert_hdf5_to_lerobot
from training.gr00t_n1 import train as gr00t_n1_train
from training.gr00t_n1.train import Config as TrainConfig


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
        self.tmp_checkpoints_dir = os.path.join(self.current_dir, "checkpoints")
        self.tmp_wandb_dir = os.path.join(self.current_dir, "wandb")
        self.hdf5_data_dir = os.path.join(self.current_dir, "test_data")

        # Create the test_data directory if it doesn't exist
        os.makedirs(self.hdf5_data_dir, exist_ok=True)

        # Create a basic config for testing
        self.test_prompt = "test_prompt"

        # Flag to allow for cleanup
        self.should_cleanup = False

        # Configure wandb to run in offline mode (no login required)
        os.environ["WANDB_MODE"] = "offline"

    def tearDown(self):
        """Clean up after each test method."""
        if self.should_cleanup:
            # safely kill the training thread
            if hasattr(self, "training_thread") and self.training_thread.is_alive():
                self.training_thread.join(timeout=10)
                if self.training_thread.is_alive():
                    self.training_thread.terminate()
                    self.training_thread.join()

            # Remove test data directory
            if os.path.exists(self.test_data_dir):
                shutil.rmtree(self.test_data_dir)

            # Remove any checkpoints in current directory
            if os.path.exists(self.tmp_checkpoints_dir):
                shutil.rmtree(self.tmp_checkpoints_dir)

            # Remove wandb directory if it exists
            if os.path.exists(self.tmp_wandb_dir):
                shutil.rmtree(self.tmp_wandb_dir)

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
        self.feature_builder = GR00TN1FeatureDict()

    def _create_dummy_hdf5_file(self):
        """Create a dummy HDF5 file with 25 steps for testing."""
        num_steps = 100
        num_episodes = 3

        # Create the test_data directory if it doesn't exist
        os.makedirs(self.hdf5_data_dir, exist_ok=True)

        for episode_idx in range(num_episodes):
            # Create a dummy HDF5 file
            hdf5_path = os.path.join(self.hdf5_data_dir, f"data_{episode_idx}.hdf5")

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

    def test_convert_hdf5_to_lerobot_gr00t_n1(self):
        """Test that HDF5 data can be converted to LeRobot format successfully."""
        convert_hdf5_to_lerobot(
            self.hdf5_data_dir, self.TEST_REPO_ID, self.test_prompt, feature_builder=GR00TN1FeatureDict()
        )
        meta_data_dir = os.path.join(self.test_data_dir, "meta")
        data_dir = os.path.join(self.test_data_dir, "data")
        video_dir = os.path.join(self.test_data_dir, "videos")

        self.assertTrue(os.path.exists(meta_data_dir), f"Meta data directory not created at {meta_data_dir}")
        self.assertTrue(os.path.exists(data_dir), f"Data directory not created at {data_dir}")
        self.assertTrue(os.path.exists(video_dir), f"Video directory not created at {video_dir}")


class TestTraining(TestBase):
    """Test the training process."""

    def test_training_runs_for_one_minute(self):
        """Test that training can run for at least one minute without errors."""
        # Set the flag to True so that the test data is cleaned up after the final test
        self.should_cleanup = True

        # Start training in a separate thread so we can stop it after a minute
        training_thread = threading.Thread(target=self._run_training)
        training_thread.daemon = True

        # Start timer
        start_time = time.time()
        training_thread.start()

        # Let training run for at least one minute
        time.sleep(60)

        # Check that training ran for at least one minute
        elapsed_time = time.time() - start_time
        self.assertGreaterEqual(elapsed_time, 60, "Training should run for at least one minute")

        # Training should still be running
        self.assertTrue(training_thread.is_alive(), "Training thread should still be running")

    def _run_training(self):
        """Run the training process."""
        try:
            # Create a Config object for training
            train_config = TrainConfig(
                dataset_path=self.test_data_dir,
                output_dir=self.tmp_checkpoints_dir,
                batch_size=1,
                num_gpus=1,
                dataloader_num_workers=1
            )
            gr00t_n1_train.main(train_config)
        except Exception as e:
            self.fail(f"Training failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
