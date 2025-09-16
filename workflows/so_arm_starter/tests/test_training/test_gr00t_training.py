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

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from helpers import cleanup_test_files, create_dummy_hdf5_files, requires_gpu_memory


class TestGR00TTraining(unittest.TestCase):
    """Test cases for GR00T training functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = tempfile.mkdtemp()
        self.test_files.append(self.temp_dir)

        # Set up checkpoint directory
        self.checkpoint_dir = "/tmp/gr00t"
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.test_files.append(self.checkpoint_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up checkpoint directory
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)

        # Clean up any LeRobot datasets created in temp directory
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            # Clean up HDF5 data directory
            hdf5_data_dir = os.path.join(self.temp_dir, "hdf5_data")
            if os.path.exists(hdf5_data_dir):
                print(f"Cleaning up HDF5 data: {hdf5_data_dir}")
                shutil.rmtree(hdf5_data_dir)

            # Clean up converted LeRobot dataset
            dataset_dir = os.path.join(self.temp_dir, "test_dataset")
            if os.path.exists(dataset_dir):
                print(f"Cleaning up LeRobot dataset: {dataset_dir}")
                shutil.rmtree(dataset_dir)

            # Clean up any other dataset directories that might be created
            for item in os.listdir(self.temp_dir):
                item_path = os.path.join(self.temp_dir, item)
                if os.path.isdir(item_path) and item not in [".", ".."]:
                    print(f"Cleaning up additional directory: {item_path}")
                    shutil.rmtree(item_path)

        cleanup_test_files(self.test_files)

    def _create_minimal_dataset(self):
        """Create minimal HDF5 files and convert them using the actual conversion script."""
        hdf5_data_dir = os.path.join(self.temp_dir, "hdf5_data")

        # Create mock HDF5 files with the expected structure for SO-ARM101
        create_dummy_hdf5_files(hdf5_data_dir)

        # Convert using the actual conversion script
        dataset_dir = os.path.join(self.temp_dir, "test_dataset")
        self._convert_hdf5_to_lerobot(hdf5_data_dir, dataset_dir)

        return dataset_dir

    def _convert_hdf5_to_lerobot(self, hdf5_data_dir, output_dir):
        """Convert HDF5 files to LeRobot format using the actual conversion script."""
        conversion_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/training/hdf5_to_lerobot.py"
        )

        if not os.path.exists(conversion_script):
            raise FileNotFoundError(f"Conversion script not found: {conversion_script}")

        # Get the first HDF5 file (the script expects a single file path)
        hdf5_files = [f for f in os.listdir(hdf5_data_dir) if f.endswith(".hdf5")]
        if not hdf5_files:
            raise FileNotFoundError("No HDF5 files found in data directory")

        hdf5_path = os.path.join(hdf5_data_dir, hdf5_files[0])

        # Set environment variable for LeRobot home to our temp directory
        env = os.environ.copy()
        env["HF_LEROBOT_HOME"] = self.temp_dir
        # Clear deprecated LEROBOT_HOME to avoid conflicts
        if "LEROBOT_HOME" in env:
            del env["LEROBOT_HOME"]

        print("Converting HDF5 to LeRobot format...")

        # Run conversion script - using only required parameters
        cmd = [
            sys.executable,
            conversion_script,
            "--repo_id",
            "test_dataset",
            "--hdf5_path",
            hdf5_path,
        ]

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,  # 2 minute timeout for conversion
        )

        if process.returncode != 0:
            raise RuntimeError(f"HDF5 conversion failed with return code {process.returncode}")

        print("HDF5 to LeRobot conversion completed successfully")

        # The dataset should be created in HF_LEROBOT_HOME/test_dataset
        converted_dataset_path = os.path.join(self.temp_dir, "test_dataset")
        if not os.path.exists(converted_dataset_path):
            raise FileNotFoundError(f"Converted dataset not found at {converted_dataset_path}")

        # Create the required modality.json file (not created by conversion script)
        self._create_modality_json(converted_dataset_path)

        return converted_dataset_path

    def _create_modality_json(self, dataset_path):
        """Create the modality.json file required for GR00T training."""
        meta_dir = os.path.join(dataset_path, "meta")
        if not os.path.exists(meta_dir):
            os.makedirs(meta_dir, exist_ok=True)

        # Create modality.json matching the training script's expectations
        modality = {
            "state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
            "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
            "video": {
                "room": {"original_key": "observation.images.room"},
                "wrist": {"original_key": "observation.images.wrist"},
            },
            "annotation": {"human.task_description": {"original_key": "task_index"}},
        }

        modality_path = os.path.join(meta_dir, "modality.json")
        with open(modality_path, "w") as f:
            json.dump(modality, f, indent=4)

        print(f"Created modality.json at: {modality_path}")
        return modality_path

    def test_dataset_structure_after_conversion(self):
        """Test that the dataset structure is correct after HDF5 to LeRobot conversion."""
        print("Testing dataset structure after HDF5 to LeRobot conversion...")

        # Create and convert dataset
        dataset_path = self._create_minimal_dataset()

        # Validate basic directory structure
        self.assertTrue(os.path.exists(dataset_path), f"Dataset directory should exist: {dataset_path}")

        # Check for required subdirectories
        expected_dirs = ["meta", "data", "videos"]
        for dir_name in expected_dirs:
            dir_path = os.path.join(dataset_path, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Required directory should exist: {dir_path}")

        # Validate metadata files
        meta_dir = os.path.join(dataset_path, "meta")
        # Check for other required metadata files
        expected_meta_files = ["info.json", "tasks.jsonl", "episodes.jsonl", "modality.json", "episodes_stats.jsonl"]
        for meta_file in expected_meta_files:
            meta_file_path = os.path.join(meta_dir, meta_file)
            if os.path.exists(meta_file_path):
                print(f"Found metadata file: {meta_file}")
            else:
                print(f"Warning: Optional metadata file not found: {meta_file}")

        # Check for video files
        video_dir = os.path.join(dataset_path, "videos", "chunk-000")
        for video_file in os.listdir(video_dir):
            video_nums = os.listdir(os.path.join(video_dir, video_file))
            self.assertGreater(len(video_nums), 0, "Should have at least one video file")
            for video_num in video_nums:
                video_path = os.path.join(video_dir, video_file, video_num)
                self.assertTrue(os.path.exists(video_path), f"Video file should exist: {video_path}")

        print("✅ Dataset structure validation completed successfully")

    @requires_gpu_memory(min_gib=30)
    def test_real_training(self):
        """Test actual GR00T training with real checkpoint saving."""
        # Find the actual training script
        training_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/training/gr00t_n1_5/train.py"
        )

        if not os.path.exists(training_script):
            raise FileNotFoundError(f"Training script not found: {training_script}")

        # Create minimal dataset (this actually creates the HDF5 files and converts them)
        dataset_path = self._create_minimal_dataset()

        # Prepare training command using the actual script interface
        cmd = [
            sys.executable,
            training_script,
            "--dataset-path",
            dataset_path,
            "--output-dir",
            self.checkpoint_dir,
            "--max-steps",
            "10",
            "--save-steps",
            "10",
            "--batch-size",
            "1",
            "--data-config",
            "so100_dualcam",  # Correct config for SO-ARM101 dual camera
            "--num-gpus",
            "1",
        ]

        # Run training
        start_time = time.time()
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Training stdout (last 500 chars): {process.stdout[-500:]}")

        if process.returncode != 0:
            print(f"Training stderr: {process.stderr}")

        # Verify checkpoints were created (only if training completed successfully)
        self.assertTrue(os.path.exists(self.checkpoint_dir), "Checkpoint directory should exist")

        # Check for checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(self.checkpoint_dir) if d.startswith("checkpoint-")]

        print(f"Found checkpoint directories: {checkpoint_dirs}")
        self.assertGreater(len(checkpoint_dirs), 0, "At least one checkpoint should be saved")

        # Verify checkpoint contents
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_dir)

            # Get list of files actually saved
            actual_files = os.listdir(checkpoint_path)
            print(f"Verified checkpoint: {checkpoint_dir}")
            print(f"  Files: {actual_files}")

            # Check for essential checkpoint files (at least some files should exist)
            self.assertGreater(len(actual_files), 0, f"Checkpoint directory {checkpoint_dir} should not be empty")

            # Check for common checkpoint files (but don't require specific model formats)
            expected_files = ["config.json", "training_args.bin", "trainer_state.json"]
            config_files_found = [f for f in actual_files if f in expected_files]

            if config_files_found:
                print(f"  Found expected config files: {config_files_found}")
            else:
                print(f"  No standard config files, but checkpoint has {len(actual_files)} files")

        # Verify training logs
        log_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".log") or f == "trainer_state.json"]
        print(f"Training logs: {log_files}")

        # Clean up during test to verify cleanup works
        self.assertTrue(os.path.exists(self.checkpoint_dir))


if __name__ == "__main__":
    unittest.main()
