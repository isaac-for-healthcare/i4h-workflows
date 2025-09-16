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
import subprocess
import sys
import tempfile
import unittest

import h5py
from helpers import cleanup_test_files, create_dummy_hdf5_files, requires_isaac_sim


class TestReplayRecording(unittest.TestCase):
    """Test cases for replay_recording.py functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = tempfile.mkdtemp()
        self.test_files.append(self.temp_dir)

        script_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "simulation", "environments")
        self.replay_script = os.path.join(script_dir, "replay_recording.py")

        if not os.path.exists(self.replay_script):
            raise FileNotFoundError(f"Replay script not found at {self.replay_script}")

        # Create test datasets
        self._create_test_datasets()

    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)

    def _create_test_datasets(self):
        """Create test datasets for replay testing."""
        self.main_hdf5_dir = os.path.join(self.temp_dir, "main_hdf5_data")
        create_dummy_hdf5_files(self.main_hdf5_dir, num_episodes=3, num_steps=20)

        # Create empty dataset
        self.empty_dataset_path = os.path.join(self.temp_dir, "empty_dataset.hdf5")
        self._create_empty_dataset(self.empty_dataset_path)

    def _create_empty_dataset(self, output_path):
        """Create an empty dataset for testing error handling."""
        with h5py.File(output_path, "w") as f:
            data_group = f.create_group("data")
            data_group.attrs["env_name"] = "Isaac-SOARM101-v0"
            data_group.attrs["total"] = 0

    def _run_replay_command(self, dataset_path, extra_args=None, timeout=60):
        """Run the replay script with given arguments."""
        cmd = [
            sys.executable,
            self.replay_script,
            "--dataset_path",
            dataset_path,
            "--task",
            "Isaac-SOARM101-v0",
            "--teleop_device",
            "keyboard",
            "--headless",  # Run without GUI
        ]

        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        return result.returncode == 0, result.stdout, result.stderr

    def test_basic_replay_functionality(self):
        """Test basic replay functionality."""
        # Use the first file created by the helper function
        dataset_path = os.path.join(self.main_hdf5_dir, "data_0.hdf5")
        success, stdout, stderr = self._run_replay_command(dataset_path, timeout=120)

        if not success:
            # Check for specific error patterns to provide helpful diagnostics
            if "No episodes found" in stderr:
                self.fail(f"No episodes found in dataset: {stderr}")
            elif "FileNotFoundError" in stderr:
                self.fail(f"Dataset file not found: {stderr}")
            elif "CUDA" in stderr:
                self.skipTest(f"CUDA/GPU related issue: {stderr}")
            else:
                self.fail(f"Basic replay test failed: {stderr}")

        self.assertTrue(success, "Basic replay functionality should work")

    def test_empty_dataset_handling(self):
        """Test that empty datasets are handled gracefully."""
        success, stdout, stderr = self._run_replay_command(self.empty_dataset_path, timeout=30)

        # The script should either fail (return code != 0) OR have error in stderr
        has_error = "KeyError" in stderr or "env_args" in stderr or "No episodes found" in stderr or not success

        self.assertTrue(
            has_error,
            f"Empty dataset should fail or show error. "
            f"Success: {success}, stdout: {stdout[:200]}, stderr: {stderr[:200]}",
        )

    def test_nonexistent_file_handling(self):
        """Test handling of non-existent dataset files."""
        nonexistent_path = "/nonexistent/path.hdf5"
        success, stdout, stderr = self._run_replay_command(nonexistent_path, timeout=30)

        # The script should either fail (return code != 0) OR have error in stderr
        has_error = "FileNotFoundError" in stderr or "does not exist" in stderr or not success

        self.assertTrue(
            has_error,
            f"Non-existent file should fail or show error. "
            f"Success: {success}, stdout: {stdout[:200]}, stderr: {stderr[:200]}",
        )

    @requires_isaac_sim
    def test_with_isaacsim_environment(self):
        """Test replay with actual Isaac Sim environment (requires Isaac Sim)."""
        dataset_path = os.path.join(self.main_hdf5_dir, "data_0.hdf5")
        success, stdout, stderr = self._run_replay_command(dataset_path, timeout=180)

        self.assertTrue(success, f"IsaacSim environment test failed: {stderr}")


if __name__ == "__main__":
    unittest.main()
