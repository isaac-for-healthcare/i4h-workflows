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
from unittest.mock import MagicMock

import torch
from simulation.environments.state_machine.data_collection import DataCollectionManager
from simulation.environments.state_machine.utils import UltrasoundState


class TestDataCollectionManager(unittest.TestCase):
    """Test cases for the DataCollectionManager class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)

        # Create data directory
        os.makedirs("./data/hdf5", exist_ok=True)

        # Mock environment
        self.mock_env = MagicMock()

        # Setup DataCollectionManager with testing flag
        self.task_name = "test_task"
        self.num_episodes = 2
        self.num_envs = 1
        self.device = "cpu"

        self.manager = DataCollectionManager(
            task_name=self.task_name,
            num_episodes=self.num_episodes,
            num_envs=self.num_envs,
            device=self.device,
            is_testing=True,
        )

        # Create dummy data
        self.dummy_obs = {"image": torch.zeros((self.num_envs, 3, 64, 64))}
        self.dummy_rel_action = torch.zeros((self.num_envs, 7))
        self.dummy_abs_action = torch.ones((self.num_envs, 7))
        self.dummy_robot_obs = torch.zeros((self.num_envs, 14))  # Assuming 14 dimensions

    def tearDown(self):
        """Clean up after each test method."""
        # Change back to original directory
        os.chdir(self.original_dir)

        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_data_collection_two_episodes(self):
        """Test collecting data for two complete episodes."""
        # Simulate data collection for two episodes
        for episode in range(self.num_episodes):
            # Record 10 steps per episode
            for _ in range(10):
                self.manager.record_step(
                    env=self.mock_env,
                    obs=self.dummy_obs,
                    rel_action=self.dummy_rel_action,
                    abs_action=self.dummy_abs_action,
                    robot_obs=self.dummy_robot_obs,
                    state=UltrasoundState.APPROACH.value,
                )

            # Complete the episode
            self.manager.on_episode_complete()

        # Check that the expected number of episodes were completed
        self.assertEqual(self.manager.completed_episodes, self.num_episodes)

        # Check that the output files were created
        hdf5_dirs = [d for d in os.listdir("./data/hdf5") if self.task_name in d]
        self.assertEqual(len(hdf5_dirs), 1, "Expected one output directory")

        output_dir = os.path.join("./data/hdf5", hdf5_dirs[0])
        hdf5_files = [f for f in os.listdir(output_dir) if f.endswith(".hdf5")]

        # Should have one file per episode
        self.assertEqual(
            len(hdf5_files), self.num_episodes, f"Expected {self.num_episodes} HDF5 files, found {len(hdf5_files)}"
        )

        # Verify file naming pattern
        for i in range(self.num_episodes):
            expected_file = f"data_{i}.hdf5"
            self.assertIn(expected_file, hdf5_files, f"Missing expected file: {expected_file}")

    def test_episode_reset(self):
        """Test resetting an episode before completion."""
        # Record some steps
        for _ in range(5):
            self.manager.record_step(
                env=self.mock_env,
                obs=self.dummy_obs,
                rel_action=self.dummy_rel_action,
                abs_action=self.dummy_abs_action,
                robot_obs=self.dummy_robot_obs,
                state=UltrasoundState.APPROACH.value,
            )

        # Reset the episode
        self.manager.on_episode_reset()

        # Verify that no episode was completed
        self.assertEqual(self.manager.completed_episodes, 0)
        # Verify that the dataset is empty
        self.assertEqual(self.manager.collector_interface._dataset, dict())

        # Now complete an episode properly
        for _ in range(10):
            self.manager.record_step(
                env=self.mock_env,
                obs=self.dummy_obs,
                rel_action=self.dummy_rel_action,
                abs_action=self.dummy_abs_action,
                robot_obs=self.dummy_robot_obs,
                state=UltrasoundState.APPROACH.value,
            )

        self.manager.on_episode_complete()
        self.assertEqual(self.manager.completed_episodes, 1)


if __name__ == "__main__":
    unittest.main()
