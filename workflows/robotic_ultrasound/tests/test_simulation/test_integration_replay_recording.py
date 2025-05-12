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

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil
import sys
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import run_with_monitoring
from parameterized import parameterized

TESTS_PATH = Path(__file__).parents[1].as_posix()

TEST_DATA_DIR = os.path.join(TESTS_PATH, "temp_test_hdf5_data_dir")
FAKE_HDF5_FILENAME = "data_0.hdf5"
ORGAN_POS = np.array([[0.69, -0.11, 0.09, -0.21, -0.00, 0.00, 0.98]] * 6)
ROBOT_JOINT_POS = np.array([[0.23, -0.75, -0.22, -2.47, -0.13, 1.73, 0.077]] * 6)
ROBOT_JOINT_VEL = np.array([[0.39, 2.32, -1.74, 2.28, 0.53, -1.36, -1.74]] * 6)
ACTIONS_DATA = np.array([[0.36, -0.15, -0.11, -0.01, 0.016, 0.012]] * 6)

TORSO_OBS_KEY = "observations/torso_obs"
JOINT_STATE_KEY = "abs_joint_pos"
JOINT_VEL_KEY = "observations/joint_vel"
ACTION_KEY_IN_HDF5 = "action"


TEST_CASES = [
    (
        f"python -u -m simulation.environments.state_machine.replay_recording "
        f"{TEST_DATA_DIR} --headless --enable_camera",
        300,
        "Completed all episodes",
    ),
    (
        f"python -u -m simulation.environments.state_machine.replay_recording "
        f"{TEST_DATA_DIR}/data_0.hdf5 --headless --enable_camera",
        300,
        "Completed all episodes",
    ),
]


class TestReplayRecording(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        cls.fake_hdf5_path = os.path.join(TEST_DATA_DIR, FAKE_HDF5_FILENAME)
        print(f"Creating fake HDF5 file at: {cls.fake_hdf5_path}")
        with h5py.File(cls.fake_hdf5_path, "w") as f:
            base_path = "data/demo_0"
            f.create_dataset(f"{base_path}/{TORSO_OBS_KEY}", data=ORGAN_POS)
            f.create_dataset(f"{base_path}/{JOINT_STATE_KEY}", data=ROBOT_JOINT_POS)
            f.create_dataset(f"{base_path}/{JOINT_VEL_KEY}", data=ROBOT_JOINT_VEL)
            f.create_dataset(f"{base_path}/{ACTION_KEY_IN_HDF5}", data=ACTIONS_DATA)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DATA_DIR):
            shutil.rmtree(TEST_DATA_DIR)

    @parameterized.expand(TEST_CASES)
    def test_replay_recording(self, command, timeout, target_line):
        # Run and monitor command
        _, found_target = run_with_monitoring(command, timeout, target_line)
        self.assertTrue(found_target)


if __name__ == "__main__":
    unittest.main()
