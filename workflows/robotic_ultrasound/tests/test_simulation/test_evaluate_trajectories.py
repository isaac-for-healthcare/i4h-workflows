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
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from simulation.evaluation.evaluate_trajectories import main as evaluate_trajectories_main


class TestEvaluateTrajectories(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = self.tmpdir.name
        self.PREDICTION_SOURCES = {"dummy_method": {"file_pattern": "pred_ep{e}.npz", "label": "Dummy", "color": "red"}}
        self.DEFAULT_RADIUS = 0.01
        self.saved_compare_name = "summary.png"
        self.num_episodes = 1

        self._create_dummy_gt_hdf5(os.path.join(self.data_root, "data_0.hdf5"))
        self._create_dummy_pred_npz(os.path.join(self.data_root, "pred_ep0.npz"))

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_dummy_gt_hdf5(self, file_path):
        with h5py.File(file_path, "w") as hf:
            demo_0 = hf.require_group("data/demo_0")
            obs = demo_0.require_group("observations")
            actions = np.zeros((5, 1, 6), dtype=np.float32)
            actions[0, 0, 5] = 1
            obs.create_dataset("robot_obs", data=actions)
            states = np.arange(5)
            demo_0.create_dataset("state", data=states, dtype="i4")

    def _create_dummy_pred_npz(self, file_path):
        pred_traj = np.array([[[0, 1, 2]], [[3, 4, 0]], [[1, 2, 3]]], dtype=np.float32).reshape(3, 1, 1, 3)
        np.savez(file_path, robot_obs=pred_traj)

    @patch("simulation.evaluation.evaluate_trajectories.plot_3d_trajectories")
    @patch("simulation.evaluation.evaluate_trajectories.plot_success_rate_vs_radius")
    @patch("builtins.print")
    def test_main_runs_with_dummy_data(self, mock_print, mock_plot_summary, mock_plot_3d):
        # Should run without error and call plotting functions
        evaluate_trajectories_main(
            episode=self.num_episodes,
            data_root=self.data_root,
            DEFAULT_RADIUS_FOR_PLOTS=self.DEFAULT_RADIUS,
            saved_compare_name=self.saved_compare_name,
            PREDICTION_SOURCES=self.PREDICTION_SOURCES,
        )
        self.assertTrue(mock_plot_3d.called)
        self.assertTrue(mock_plot_summary.called)


if __name__ == "__main__":
    unittest.main()
