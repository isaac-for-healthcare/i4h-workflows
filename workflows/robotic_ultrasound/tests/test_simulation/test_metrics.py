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

import unittest

import numpy as np
from simulation.evaluation.metrics import compute_trajectory_overlap_and_distance


class TestComputeTrajectoryMetrics(unittest.TestCase):
    def test_identical_trajectories(self):
        traj = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
        percentage, avg_distance = compute_trajectory_overlap_and_distance(traj, traj, radius=0.1)
        self.assertEqual(percentage, 100.0)
        self.assertEqual(avg_distance, 0.0)

    def test_no_overlap(self):
        gt_traj = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        pred_traj = np.array([[10, 10, 10], [11, 11, 11]], dtype=float)
        percentage, avg_distance = compute_trajectory_overlap_and_distance(gt_traj, pred_traj, radius=0.1)
        self.assertEqual(percentage, 0.0)
        expected_avg_dist = (
            np.linalg.norm([(10 - 0), (10 - 0), (10 - 0)]) + np.linalg.norm([(10 - 1), (10 - 1), (10 - 1)])
        ) / 2
        self.assertAlmostEqual(avg_distance, expected_avg_dist, places=5)

    def test_partial_overlap(self):
        gt_traj = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5]], dtype=float)
        pred_traj = np.array([[0.05, 0.05, 0.05], [0.9, 0.9, 0.9], [10, 10, 10]], dtype=float)
        radius = 0.2
        percentage, avg_distance = compute_trajectory_overlap_and_distance(gt_traj, pred_traj, radius=radius)
        expected_percentage = (2 / 3) * 100.0
        self.assertAlmostEqual(percentage, expected_percentage, places=5)

        dist1 = np.linalg.norm(np.array([0, 0, 0]) - np.array([0.05, 0.05, 0.05]))
        dist2 = np.linalg.norm(np.array([1, 1, 1]) - np.array([0.9, 0.9, 0.9]))
        dist3 = np.linalg.norm(np.array([5, 5, 5]) - np.array([0.9, 0.9, 0.9]))
        expected_avg_dist = (dist1 + dist2 + dist3) / 3
        self.assertAlmostEqual(avg_distance, expected_avg_dist, places=5)

    def test_empty_predicted_trajectory(self):
        gt_traj = np.array([[0, 0, 0]], dtype=float)
        pred_traj = np.array([], dtype=float).reshape(0, 3)
        percentage, avg_distance = compute_trajectory_overlap_and_distance(gt_traj, pred_traj)
        self.assertEqual(percentage, 0.0)
        self.assertEqual(avg_distance, np.inf)

    def test_empty_ground_truth_trajectory(self):
        gt_traj = np.array([], dtype=float).reshape(0, 3)
        pred_traj = np.array([[0, 0, 0]], dtype=float)
        percentage, avg_distance = compute_trajectory_overlap_and_distance(gt_traj, pred_traj)
        self.assertEqual(percentage, 0.0)
        self.assertEqual(avg_distance, 0.0)

    def test_all_gt_covered_by_single_pred_point(self):
        gt_traj = np.array([[0, 0, 0], [0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]], dtype=float)
        pred_traj = np.array([[0, 0, 0]], dtype=float)
        percentage, avg_distance = compute_trajectory_overlap_and_distance(gt_traj, pred_traj, radius=0.05)
        self.assertEqual(percentage, 100.0)
        expected_avg_dist = (0.0 + np.linalg.norm([0.01, 0.01, 0.01]) + np.linalg.norm([-0.01, -0.01, -0.01])) / 3
        self.assertAlmostEqual(avg_distance, expected_avg_dist, places=5)

    def test_different_lengths(self):
        gt_traj = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
        pred_traj = np.array([[0.05, 0.05, 0.05], [1.05, 1.05, 1.05]], dtype=float)
        percentage, avg_distance = compute_trajectory_overlap_and_distance(gt_traj, pred_traj, radius=0.1)
        self.assertEqual(percentage, 50.0)

        dist_p0 = np.linalg.norm(gt_traj[0] - pred_traj[0])
        dist_p1 = np.linalg.norm(gt_traj[1] - pred_traj[1])
        dist_p2 = np.linalg.norm(gt_traj[2] - pred_traj[1])
        dist_p3 = np.linalg.norm(gt_traj[3] - pred_traj[1])
        expected_avg_dist = (dist_p0 + dist_p1 + dist_p2 + dist_p3) / 4
        self.assertAlmostEqual(avg_distance, expected_avg_dist, places=5)

    def test_radius_impact(self):
        gt_traj = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        pred_traj = np.array([[0.05, 0, 0], [0.5, 0, 0]], dtype=float)

        # Radius 0.01: only gt[0] cannot be covered by pred[0] (dist 0.05), gt[1] vs pred[1] (dist 0.5).
        percentage_r001, _ = compute_trajectory_overlap_and_distance(gt_traj, pred_traj, radius=0.01)
        self.assertEqual(percentage_r001, 0.0)

        # Radius 0.06: gt[0] covered by pred[0] (dist 0.05 < 0.06). gt[1] not covered by any (min dist 0.5 > 0.06)
        percentage_r006, _ = compute_trajectory_overlap_and_distance(gt_traj, pred_traj, radius=0.06)
        self.assertEqual(percentage_r006, 50.0)  # 1 out of 2 points

        # Radius 0.6: gt[0] covered (dist 0.05). gt[1] covered by pred[1] (dist 0.5 < 0.6)
        percentage_r06, _ = compute_trajectory_overlap_and_distance(gt_traj, pred_traj, radius=0.6)
        self.assertEqual(percentage_r06, 100.0)  # 2 out of 2 points

    def test_invalid_dimensions_gt(self):
        gt_traj = np.array([[0, 0], [1, 1]], dtype=float)  # Incorrect shape
        pred_traj = np.array([[0, 0, 0]], dtype=float)
        with self.assertRaises(ValueError):
            compute_trajectory_overlap_and_distance(gt_traj, pred_traj)

    def test_invalid_dimensions_pred(self):
        gt_traj = np.array([[0, 0, 0]], dtype=float)
        pred_traj = np.array([[0, 0]], dtype=float)  # Incorrect shape
        with self.assertRaises(ValueError):
            compute_trajectory_overlap_and_distance(gt_traj, pred_traj)

    def test_invalid_ndim_gt(self):
        gt_traj = np.array([0, 0, 0, 1, 1, 1], dtype=float)  # 1D array
        pred_traj = np.array([[0, 0, 0]], dtype=float)
        with self.assertRaises(ValueError):
            compute_trajectory_overlap_and_distance(gt_traj, pred_traj)

    def test_invalid_ndim_pred(self):
        gt_traj = np.array([[0, 0, 0]], dtype=float)
        pred_traj = np.array([0, 0, 0, 1, 1, 1], dtype=float)  # 1D array
        with self.assertRaises(ValueError):
            compute_trajectory_overlap_and_distance(gt_traj, pred_traj)


if __name__ == "__main__":
    unittest.main()
