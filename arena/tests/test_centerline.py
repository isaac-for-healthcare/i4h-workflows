# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from i4h_arena.medical.centerline import ordered_centerline_path, sample_polyline


def test_ordered_centerline_uses_lowest_endpoint_and_farthest_branch() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0],
            [2.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [2, 4]], dtype=np.int64)

    path = ordered_centerline_path(points, edges, target_spacing_mm=0.5)

    np.testing.assert_allclose(path[0], points[0])
    assert np.linalg.norm(path[-1] - points[3]) < 1e-5


def test_sample_polyline_clamps_and_interpolates_arc_length() -> None:
    path = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float32)

    sampled = sample_polyline(path, np.asarray([-1.0, 0.5, 2.0, 5.0]))

    np.testing.assert_allclose(sampled, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
