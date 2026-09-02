# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from i4h_arena.adapters.scene_view import ArenaSceneView
from i4h_arena.medical.catheter import CatheterState, StaticCatheterStateProvider
from i4h_arena.medical.synthetic_fluoroscopy import SyntheticFluoroscopyRenderer


def _catheter() -> CatheterState:
    return CatheterState(
        positions_world_m=np.array(
            [[[-0.1, 0.1, 0.0], [-0.04, 0.03, 0.0], [0.03, -0.02, 0.0], [0.1, -0.1, 0.0]]],
            dtype=np.float32,
        ),
        valid_nodes=np.array([4], dtype=np.int32),
    )


def test_catheter_provider_broadcasts_one_environment() -> None:
    state = StaticCatheterStateProvider(_catheter()).snapshot(3)

    assert state.positions_world_m.shape == (3, 4, 3)
    np.testing.assert_array_equal(state.valid_nodes, [4, 4, 4])


def test_synthetic_renderer_matches_arena_camera_contract() -> None:
    output = SyntheticFluoroscopyRenderer(width=96, height=64).render(_catheter())
    sensor = SimpleNamespace(data=SimpleNamespace(output=output))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"fluoroscopy": sensor}, num_envs=1))

    frame = ArenaSceneView(env, cameras=("fluoroscopy",)).camera("fluoroscopy")

    assert frame is not None
    assert (frame.height, frame.width, frame.encoding) == (64, 96, "rgb8")
    image = frame.to_array()
    assert image.shape == (64, 96, 3)
    assert image.min() == 12
    assert image.max() > image.min()

    guidance = ArenaSceneView(env, cameras=("fluoroscopy",)).camera("fluoroscopy", output="guidance")
    assert guidance is not None
    guidance_image = guidance.to_array()
    assert np.any(guidance_image[..., 1] > guidance_image[..., 0])
    assert not np.array_equal(guidance_image, image)


def test_synthetic_ap_projection_uses_world_xy_detector_plane() -> None:
    renderer = SyntheticFluoroscopyRenderer(width=96, height=64)
    points = np.array([[[-0.2, -0.2, -8.0], [0.2, 0.2, 9.0]]], dtype=np.float32)
    catheter = CatheterState(positions_world_m=points, valid_nodes=np.array([2], dtype=np.int32))
    reference = renderer.render(catheter)["rgb"]

    points[:, :, 2] = (30.0, -30.0)
    moved_along_beam = renderer.render(catheter)["rgb"]

    np.testing.assert_array_equal(reference, moved_along_beam)
    catheter_pixels = np.argwhere(reference[0, :, :, 0] == 12)
    assert np.ptp(catheter_pixels[:, 0]) > 20
    assert np.ptp(catheter_pixels[:, 1]) > 40


def test_catheter_state_rejects_invalid_active_length() -> None:
    with pytest.raises(ValueError, match="between zero and num_nodes"):
        CatheterState(
            positions_world_m=np.zeros((1, 2, 3), dtype=np.float32),
            valid_nodes=np.array([3], dtype=np.int32),
        )
