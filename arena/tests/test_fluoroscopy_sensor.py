# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from i4h_arena.medical.catheter import CatheterState, StaticCatheterStateProvider
from i4h_arena.medical.synthetic_fluoroscopy import SyntheticFluoroscopyRenderer
from i4h_arena.ui.sensor_view import SensorView


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

    frame = SensorView(env, ("fluoroscopy",)).camera("fluoroscopy")

    assert frame is not None
    assert (frame.height, frame.width, frame.encoding) == (64, 96, "rgb8")
    image = frame.to_array()
    assert image.shape == (64, 96, 3)
    assert image.min() == 12
    assert image.max() > image.min()

    guidance = SensorView(env, ("fluoroscopy",)).camera("fluoroscopy", output="guidance")
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


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("valid", [0, 1, 4])
def test_synthetic_cached_background_matches_original_pixels_and_isolation(num_envs, valid):
    from i4h_arena.medical.fluoroscopy_guidance import draw_catheter_guidance

    renderer = SyntheticFluoroscopyRenderer(width=47, height=31)
    state = StaticCatheterStateProvider(_catheter()).snapshot(num_envs)
    state.valid_nodes[:] = valid
    for offset in [0.0, 0.03, -0.02, 0.0]:
        state.positions_world_m[:, :, 0] += offset
        # Original uncached renderer's image formation, including quantization order.
        attenuation = np.repeat(renderer._attenuation[None, ..., None], num_envs, axis=0)
        transmission = np.exp(-attenuation[..., 0])
        rgb = np.repeat(np.rint(255.0 * transmission)[..., None], 3, axis=-1).astype(np.uint8)
        guidance = rgb.copy()
        for index, count in enumerate(state.valid_nodes):
            if count >= 2:
                pixels = renderer._project_catheter(state.positions_world_m[index, :count])
                renderer._draw_catheter(rgb[index], pixels)
                guidance[index] = draw_catheter_guidance(rgb[index], pixels)
        expected = dict(
            rgb=rgb, guidance=guidance, dsa=rgb.copy(), dsa_guidance=guidance.copy(), attenuation=attenuation
        )
        actual = renderer.render(state)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
            assert actual[key].dtype == expected[key].dtype
        for key in actual:
            actual[key][0] = 0
            np.testing.assert_array_equal(actual[key][1:], expected[key][1:])
            for peer in expected.keys() - {key}:
                np.testing.assert_array_equal(actual[peer], expected[peer])
            fresh = renderer.render(state)
            for channel in expected:
                np.testing.assert_array_equal(fresh[channel], expected[channel])
            actual = renderer.render(state)
