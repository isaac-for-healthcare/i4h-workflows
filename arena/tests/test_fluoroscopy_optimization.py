# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare optimized display paths against the frozen, identical-input reference."""

from __future__ import annotations

import types
import weakref
from itertools import combinations

import numpy as np
import pytest
import torch
from fixtures import fluoroscopy_reference as reference

from i4h_arena.medical.catheter import CatheterState
from i4h_arena.medical.slang_fluoroscopy import SlangFluoroscopyRenderer
from i4h_arena.medical.synthetic_fluoroscopy import SyntheticFluoroscopyRenderer
from i4h_arena.ui.sensor_image_window import SensorImageWindow, frame_rgba
from i4h_common.types import CameraFrame


@pytest.mark.parametrize(
    "height,width",
    [(0, 0), (0, 8), (8, 0), (1, 1), (3, 79), (117, 5), (512, 512)]
    + [tuple(map(int, shape)) for shape in np.random.default_rng(7).integers(1, 257, (20, 2))],
)
def test_rgba_exact_owned_and_retained(height, width):
    data = np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8).tobytes()
    frame = CameraFrame(name="test", height=height, width=width, data=data, encoding="rgb8")
    actual = frame_rgba(frame)
    np.testing.assert_array_equal(actual, reference.frame_rgba(frame))
    assert actual.flags.c_contiguous and actual.flags.owndata
    saved = actual.copy()
    next_frame = frame_rgba(frame)
    next_frame[...] = 0
    np.testing.assert_array_equal(actual, saved)
    assert frame.data == data


@pytest.mark.parametrize(
    "height,width,data,encoding",
    [(2, 2, b"bad", "rgb8"), (1, 1, b"1234", "rgb8"), (-1, 1, b"", "rgb8"), (1, 1, b"123", "bgr8")],
)
def test_rgba_rejects_same_malformed_inputs(height, width, data, encoding):
    frame = CameraFrame(name="test", height=height, width=width, data=data, encoding=encoding)
    for convert in (reference.frame_rgba, frame_rgba):
        with pytest.raises(ValueError):
            convert(frame)


@pytest.mark.parametrize("width,height", [(1, 1), (7, 13), (127, 63)])
def test_strokes_exact_across_projection_edges_and_chunk_boundary(width, height):
    renderer = SyntheticFluoroscopyRenderer(width=width, height=height)
    cases = [
        [],
        [[1, 1]],
        [[1, 1], [1, 1]],
        [[-5, 0], [width + 5, 0]],
        [[0, -5], [0, height + 5]],
        [[-10, -10], [width + 10, height + 10]],
        [[0.5, 0.5], [2.5, 4.5]],
        [[-50, -20], [-10, -20]],
        [[0, 0], [66000, 0]],
    ]
    rng = np.random.default_rng(78)
    cases.extend(rng.uniform(-20, max(width, height) + 20, (50, 30, 2)))
    for pixels in cases:
        pixels = np.asarray(pixels).reshape(-1, 2)
        expected = np.full((height, width, 3), 255, dtype=np.uint8)
        actual = expected.copy()
        reference._draw_catheter(renderer, expected, pixels)
        renderer._draw_catheter(actual, pixels)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("worlds", [1, 8, 16])
def test_synthetic_all_channels_moving_and_independent(worlds):
    renderer = SyntheticFluoroscopyRenderer(width=79, height=51)
    old = SyntheticFluoroscopyRenderer(width=79, height=51)
    old._draw_catheter = types.MethodType(reference._draw_catheter, old)
    rng = np.random.default_rng(91)
    previous = None
    for tick in range(5):
        points = rng.uniform(-0.5, 0.5, (worlds, 21, 3)).astype(np.float32)
        counts = np.take([0, 1, 2, 21], (np.arange(worlds) + tick) % 4)
        state = CatheterState(points, counts) if tick else CatheterState.empty(worlds)
        source = state.positions_world_m.copy()
        expected, actual = old.render(state), renderer.render(state)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(source, state.positions_world_m)
        for a, b in combinations(actual.values(), 2):
            assert not np.shares_memory(a, b)
        if previous:
            for key in actual:
                np.testing.assert_array_equal(previous[0][key], previous[1][key])
                assert not np.shares_memory(actual[key], previous[0][key])
        previous = actual, {key: value.copy() for key, value in actual.items()}


@pytest.mark.parametrize("annotate", [False, True])
@pytest.mark.parametrize("worlds", [1, 8, 16])
@pytest.mark.parametrize("valid", [0, 1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dsa_transfer_exactness_and_lifetime(monkeypatch, annotate, worlds, valid, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    renderer = object.__new__(SlangFluoroscopyRenderer)
    renderer._project_catheter = lambda points, projection: (points[:, :2], np.ones(len(points), dtype=bool))
    state = CatheterState(np.repeat([[[10, 10, 0], [30, 20, 0]]], worlds, axis=0), np.full(worlds, valid))
    frames = tuple(torch.randint(0, 256, (worlds, 48, 64, 3), dtype=torch.uint8, device=device) for _ in range(2))
    # Production initializes guidance channels as independent clones of their source channels.
    frames = frames[0], frames[0].clone(), frames[1], frames[1].clone()
    old_frames = tuple(t.clone() for t in frames)
    cpu_calls = []
    original = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        cpu_calls.append(tensor)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    expected = reference._draw_catheter_overlays(renderer, state, [None] * worlds, annotate=annotate, frames=old_frames)
    assert len(cpu_calls) == (0 if valid < 2 else 4 if annotate else 3)
    cpu_calls.clear()
    before = tuple(t.clone() for t in frames)
    actual = renderer._draw_catheter_overlays(state, [None] * worlds, annotate=annotate, frames=frames)
    # Preserve outputs and avoid extra host round trips relative to the reference.
    assert len(cpu_calls) <= (0 if valid < 2 else 4 if annotate else 3)
    if not annotate or valid < 2:
        assert actual[0] is frames[0] and actual[2] is frames[2]
    for a, b, source, old_source in zip(actual, expected, frames, old_frames, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(source, old_source, rtol=0, atol=0)
    if not annotate:
        torch.testing.assert_close(frames[0], before[0], rtol=0, atol=0)
        torch.testing.assert_close(frames[2], before[2], rtol=0, atol=0)


def test_window_retains_owned_payload_after_provider_borrows_it():
    frame = CameraFrame(name="test", height=1, width=2, data=bytes(range(6)), encoding="rgb8")
    borrowed = []
    window = object.__new__(SensorImageWindow)
    window.name = "test"
    window._readout = window._readout_label = window._last_frame = None
    window._outputs = (("RGB", "rgb"),)
    window._output_index = 0
    window._view = types.SimpleNamespace(camera=lambda *args, **kwargs: frame)
    window._provider = types.SimpleNamespace(set_data_array=lambda array, size: borrowed.append(weakref.ref(array)))
    window.update()
    assert borrowed[0]() is window._last_rgba
    assert window._last_rgba.flags.owndata and window._last_rgba.flags.c_contiguous
    expected = window._last_rgba.copy()
    window.update()
    assert len(borrowed) == 1
    np.testing.assert_array_equal(borrowed[0](), expected)
