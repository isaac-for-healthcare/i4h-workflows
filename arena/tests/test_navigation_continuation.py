# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host restoration validates before mutation and flushes host caches before numerical history."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from i4h_arena.medical import navigation_continuation as continuation


def test_invalid_snapshot_does_not_forward_or_mutate_host(monkeypatch):
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    calls = []
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        i4h=SimpleNamespace(validate_continuation=lambda payload, ids: None),
        sim=SimpleNamespace(forward=lambda: calls.append("forward")),
    )
    monkeypatch.setattr(continuation, "configuration_identity", lambda env: np.array([1, 2, 3], np.uint8))
    monkeypatch.setattr(continuation, "host_arrays", lambda env: {"value": value})
    bundle = dict(
        identity=np.array([[1, 2, 3]], np.uint8),
        numerical={},
        host=dict(
            value=np.array([[7.0, 8.0]], np.float64),
            progress_events=np.zeros((1, 65536), np.uint8),
        ),
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        continuation.restore(env, bundle, [1])
    assert not calls
    torch.testing.assert_close(value, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_host_caches_are_flushed_before_selected_numerical_restore(monkeypatch):
    import warp as wp

    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    calls = []
    sensor = SimpleNamespace(_is_outdated=wp.array([False, False], dtype=wp.bool, device="cpu"), _data_dirty=True)
    catheter = SimpleNamespace(_refresh_positions=lambda: calls.append("refresh"))
    drive = SimpleNamespace(_tip_bend_angle=torch.zeros(2), _write_pose=lambda: calls.append("pose"))

    def numerical_restore(bundle, ids):
        assert value[1].tolist() == [7.0, 8.0]
        calls.append("numerical")

    component = SimpleNamespace(
        physics=SimpleNamespace(coupled=None),
        validate_continuation=lambda payload, ids: None,
        restore_continuation=numerical_restore,
        order_inputs=lambda: None,
        sync_views=lambda: None,
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        i4h=component,
        progress_tracker=None,
        sim=SimpleNamespace(forward=lambda: calls.append("forward")),
        scene={"fluoroscopy": sensor, "catheter": catheter},
        action_manager=SimpleNamespace(get_term=lambda name: drive),
        common_step_counter=9,
    )
    monkeypatch.setattr(continuation, "configuration_identity", lambda env: np.array([1, 2, 3], np.uint8))
    monkeypatch.setattr(continuation, "host_arrays", lambda env: {"value": value})
    bundle = dict(
        identity=np.array([[1, 2, 3]], np.uint8),
        numerical={},
        host=dict(value=np.array([[7.0, 8.0]], np.float32), progress_events=np.zeros((1, 65536), np.uint8)),
    )
    continuation.restore(env, bundle, [1])
    assert calls == ["forward", "pose", "numerical", "refresh"]
    assert not sensor._data_dirty
    torch.testing.assert_close(value[0], torch.tensor([1.0, 2.0]))
