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
            fluoroscopy_dirty=np.zeros((1, 1), bool),
        ),
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        continuation.restore(env, bundle, [1])
    assert not calls
    torch.testing.assert_close(value, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
