# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Newton and scene-data backed state providers.

Both providers are exercised against fakes rather than a live stack: the
catheter provider only needs an object exposing ``particle_q``, and the C-arm
provider only needs the two ``SceneDataProvider`` methods it calls. That keeps
the index arithmetic and the environment fan-out under test without Isaac Sim.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from i4h_arena.medical.newton_providers import NewtonRodCatheterStateProvider, SceneDataCArmStateProvider

NUM_POINTS = 4
NUM_ENVS = 3


class FakeWarpArray:
    """Minimal stand-in for a warp array: only ``numpy()`` and ``len``."""

    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=np.float32)

    def numpy(self) -> np.ndarray:
        return self._values

    def __len__(self) -> int:
        return len(self._values)


def _particles(num_particles: int, *, offset_value: float = 0.0) -> FakeWarpArray:
    """Distinct, ordered positions so a mis-sliced range is visible."""
    base = np.arange(num_particles * 3, dtype=np.float32).reshape(num_particles, 3)
    return FakeWarpArray(base + offset_value)


def test_reshapes_particles_into_per_env_polylines():
    particles = _particles(NUM_POINTS * NUM_ENVS)
    provider = NewtonRodCatheterStateProvider(
        lambda: SimpleNamespace(particle_q=particles),
        offset=0,
        num_points=NUM_POINTS,
        num_envs=NUM_ENVS,
        radius_m=0.001,
    )

    state = provider.snapshot(NUM_ENVS)

    assert state.positions_world_m.shape == (NUM_ENVS, NUM_POINTS, 3)
    np.testing.assert_array_equal(state.valid_nodes, [NUM_POINTS] * NUM_ENVS)
    assert state.radius_m == pytest.approx(0.001)
    # Environment 1 must be the second contiguous block, not a stride.
    expected = particles.numpy()[NUM_POINTS : 2 * NUM_POINTS]
    np.testing.assert_array_equal(state.positions_world_m[1], expected)


def test_honours_a_nonzero_particle_offset():
    """A rod sharing Newton's particle array with other bodies starts mid-buffer."""
    leading = 7
    particles = _particles(leading + NUM_POINTS * NUM_ENVS)
    provider = NewtonRodCatheterStateProvider(
        lambda: SimpleNamespace(particle_q=particles),
        offset=leading,
        num_points=NUM_POINTS,
        num_envs=NUM_ENVS,
        radius_m=0.001,
    )

    state = provider.snapshot(NUM_ENVS)

    expected = particles.numpy()[leading : leading + NUM_POINTS]
    np.testing.assert_array_equal(state.positions_world_m[0], expected)


def test_applies_the_world_origin_offset():
    particles = _particles(NUM_POINTS)
    origin = np.array([1.5, -2.0, 0.25], dtype=np.float32)
    provider = NewtonRodCatheterStateProvider(
        lambda: SimpleNamespace(particle_q=particles),
        offset=0,
        num_points=NUM_POINTS,
        num_envs=1,
        radius_m=0.001,
        origin_world_m=origin,
    )

    state = provider.snapshot(1)

    np.testing.assert_allclose(state.positions_world_m[0], particles.numpy() + origin)


def test_reads_the_state_fresh_on_every_snapshot():
    """Newton swaps state objects between substeps, so a cached one goes stale."""
    states = [
        SimpleNamespace(particle_q=_particles(NUM_POINTS, offset_value=0.0)),
        SimpleNamespace(particle_q=_particles(NUM_POINTS, offset_value=10.0)),
    ]
    calls = {"n": 0}

    def next_state():
        state = states[min(calls["n"], len(states) - 1)]
        calls["n"] += 1
        return state

    provider = NewtonRodCatheterStateProvider(next_state, offset=0, num_points=NUM_POINTS, num_envs=1, radius_m=0.001)

    first = provider.snapshot(1).positions_world_m.copy()
    second = provider.snapshot(1).positions_world_m

    assert not np.allclose(first, second)


def test_rejects_a_mismatched_environment_count():
    provider = NewtonRodCatheterStateProvider(
        lambda: SimpleNamespace(particle_q=_particles(NUM_POINTS)),
        offset=0,
        num_points=NUM_POINTS,
        num_envs=1,
        radius_m=0.001,
    )

    with pytest.raises(ValueError, match="environment"):
        provider.snapshot(2)


def test_reports_a_missing_particle_buffer_actionably():
    """The fix is scene-setup ordering, so the message has to say so."""
    provider = NewtonRodCatheterStateProvider(
        lambda: SimpleNamespace(), offset=0, num_points=NUM_POINTS, num_envs=1, radius_m=0.001
    )

    with pytest.raises(RuntimeError, match="MODEL_INIT"):
        provider.snapshot(1)


def test_detects_a_range_that_overruns_the_buffer():
    provider = NewtonRodCatheterStateProvider(
        lambda: SimpleNamespace(particle_q=_particles(NUM_POINTS)),
        offset=2,
        num_points=NUM_POINTS,
        num_envs=1,
        radius_m=0.001,
    )

    with pytest.raises(RuntimeError, match="does not fit"):
        provider.snapshot(1)


def test_from_particle_range_derives_points_per_env():
    particle_range = SimpleNamespace(offset=5, count=NUM_POINTS * NUM_ENVS, num_envs=NUM_ENVS)
    particles = _particles(5 + NUM_POINTS * NUM_ENVS)

    provider = NewtonRodCatheterStateProvider.from_particle_range(
        lambda: SimpleNamespace(particle_q=particles), particle_range, radius_m=0.002
    )
    state = provider.snapshot(NUM_ENVS)

    assert state.positions_world_m.shape == (NUM_ENVS, NUM_POINTS, 3)


# --------------------------------------------------------------------------- #
# C-arm provider
# --------------------------------------------------------------------------- #
class FakeSceneDataProvider:
    """Records the mapping it was asked for and serves canned transforms."""

    def __init__(self, transforms: np.ndarray, *, succeed: bool = True):
        self._transforms = np.asarray(transforms, dtype=np.float32)
        self._succeed = succeed
        self.requested_paths: list[str] | None = None
        self.allow_passthrough: bool | None = None

    def create_mapping(self, paths):
        self.requested_paths = list(paths)
        return np.arange(len(paths), dtype=np.int32)

    def get_transforms(self, output, mapping=None, allow_passthrough=True):
        self.allow_passthrough = allow_passthrough
        if not self._succeed:
            return False
        output.transforms = FakeWarpArray(self._transforms)
        return True


def _identity_transforms(num_envs: int) -> np.ndarray:
    """Sources below, detectors above, all with identity orientation (XYZW)."""
    rows = []
    for env in range(num_envs):
        rows.append([0.0, 0.0, -0.5 - env, 0.0, 0.0, 0.0, 1.0])
    for env in range(num_envs):
        rows.append([0.0, 0.0, 0.5 + env, 0.0, 0.0, 0.0, 1.0])
    return np.asarray(rows, dtype=np.float32)


@pytest.fixture(autouse=True)
def stub_scene_data_format(monkeypatch):
    """Stand in for isaaclab.scene_data, which needs a full Isaac Lab install."""
    import sys
    import types

    module = types.ModuleType("isaaclab.scene_data")

    class _Transform:
        def __init__(self):
            self.transforms = None

    module.SceneDataFormat = SimpleNamespace(Transform=_Transform)
    isaaclab = sys.modules.get("isaaclab") or types.ModuleType("isaaclab")
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.scene_data", module)
    yield


def _carm(num_envs: int, **kwargs) -> tuple[SceneDataCArmStateProvider, FakeSceneDataProvider]:
    backend = FakeSceneDataProvider(_identity_transforms(num_envs), **kwargs)
    provider = SceneDataCArmStateProvider(
        backend,
        source_paths=[f"/World/envs/env_{i}/CArm/Orbit/Source" for i in range(num_envs)],
        detector_paths=[f"/World/envs/env_{i}/CArm/Orbit/Detector" for i in range(num_envs)],
        detector_size_m=(0.6144, 0.6144),
    )
    return provider, backend


def test_carm_splits_sources_from_detectors():
    provider, _ = _carm(NUM_ENVS)

    state = provider.snapshot(NUM_ENVS)

    assert state.source_world_m.shape == (NUM_ENVS, 3)
    assert state.detector_center_world_m.shape == (NUM_ENVS, 3)
    # Sources sit below the isocenter, detectors above; swapping them would
    # mirror the projection.
    assert np.all(state.source_world_m[:, 2] < 0.0)
    assert np.all(state.detector_center_world_m[:, 2] > 0.0)


def test_carm_maps_both_bodies_in_one_request():
    provider, backend = _carm(NUM_ENVS)
    provider.snapshot(NUM_ENVS)

    assert backend.requested_paths is not None
    assert len(backend.requested_paths) == 2 * NUM_ENVS
    assert backend.requested_paths[0].endswith("Source")
    assert backend.requested_paths[NUM_ENVS].endswith("Detector")


def test_carm_refuses_passthrough_so_the_mapping_applies():
    """A zero-copy view of the backend array would ignore the remap."""
    provider, backend = _carm(NUM_ENVS)
    provider.snapshot(NUM_ENVS)

    assert backend.allow_passthrough is False


def test_carm_derives_the_detector_axis_from_orientation():
    provider, _ = _carm(1)

    state = provider.snapshot(1)

    # Identity orientation leaves local +X as world +X.
    np.testing.assert_allclose(state.detector_x_axis_world[0], [1.0, 0.0, 0.0], atol=1e-6)


def test_carm_surfaces_a_failed_conversion():
    provider, _ = _carm(1, succeed=False)

    with pytest.raises(RuntimeError, match="could not convert"):
        provider.snapshot(1)


def test_carm_rejects_unpaired_prim_paths():
    backend = FakeSceneDataProvider(_identity_transforms(1))
    with pytest.raises(ValueError, match="one of each per environment"):
        SceneDataCArmStateProvider(
            backend,
            source_paths=["/a", "/b"],
            detector_paths=["/c"],
            detector_size_m=(0.5, 0.5),
        )
