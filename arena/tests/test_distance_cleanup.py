# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the post-containment distance cleanup sweep.

This is the pass that makes post-solve containment survivable: containment is
the last word on position and overrides the stretch constraints, leaving chords
at 7-364% of rest, which renders as overlapping particle clusters separated by
long gaps. These exercise the real Warp kernel on the CPU device rather than a
reimplementation, so the two-colour indexing and the pinned-particle handling
are covered as shipped.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")

# Below the skip guard on purpose: the solver package imports warp, so a machine
# without it has to skip rather than error at collection.
from catheter_vasculature_solver.cath_rod_solver import _distance_cleanup_kernel  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _warp_cpu(tmp_path_factory):
    """Initialize Warp with a writable kernel cache.

    The default cache lives under the user's home directory, which is not
    writable in every environment these run in; without this the failure looks
    like six broken tests rather than one unwritable path.
    """
    wp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("warp_cache"))
    wp.init()


def _sweep(positions, inv_masses, rest_lengths, *, num_envs, iterations=1, relaxation=1.0):
    """Run the kernel to convergence on the CPU device and return positions."""
    points_per_env = positions.shape[0] // num_envs
    edges_per_env = rest_lengths.shape[0] // num_envs
    pos = wp.array(np.asarray(positions, dtype=np.float32), dtype=wp.vec3, device="cpu")
    inv = wp.array(np.asarray(inv_masses, dtype=np.float32), dtype=float, device="cpu")
    rest = wp.array(np.asarray(rest_lengths, dtype=np.float32), dtype=float, device="cpu")
    for _ in range(iterations):
        for color in (0, 1):
            wp.launch(
                _distance_cleanup_kernel,
                dim=(num_envs, edges_per_env),
                inputs=[pos, inv, rest, points_per_env, edges_per_env, color, relaxation],
                device="cpu",
            )
    return pos.numpy()


def _chords(positions):
    return np.linalg.norm(np.diff(np.asarray(positions), axis=0), axis=1)


def test_bunched_and_stretched_chords_converge_to_rest():
    """The measured pathology, reproduced and repaired.

    Particles are seeded with the spread the solver actually produces -- some
    edges collapsed to a few percent of rest, one stretched several times over.
    """
    rest = 0.00758
    count = 12
    # Cumulative arc length with the observed distortion baked in.
    factors = np.array([0.07, 0.07, 3.6, 1.0, 0.1, 1.0, 2.4, 0.2, 1.0, 1.0, 0.9])
    x = np.concatenate([[0.0], np.cumsum(factors * rest)])
    positions = np.stack([x, np.zeros(count), np.zeros(count)], axis=-1)

    before = _chords(positions) / rest
    assert before.min() < 0.2 and before.max() > 3.0, "seed must contain the real distortion"

    result = _sweep(
        positions,
        np.ones(count),
        np.full(count - 1, rest),
        num_envs=1,
        iterations=200,
    )
    after = _chords(result) / rest
    assert abs(after.min() - 1.0) < 0.02, f"shortest chord still {after.min():.3f} of rest"
    assert abs(after.max() - 1.0) < 0.02, f"longest chord still {after.max():.3f} of rest"


def test_a_pinned_root_is_not_moved():
    """A prescribed root must stay where its mount put it.

    Zero inverse mass is how the arm-driven drive holds the proximal particle, so
    a sweep that shifted it would fight the hand pose every step.
    """
    rest = 0.01
    positions = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.06, 0.0, 0.0]])
    inv_masses = np.array([0.0, 1.0, 1.0])

    result = _sweep(positions, inv_masses, np.full(2, rest), num_envs=1, iterations=200)

    np.testing.assert_allclose(result[0], positions[0], atol=1e-7)
    np.testing.assert_allclose(_chords(result), [rest, rest], atol=1e-5)


def test_an_already_correct_chain_is_left_alone():
    """No drift when there is nothing to fix, so it is safe to always run."""
    rest = 0.00758
    count = 10
    positions = np.stack([np.arange(count) * rest, np.zeros(count), np.zeros(count)], axis=-1)
    result = _sweep(positions, np.ones(count), np.full(count - 1, rest), num_envs=1, iterations=50)
    np.testing.assert_allclose(result, positions, atol=1e-6)


def test_a_collapsed_edge_does_not_produce_nans():
    """Coincident particles have no direction to separate along.

    Containment can drive two particles onto the same point, and a divide by the
    zero distance there would poison the whole rod.
    """
    rest = 0.01
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    result = _sweep(positions, np.ones(3), np.full(2, rest), num_envs=1, iterations=20)
    assert np.all(np.isfinite(result))


def test_environments_do_not_leak_into_each_other():
    """Batched indexing must not join the last particle of one rod to the next.

    The kernel is launched per environment and indexes a flat buffer, so an
    off-by-one in the stride would silently weld neighbouring rods together.
    """
    rest = 0.01
    per_env = 4
    first = np.stack([np.arange(per_env) * rest, np.zeros(per_env), np.zeros(per_env)], -1)
    # Second rod far away and badly spaced; repairing it must not drag the first.
    second = np.stack([np.arange(per_env) * rest * 3.0, np.full(per_env, 5.0), np.zeros(per_env)], -1)
    positions = np.concatenate([first, second])

    result = _sweep(
        positions,
        np.ones(2 * per_env),
        np.full(2 * (per_env - 1), rest),
        num_envs=2,
        iterations=200,
    )

    np.testing.assert_allclose(result[:per_env], first, atol=1e-6)
    np.testing.assert_allclose(_chords(result[per_env:]), np.full(per_env - 1, rest), atol=1e-5)


def test_relaxation_below_one_still_converges_only_slower():
    """Under-relaxation is a rate control, not a different fixed point."""
    rest = 0.01
    count = 8
    positions = np.stack([np.arange(count) * rest * 2.0, np.zeros(count), np.zeros(count)], axis=-1)
    args = (positions, np.ones(count), np.full(count - 1, rest))
    gentle = _chords(_sweep(*args, num_envs=1, iterations=8, relaxation=0.3)) / rest
    full = _chords(_sweep(*args, num_envs=1, iterations=8, relaxation=1.0)) / rest

    assert abs(full.mean() - 1.0) < abs(gentle.mean() - 1.0)
    assert abs(_chords(_sweep(*args, num_envs=1, iterations=400, relaxation=0.3)).mean() / rest - 1.0) < 0.02
