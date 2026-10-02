# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for two-sided centerline containment.

Containment used to be a one-sided radial test: a particle outside the wall was
projected back, and a particle anywhere inside was corrected by exactly nothing.
That is why sweeping the distance cleanup harder relocated the shaft bend rather
than removing it -- excess arc length has to go somewhere, and with no
preference for the axis anywhere inside the lumen, sideways was free.

These drive the real Warp kernel on the CPU device rather than a
reimplementation, so the deadband, the no-overshoot bound, and the promise that
an interior pull is never reported as wall contact are covered as shipped.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")

# Below the skip guard on purpose: the solver package imports warp, so a machine
# without it has to skip rather than error at collection.
from catheter_vasculature_solver.vessel_deformation.centerline_containment import (  # noqa: E402
    apply_centerline_corrections_kernel,
    project_centerline_containment_batched_kernel,
    project_centerline_containment_kernel,
)

# A straight tube of uniform radius along +x, which makes "distance from the
# axis" the y coordinate and every expectation below a number you can check by
# hand.
_AXIS_NODES = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32)
_EDGES = np.array([[0, 1], [1, 2]], dtype=np.int32)
_TUBE_RADIUS_M = 0.010
_CATHETER_RADIUS_M = 0.001
#: Radius the wire's centre is free to occupy before its surface meets the wall.
FREE_RADIUS_M = _TUBE_RADIUS_M - _CATHETER_RADIUS_M


@pytest.fixture(scope="module", autouse=True)
def _warp_cpu(tmp_path_factory):
    """Initialize Warp with a writable kernel cache.

    The default cache lives under the user's home directory, which is not
    writable in every environment these run in; without this the failure looks
    like a file of broken tests rather than one unwritable path.
    """
    wp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("warp_cache"))
    wp.init()


def _contain(offset_m: float, *, deadband: float, stiffness: float, two_way: bool = False):
    """Project one particle sitting ``offset_m`` off the axis.

    Returns ``(moved_offset_m, contact_depth_m, contact_count, vessel_moved)``,
    which between them answer where the particle went, whether the run called it
    a wall contact, and whether the wall was pushed.
    """
    # One catheter particle, no catheter edges, so the launch covers exactly the
    # sample under test and nothing interpolated.
    point = np.array([[0.01, offset_m, 0.0]], dtype=np.float32)
    catheter = wp.array(point, dtype=wp.vec3, device="cpu")
    catheter_inv = wp.array(np.ones(1, dtype=np.float32), dtype=wp.float32, device="cpu")
    centerline = wp.array(_AXIS_NODES, dtype=wp.vec3, device="cpu")
    edges = wp.array(_EDGES, dtype=wp.vec2i, device="cpu")
    radii = wp.array(np.full(3, _TUBE_RADIUS_M, dtype=np.float32), dtype=wp.float32, device="cpu")
    # A movable wall, so "the vessel was not pushed" is a real observation rather
    # than a locked node that could not have moved anyway.
    vessel_inv = wp.array(np.ones(3, dtype=np.float32), dtype=wp.float32, device="cpu")

    cath_corr = wp.zeros(1, dtype=wp.vec3, device="cpu")
    cath_counts = wp.zeros(1, dtype=wp.float32, device="cpu")
    vessel_corr = wp.zeros(3, dtype=wp.vec3, device="cpu")
    vessel_counts = wp.zeros(3, dtype=wp.float32, device="cpu")
    depth = wp.zeros(1, dtype=wp.float32, device="cpu")
    count = wp.zeros(1, dtype=wp.int32, device="cpu")

    wp.launch(
        project_centerline_containment_kernel,
        dim=1,
        inputs=[
            catheter,
            catheter_inv,
            1,
            centerline,
            edges,
            radii,
            vessel_inv,
            -1,
            -1,
            _CATHETER_RADIUS_M,
            1.0,
            deadband,
            stiffness,
            1 if two_way else 0,
            1.0,
            cath_corr,
            cath_counts,
            vessel_corr,
            vessel_counts,
            depth,
            count,
        ],
        device="cpu",
    )
    wp.launch(
        apply_centerline_corrections_kernel,
        dim=1,
        inputs=[catheter, catheter_inv, cath_corr, cath_counts],
        device="cpu",
    )
    vessel_moved = bool(np.any(np.abs(vessel_corr.numpy()) > 1.0e-9))
    return float(catheter.numpy()[0][1]), float(depth.numpy()[0]), int(count.numpy()[0]), vessel_moved


# --------------------------------------------------------------------------- #
# The deadband
# --------------------------------------------------------------------------- #


def test_a_wire_inside_the_deadband_is_still_left_alone():
    """The point of a deadband: the inner lumen stays somewhere the wire may be.

    Without this the pull would be prescribing the route rather than bounding
    it, and the shape near the axis would be the centerline's answer instead of
    the solve's.
    """
    offset = 0.4 * FREE_RADIUS_M

    moved, _depth, count, _vessel = _contain(offset, deadband=0.5, stiffness=1.0)

    assert moved == pytest.approx(offset)
    assert count == 0


def test_a_wire_past_the_deadband_is_pulled_back_toward_the_axis():
    offset = 0.8 * FREE_RADIUS_M

    moved, _depth, _count, _vessel = _contain(offset, deadband=0.5, stiffness=1.0)

    assert moved < offset


def test_a_full_strength_pull_stops_exactly_on_the_deadband():
    """The bound that makes stiffness safe to raise.

    At unit stiffness the correction is exactly the overshoot past the deadband,
    so the sample lands on that surface and no stiffness in ``(0, 1]`` can carry
    it further -- past the deadband, through the axis, or out the far side.
    """
    moved, _depth, _count, _vessel = _contain(0.9 * FREE_RADIUS_M, deadband=0.5, stiffness=1.0)

    assert moved == pytest.approx(0.5 * FREE_RADIUS_M, abs=1.0e-6)


def test_a_partial_pull_moves_part_of_the_way():
    offset = 0.9 * FREE_RADIUS_M
    deadband_m = 0.5 * FREE_RADIUS_M

    moved, _depth, _count, _vessel = _contain(offset, deadband=0.5, stiffness=0.25)

    assert moved == pytest.approx(offset - 0.25 * (offset - deadband_m), abs=1.0e-6)


def test_the_pull_never_crosses_the_axis():
    """A correction that overshot the axis would flip the wire's bend direction."""
    for fraction in (0.55, 0.7, 0.85, 0.99):
        moved, _depth, _count, _vessel = _contain(fraction * FREE_RADIUS_M, deadband=0.0, stiffness=1.0)

        assert moved >= 0.0


# --------------------------------------------------------------------------- #
# Staying compatible with the one-sided behaviour
# --------------------------------------------------------------------------- #


def test_zero_stiffness_is_the_old_one_sided_test():
    """The shipped default has to be able to reproduce prior measurements."""
    offset = 0.95 * FREE_RADIUS_M

    moved, _depth, count, _vessel = _contain(offset, deadband=0.5, stiffness=0.0)

    assert moved == pytest.approx(offset)
    assert count == 0


def test_a_wire_through_the_wall_is_still_projected_back_in_full():
    """Interior work must not have softened the contact that was already right."""
    outside = FREE_RADIUS_M + 0.002

    moved, depth, count, _vessel = _contain(outside, deadband=0.5, stiffness=0.25)

    assert moved == pytest.approx(FREE_RADIUS_M, abs=1.0e-6)
    assert depth == pytest.approx(0.002, abs=1.0e-6)
    assert count == 1


# --------------------------------------------------------------------------- #
# Keeping the probe and the wall honest
# --------------------------------------------------------------------------- #


def test_an_interior_pull_is_not_reported_as_penetration():
    """Otherwise every containment number we have measured becomes unreadable.

    ``contact_depth`` is what the probe prints as worst penetration. Counting a
    pull that happened well inside the lumen would report a wall contact that
    never occurred.
    """
    _moved, depth, count, _vessel = _contain(0.9 * FREE_RADIUS_M, deadband=0.5, stiffness=1.0)

    assert depth == pytest.approx(0.0)
    assert count == 0


def test_an_interior_pull_does_not_deform_the_vessel():
    """It is a regulariser on the wire, not a force the wall exerts.

    A wall that moved here would be responding to a contact that never happened,
    and under two-way coupling that error feeds straight back into the rod.
    """
    _moved, _depth, _count, vessel_moved = _contain(0.9 * FREE_RADIUS_M, deadband=0.5, stiffness=1.0, two_way=True)

    assert not vessel_moved


def test_a_real_contact_still_deforms_the_vessel():
    """The counterpart, so the check above cannot pass by breaking two-way."""
    _moved, _depth, _count, vessel_moved = _contain(FREE_RADIUS_M + 0.002, deadband=0.5, stiffness=1.0, two_way=True)

    assert vessel_moved


# --------------------------------------------------------------------------- #
# The batched kernel
# --------------------------------------------------------------------------- #


def _contain_batched(offsets_m, *, deadband: float, stiffness: float):
    """Project one particle per env through the multi-env kernel.

    The batched kernel took the same two new arguments, and both launches pass
    them positionally, so a mismatch here would compile and then silently read
    the deadband out of ``two_way``. Only a multi-env run would show that, which
    is not where anyone is looking.
    """
    num_envs = len(offsets_m)
    points = np.array([[0.01, offset, 0.0] for offset in offsets_m], dtype=np.float32)
    catheter = wp.array(points, dtype=wp.vec3, device="cpu")
    catheter_inv = wp.array(np.ones(num_envs, dtype=np.float32), dtype=wp.float32, device="cpu")
    # Nodes and edges are tiled per env, and the kernel's edges carry global node
    # indices, exactly as CosseratRod lays them out.
    nodes = np.tile(_AXIS_NODES, (num_envs, 1))
    edges = np.concatenate([_EDGES + env * len(_AXIS_NODES) for env in range(num_envs)])
    centerline = wp.array(nodes, dtype=wp.vec3, device="cpu")
    edge_array = wp.array(edges, dtype=wp.vec2i, device="cpu")
    radii = wp.array(np.full(len(nodes), _TUBE_RADIUS_M, dtype=np.float32), dtype=wp.float32, device="cpu")
    vessel_inv = wp.array(np.ones(len(nodes), dtype=np.float32), dtype=wp.float32, device="cpu")

    cath_corr = wp.zeros(num_envs, dtype=wp.vec3, device="cpu")
    cath_counts = wp.zeros(num_envs, dtype=wp.float32, device="cpu")
    vessel_corr = wp.zeros(len(nodes), dtype=wp.vec3, device="cpu")
    vessel_counts = wp.zeros(len(nodes), dtype=wp.float32, device="cpu")
    depth = wp.zeros(num_envs, dtype=wp.float32, device="cpu")
    count = wp.zeros(num_envs, dtype=wp.int32, device="cpu")

    wp.launch(
        project_centerline_containment_batched_kernel,
        dim=(num_envs, 1),
        inputs=[
            catheter,
            catheter_inv,
            1,
            centerline,
            edge_array,
            radii,
            vessel_inv,
            len(_AXIS_NODES),
            len(_EDGES),
            -1,
            -1,
            _CATHETER_RADIUS_M,
            1.0,
            deadband,
            stiffness,
            0,
            1.0,
            cath_corr,
            cath_counts,
            vessel_corr,
            vessel_counts,
            depth,
            count,
        ],
        device="cpu",
    )
    wp.launch(
        apply_centerline_corrections_kernel,
        dim=num_envs,
        inputs=[catheter, catheter_inv, cath_corr, cath_counts],
        device="cpu",
    )
    return catheter.numpy()[:, 1], count.numpy()


def test_the_batched_kernel_applies_the_same_deadband_per_env():
    inside = 0.4 * FREE_RADIUS_M
    past_deadband = 0.9 * FREE_RADIUS_M

    moved, counts = _contain_batched([inside, past_deadband], deadband=0.5, stiffness=1.0)

    assert moved[0] == pytest.approx(inside)
    assert moved[1] == pytest.approx(0.5 * FREE_RADIUS_M, abs=1.0e-6)
    # Neither env touched a wall, so neither may claim a contact.
    assert list(counts) == [0, 0]


def test_the_batched_kernel_still_projects_a_wall_contact():
    moved, counts = _contain_batched([FREE_RADIUS_M + 0.002], deadband=0.5, stiffness=0.25)

    assert moved[0] == pytest.approx(FREE_RADIUS_M, abs=1.0e-6)
    assert list(counts) == [1]


# --------------------------------------------------------------------------- #
# Alternating containment with the cleanup sweeps
# --------------------------------------------------------------------------- #


class _Schedule:
    """Records the order containment and cleanup were actually run in.

    The scheduling is the fix here, so it is what these assert on. Driving the
    real solver would need a vessel, a rod and a device; the ordering it asks
    for is a property of the method itself.
    """

    def __init__(self, *, iterations: int, rounds: int) -> None:
        self.containment_cleanup_iterations = iterations
        self.containment_cleanup_rounds = rounds
        self.calls: list[tuple[str, int]] = []

    def _project_distance_cleanup(self, *_args, iterations=None, **_kwargs) -> None:
        sweeps = self.containment_cleanup_iterations if iterations is None else int(iterations)
        self.calls.append(("cleanup", sweeps))

    def run(self) -> None:
        from catheter_vasculature_solver.cath_rod_solver import CathRodSolver

        CathRodSolver._project_containment_and_cleanup(
            self,
            lambda: self.calls.append(("containment", 0)),
            None,
            None,
            None,
            num_envs=1,
            points_per_env=41,
            edges_per_env=40,
            dev="cpu",
        )


def test_a_single_round_is_the_original_sequencing():
    """So prior measurements stay reproducible and the A/B is honest."""
    schedule = _Schedule(iterations=128, rounds=1)

    schedule.run()

    assert schedule.calls == [("containment", 0), ("cleanup", 128)]


def test_the_rounds_alternate_containment_with_cleanup():
    schedule = _Schedule(iterations=128, rounds=4)

    schedule.run()

    assert [name for name, _ in schedule.calls] == ["containment", "cleanup"] * 4


def test_the_sweep_budget_is_divided_across_rounds_not_multiplied():
    """Alternating buys agreement between the two, not more cleanup.

    If rounds multiplied the budget instead, raising the round count would be
    confounded with simply sweeping harder and no comparison would mean anything.
    """
    schedule = _Schedule(iterations=128, rounds=8)

    schedule.run()

    sweeps = [count for name, count in schedule.calls if name == "cleanup"]
    assert sweeps == [16] * 8
    assert sum(sweeps) == 128


def test_an_uneven_budget_still_spends_every_sweep():
    """A remainder silently dropped would make the totals disagree with the knob."""
    schedule = _Schedule(iterations=130, rounds=4)

    schedule.run()

    sweeps = [count for name, count in schedule.calls if name == "cleanup"]
    assert sum(sweeps) == 130
    assert sweeps == [33, 33, 32, 32]


def test_containment_still_runs_when_there_is_nothing_to_contain_against():
    """A scene with no vessel must still get its cleanup rather than nothing."""
    schedule = _Schedule(iterations=64, rounds=4)

    from catheter_vasculature_solver.cath_rod_solver import CathRodSolver

    CathRodSolver._project_containment_and_cleanup(
        schedule,
        None,
        None,
        None,
        None,
        num_envs=1,
        points_per_env=41,
        edges_per_env=40,
        dev="cpu",
    )

    assert [name for name, _ in schedule.calls] == ["cleanup"] * 4
