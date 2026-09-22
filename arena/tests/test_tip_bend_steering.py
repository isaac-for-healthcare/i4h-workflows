# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for steering the catheter tip at runtime.

The tip's shape is carried in the bend constraint's *rest* state, which makes it
unlike the other two commands: insertion and rotation are rates the solver
spends, while the bend is a shape it holds. Two things follow, and both are the
kind of error that produces a plausible-looking wire rather than a failure.

The first is that the update has to add to the authored rest curvature instead of
writing over the buffer. The kernel used to zero every non-tip edge, so steering
would quietly erase a body shape seeded from the vessel path.

The second is that the angle has to be per environment. A scalar shared by every
env is fine for a fixed pre-bend but silently couples a batched rollout, where
each env is steering on its own action.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from i4h_arena.medical.newton_catheter_physics import tip_bend_rest_component

import yaml

ARENA = Path(__file__).parents[1] / "i4h_arena"
PLAIN_CATHETER = ARENA / "embodiments" / "catheter.py"
ARM_CATHETER = ARENA / "embodiments" / "franka_catheter.py"
PLAIN_MANIFEST = ARENA / "embodiments" / "manifest" / "catheter.yaml"
ARM_MANIFEST = ARENA / "embodiments" / "manifest" / "franka_catheter.yaml"
SCENE_MANIFESTS = (
    ARENA / "scenes" / "manifest" / "endoluminal_navigation.yaml",
    ARENA / "scenes" / "manifest" / "endoluminal_navigation_arm.yaml",
)

#: Where the bend sits in the action vector. Between rotation and the C-arm, so
#: the instrument's own commands stay contiguous.
TIP_BEND_INDEX = 2


def class_default(source_path: Path, class_name: str, field: str) -> Any:
    """One annotated class attribute default, read without importing Isaac Sim."""
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for statement in node.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == field
                    and statement.value is not None
                ):
                    return ast.literal_eval(statement.value)
    raise AssertionError(f"{class_name}.{field} not found in {source_path.name}")


def action_dim(source_path: Path, class_name: str) -> int:
    """The ``action_dim`` property's literal return, for a term we cannot import."""
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for statement in node.body:
                if isinstance(statement, ast.FunctionDef) and statement.name == "action_dim":
                    for inner in ast.walk(statement):
                        if isinstance(inner, ast.Return) and inner.value is not None:
                            return int(ast.literal_eval(inner.value))
    raise AssertionError(f"{class_name}.action_dim not found in {source_path.name}")


# --------------------------------------------------------------------------- #
# The command contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source_path, class_name",
    [
        (PLAIN_CATHETER, "CatheterVelocityAction"),
        (ARM_CATHETER, "ArmDrivenCatheterAction"),
    ],
)
def test_both_catheter_terms_carry_a_third_command(source_path: Path, class_name: str) -> None:
    """Insertion, rotation, and now the tip bend, on both drives.

    One keyboard maps both embodiments, so a term that stayed two wide would
    read the C-arm's orbit as its own bend command.
    """
    assert action_dim(source_path, class_name) == 3


@pytest.mark.parametrize("field", ["max_tip_bend_rate_radps", "max_tip_bend_rad"])
def test_both_drives_share_one_tip_bend_ceiling(field: str) -> None:
    """A mismatch would clip whichever embodiment held the lower ceiling."""
    plain = class_default(PLAIN_CATHETER, "CatheterVelocityActionCfg", field)
    arm = class_default(ARM_CATHETER, "ArmDrivenCatheterActionCfg", field)

    assert plain == pytest.approx(arm)


def test_a_full_deflection_takes_about_a_second() -> None:
    """The rate ceiling is paced against the angle limit on purpose.

    The bend is imposed on the rest state, so stepping it faster than the solve
    relaxes asks the wire to snap to a new shape rather than curl into it.
    """
    rate = class_default(PLAIN_CATHETER, "CatheterVelocityActionCfg", "max_tip_bend_rate_radps")
    limit = class_default(PLAIN_CATHETER, "CatheterVelocityActionCfg", "max_tip_bend_rad")

    assert 0.5 <= limit / rate <= 2.0


# --------------------------------------------------------------------------- #
# The recorded contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("manifest_path", [PLAIN_MANIFEST, ARM_MANIFEST])
def test_the_bend_is_named_in_both_the_action_and_the_state(manifest_path: Path) -> None:
    """A commanded bend the recording cannot see is untrainable.

    A policy that steers has to observe the shape it asked for, so the angle is
    reported as a virtual joint next to insertion and rotation.
    """
    manifest = yaml.safe_load(manifest_path.read_text())

    assert manifest["action_names"][TIP_BEND_INDEX] == "tip_bend_rate_radps"
    assert manifest["state_names"][TIP_BEND_INDEX] == "tip_bend_rad"


def test_the_carm_still_comes_after_the_instrument() -> None:
    """Ordering is the recording's compatibility story, so it is pinned.

    The instrument's columns lead and the gantry follows, which is what lets a
    recording made on the arm keep the plain scene's leading columns.
    """
    manifest = yaml.safe_load(PLAIN_MANIFEST.read_text())

    for names in (manifest["action_names"], manifest["state_names"]):
        assert names[-1].startswith("carm_")
    for splits in (manifest["state_split"], manifest["action_split"]):
        groups = {name: (start, end) for name, start, end in splits}
        assert groups["catheter"] == (0, 3)
        assert groups["carm"] == (3, 4)


@pytest.mark.parametrize("manifest_path", SCENE_MANIFESTS)
def test_the_scene_advertises_the_widened_action(manifest_path: Path) -> None:
    """``dof`` is what a task's ``requires`` is checked against."""
    assert yaml.safe_load(manifest_path.read_text())["dof"] == 4


# --------------------------------------------------------------------------- #
# The kernel that imposes the shape
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def kernels():
    """The solver's tip-bend kernels, skipped where the solver is not installed."""
    pytest.importorskip("warp")
    solver = pytest.importorskip("catheter_vasculature_solver.cath_rod_solver")
    return solver


def _contain(kernels, baseline_np: np.ndarray, angles_np: np.ndarray, edges: int, tip: int) -> np.ndarray:
    """Run the batched kernel and return ``(envs, edges, 3)`` rest curvature."""
    import warp as wp

    envs = angles_np.size
    baseline = wp.array(baseline_np.reshape(-1, 3), dtype=wp.vec3, device="cpu")
    rest = wp.zeros(envs * edges, dtype=wp.vec3, device="cpu")
    wp.launch(
        kernels._update_tip_rest_darboux_batched_kernel,  # noqa: SLF001
        dim=(envs, edges),
        inputs=[rest, baseline, edges, tip, wp.array(angles_np, dtype=wp.float32, device="cpu")],
        device="cpu",
    )
    return rest.numpy().reshape(envs, edges, 3)


def _ramped_baseline(envs: int, edges: int) -> np.ndarray:
    """Authored curvature that is non-zero and varies per edge.

    Non-uniform on purpose: a kernel that overwrote the buffer with a single
    value would still match a flat baseline on the tip edges.
    """
    baseline = np.zeros((envs, edges, 3), dtype=np.float32)
    for env in range(envs):
        for edge in range(edges):
            baseline[env, edge] = (0.01 * (edge + 1), 0.02, -0.03)
    return baseline


def test_steering_leaves_the_authored_body_shape_alone(kernels) -> None:
    """The regression this was written for.

    The kernel used to write ``vec3(0, 0, 0)`` over every non-tip edge, so any
    rest curvature seeded from the vessel path was erased on the first steer.
    """
    edges, tip, envs = 10, 4, 3
    baseline = _ramped_baseline(envs, edges)
    out = _contain(kernels, baseline, np.array([0.8, -0.4, 0.0], dtype=np.float32), edges, tip)

    np.testing.assert_allclose(out[:, : edges - tip], baseline[:, : edges - tip], atol=1e-6)


def test_the_bend_is_spread_over_the_tip_edges(kernels) -> None:
    """Each tip edge takes an equal share, added to what it already carried.

    The share is ``sin(angle / (2n - 1))``, not ``angle / n``: ``rest_darboux``
    turns the frames while the polyline follows the midpoints between them, so
    the last half-hinge of frame rotation falls past the final segment.
    Asserting against the host helper also pins kernel and host to one mapping.
    """
    edges, tip = 10, 4
    baseline = _ramped_baseline(1, edges)
    angle = 0.8
    out = _contain(kernels, baseline, np.array([angle], dtype=np.float32), edges, tip)

    added = out[0, edges - tip :] - baseline[0, edges - tip :]
    np.testing.assert_allclose(added[:, 0], tip_bend_rest_component(angle, tip), atol=1e-6)
    # Only the local X component: aiming the bend is the rotation command's job.
    np.testing.assert_allclose(added[:, 1:], 0.0, atol=1e-7)


def test_the_tip_edges_carry_less_than_an_even_share(kernels) -> None:
    """Guards the direction of the correction, which a sign slip would invert.

    Dividing by ``2n - 1`` rather than ``n`` has to make each edge do *less*,
    since the polyline picks up nearly twice the turn the naive reading expects.
    """
    edges, tip, angle = 10, 4, 0.8
    baseline = _ramped_baseline(1, edges)
    out = _contain(kernels, baseline, np.array([angle], dtype=np.float32), edges, tip)

    added = float((out[0, edges - tip :, 0] - baseline[0, edges - tip :, 0])[0])

    assert added < angle / tip
    assert added == pytest.approx(0.114037, abs=1e-5)


def test_each_env_steers_on_its_own_angle(kernels) -> None:
    """A shared scalar would couple every env in a batched rollout."""
    edges, tip = 10, 4
    angles = np.array([0.8, -0.4, 0.0], dtype=np.float32)
    baseline = _ramped_baseline(angles.size, edges)
    out = _contain(kernels, baseline, angles, edges, tip)

    for env, angle in enumerate(angles):
        added = out[env, edges - tip :, 0] - baseline[env, edges - tip :, 0]
        np.testing.assert_allclose(added, tip_bend_rest_component(float(angle), tip), atol=1e-6)


def test_a_zero_angle_is_exactly_the_authored_shape(kernels) -> None:
    """Releasing the key has to leave the rod as authored, not merely close."""
    edges, tip = 10, 4
    baseline = _ramped_baseline(1, edges)
    out = _contain(kernels, baseline, np.array([0.0], dtype=np.float32), edges, tip)

    np.testing.assert_allclose(out[0], baseline[0], atol=1e-7)


def test_steering_is_absolute_rather_than_cumulative(kernels) -> None:
    """Applying the same angle twice must not bend twice as far.

    The term hands over an angle every step, so a kernel that accumulated would
    curl the tip further on each tick a key was held down.
    """
    edges, tip = 10, 4
    baseline = _ramped_baseline(1, edges)
    angles = np.array([0.8], dtype=np.float32)
    once = _contain(kernels, baseline, angles, edges, tip)
    twice = _contain(kernels, baseline, angles, edges, tip)

    np.testing.assert_allclose(twice, once, atol=1e-7)
