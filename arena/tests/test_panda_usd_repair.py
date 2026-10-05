# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Panda repair passes, exercised against real USD stages.

These run on in-memory stages rather than the shipped asset, which is remote and
needs Kit's resolver to open. The stages are built to reproduce the three things
about the real bundle that matter: colliders applied to every visual mesh, two
wrist joints authored with parent and child swapped, and a camera rig whose
screen is a zero-thickness mesh.

Worth having as real stages rather than source inspection because the collider
strip is load-bearing for stability, not just tidiness. The crash it addresses
lives in ``_FinalizeCollisionDescs<UsdPhysicsMeshShapeDesc>``, which only runs
over prims that still carry ``CollisionAPI``, so "did the API actually go away"
is the property under test and a mocked stage could not answer it.
"""

from __future__ import annotations

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from i4h_arena.embodiments.panda_usd_repair import (
    anchor_root_to_world,
    orient_joints_away_from_root,
    prune_unused_wrist_hardware,
    repair_layer_path,
    strip_collision_geometry,
)

ROOT = "/panda"
LINKS = ("panda_link0", "panda_link1", "panda_link2")


def add_mesh(stage: Usd.Stage, path: str, *, collider: bool) -> Usd.Prim:
    """A single-quad mesh, optionally declared as a collider."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    prim = mesh.GetPrim()
    if collider:
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    return prim


@pytest.fixture
def stage() -> Usd.Stage:
    """A miniature Panda: three links, two joints, colliders on every mesh."""
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim(ROOT, "Xform")

    for name in LINKS:
        body = stage.DefinePrim(f"{ROOT}/{name}", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).GetMassAttr().Set(2.5)
        add_mesh(stage, f"{ROOT}/{name}/Visual/Mesh", collider=True)

    for index in (1, 2):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ROOT}/joint{index}")
        joint.GetBody0Rel().SetTargets([f"{ROOT}/{LINKS[index - 1]}"])
        joint.GetBody1Rel().SetTargets([f"{ROOT}/{LINKS[index]}"])

    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT))
    return stage


def colliders(stage: Usd.Stage) -> list[str]:
    return [
        prim.GetPath().pathString
        for prim in Usd.PrimRange.AllPrims(stage.GetPseudoRoot())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def meshes(stage: Usd.Stage) -> list[str]:
    return [
        prim.GetPath().pathString for prim in Usd.PrimRange.AllPrims(stage.GetPseudoRoot()) if prim.IsA(UsdGeom.Mesh)
    ]


# -- stripping collision geometry ----------------------------------------


def test_every_collider_is_removed(stage) -> None:
    """The parallel finalize loop iterates this set, so it has to be empty."""
    assert colliders(stage)
    strip_collision_geometry([stage.GetDefaultPrim()])
    assert colliders(stage) == []


def test_the_mesh_specific_schema_goes_too(stage) -> None:
    """MeshCollisionAPI only qualifies a collider; left behind it is a lie."""
    strip_collision_geometry([stage.GetDefaultPrim()])
    for path in meshes(stage):
        assert not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.MeshCollisionAPI)


def test_the_visual_geometry_survives(stage) -> None:
    """The arm still has to render; only the physics opinion is dropped."""
    before = meshes(stage)
    strip_collision_geometry([stage.GetDefaultPrim()])
    assert meshes(stage) == before


def test_mass_survives(stage) -> None:
    """Inertia falls back to collision geometry when MassAPI is absent, so
    removing colliders must not also remove the authored mass."""
    strip_collision_geometry([stage.GetDefaultPrim()])
    for name in LINKS:
        prim = stage.GetPrimAtPath(f"{ROOT}/{name}")
        assert prim.HasAPI(UsdPhysics.MassAPI)
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(2.5)


def test_the_joints_survive(stage) -> None:
    strip_collision_geometry([stage.GetDefaultPrim()])
    joints = [p for p in Usd.PrimRange.AllPrims(stage.GetPseudoRoot()) if p.IsA(UsdPhysics.Joint)]
    assert len(joints) == 2


def test_the_count_reports_what_was_removed(stage) -> None:
    """Lets a caller tell "nothing to do" from "found nothing to inspect"."""
    assert strip_collision_geometry([stage.GetDefaultPrim()]) == len(LINKS)


def test_stripping_twice_is_a_no_op(stage) -> None:
    """The bake is cached, so a re-bake over a stripped stage must be safe."""
    strip_collision_geometry([stage.GetDefaultPrim()])
    assert strip_collision_geometry([stage.GetDefaultPrim()]) == 0


def test_a_stage_with_no_colliders_is_left_alone() -> None:
    empty = Usd.Stage.CreateInMemory()
    empty.DefinePrim(ROOT, "Xform")
    add_mesh(empty, f"{ROOT}/Visual/Mesh", collider=False)
    assert strip_collision_geometry([empty.GetPrimAtPath(ROOT)]) == 0


# -- interaction with the other two passes -------------------------------


def test_reversed_joints_are_still_turned_around_after_stripping(stage) -> None:
    """The passes run in sequence over one stage; neither may undo the other."""
    joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"{ROOT}/joint2"))
    joint.GetBody0Rel().SetTargets([f"{ROOT}/{LINKS[2]}"])
    joint.GetBody1Rel().SetTargets([f"{ROOT}/{LINKS[1]}"])

    strip_collision_geometry([stage.GetDefaultPrim()])
    assert orient_joints_away_from_root([stage.GetDefaultPrim()]) == 1
    assert [t.pathString for t in joint.GetBody0Rel().GetTargets()] == [f"{ROOT}/{LINKS[1]}"]


def test_pruned_hardware_is_not_counted_as_stripped(stage) -> None:
    """Pruning runs first, so the camera's colliders are already gone and the
    strip must not resurrect or recount them."""
    rig = stage.DefinePrim(f"{ROOT}/D405_rigid", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(rig)
    add_mesh(stage, f"{ROOT}/D405_rigid/Visual/screen/Mesh", collider=True)

    assert prune_unused_wrist_hardware([stage.GetDefaultPrim()])
    stripped = strip_collision_geometry([stage.GetDefaultPrim()])
    assert stripped == len(LINKS)


# -- grounding the root --------------------------------------------------


def free_root_joint(stage: Usd.Stage) -> Usd.Prim:
    """The root joint as the real bundle authors it.

    A generic ``PhysicsJoint`` -- a D6 with every axis free, which is a
    floating base spelled differently -- and named backwards, with the body on
    ``body0`` and the world end left empty.
    """
    prim = stage.DefinePrim(f"{ROOT}/rootJoint", "PhysicsJoint")
    joint = UsdPhysics.Joint(prim)
    joint.GetBody0Rel().SetTargets([f"{ROOT}/{LINKS[0]}"])
    joint.GetBody1Rel().SetTargets([])
    return prim


def root_joints(stage: Usd.Stage) -> list[Usd.Prim]:
    return [
        prim
        for prim in Usd.PrimRange.AllPrims(stage.GetPseudoRoot())
        if prim.IsA(UsdPhysics.Joint)
        and len(UsdPhysics.Joint(prim).GetBody0Rel().GetTargets())
        + len(UsdPhysics.Joint(prim).GetBody1Rel().GetTargets())
        == 1
    ]


def test_the_free_root_joint_becomes_a_fixed_one(stage) -> None:
    """The whole point: a free D6 root is what makes Newton build a floating base."""
    prim = free_root_joint(stage)

    assert anchor_root_to_world([stage.GetDefaultPrim()]) == 1

    assert stage.GetPrimAtPath(prim.GetPath()).IsA(UsdPhysics.FixedJoint)


def test_the_existing_root_joint_is_retyped_rather_than_duplicated(stage) -> None:
    """Newton refuses to merge two roots, so adding one beside it fails the build.

    Reproduces a real failure: leaving the shipped joint in place and defining
    a second one aborted model building with "Cannot merge joint .../rootJoint
    of type D6Joint into a D6 joint".
    """
    free_root_joint(stage)

    anchor_root_to_world([stage.GetDefaultPrim()])

    assert len(root_joints(stage)) == 1


def test_the_grounded_end_is_named_on_body1(stage) -> None:
    """An unset body0 is the world; naming the body there instead grounds nothing."""
    free_root_joint(stage)

    anchor_root_to_world([stage.GetDefaultPrim()])

    joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"{ROOT}/rootJoint"))
    assert list(joint.GetBody0Rel().GetTargets()) == []
    assert [t.pathString for t in joint.GetBody1Rel().GetTargets()] == [f"{ROOT}/{LINKS[0]}"]


def test_a_root_joint_is_created_when_the_asset_has_none(stage) -> None:
    """A re-export that simply drops the joint must still ground the arm."""
    assert anchor_root_to_world([stage.GetDefaultPrim()]) == 1

    grounded = root_joints(stage)
    assert len(grounded) == 1
    assert grounded[0].IsA(UsdPhysics.FixedJoint)


def test_competing_root_joints_are_refused(stage) -> None:
    """Grounding one would leave the other contradicting it."""
    free_root_joint(stage)
    second = UsdPhysics.FixedJoint.Define(stage, f"{ROOT}/otherRoot")
    second.GetBody1Rel().SetTargets([f"{ROOT}/{LINKS[1]}"])

    with pytest.raises(RuntimeError, match="2 root joints"):
        anchor_root_to_world([stage.GetDefaultPrim()])


def test_a_renamed_root_body_is_refused() -> None:
    """Guessing would ground the wrong link and silently misplace every DoF."""
    odd = Usd.Stage.CreateInMemory()
    odd.DefinePrim(ROOT, "Xform")
    UsdPhysics.RigidBodyAPI.Apply(odd.DefinePrim(f"{ROOT}/base_link", "Xform"))

    with pytest.raises(RuntimeError, match="no body named 'panda_link0'"):
        anchor_root_to_world([odd.GetPrimAtPath(ROOT)])


def test_grounding_twice_is_stable(stage) -> None:
    """The bake is cached, so a re-bake must not accumulate roots."""
    free_root_joint(stage)

    anchor_root_to_world([stage.GetDefaultPrim()])
    anchor_root_to_world([stage.GetDefaultPrim()])

    assert len(root_joints(stage)) == 1


def test_the_root_joint_is_not_turned_around_by_the_reorient_pass(stage) -> None:
    """It names one body by design, which the reorient pass has to leave alone.

    If that pass claimed it, the arm's declared root would be rewritten to
    point at the world and the grounding would be undone.
    """
    free_root_joint(stage)
    anchor_root_to_world([stage.GetDefaultPrim()])

    assert orient_joints_away_from_root([stage.GetDefaultPrim()]) == 0

    joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"{ROOT}/rootJoint"))
    assert [t.pathString for t in joint.GetBody1Rel().GetTargets()] == [f"{ROOT}/{LINKS[0]}"]


# -- the cached bake -----------------------------------------------------


def test_the_bake_is_a_binary_crate() -> None:
    """It holds a flattened copy of an 85 MB asset, not a few overrides."""
    assert repair_layer_path().suffix == ".usdc"


def test_the_bake_path_is_keyed_by_the_repair_version() -> None:
    """A stale bake from before the collider strip must never be reused."""
    from i4h_arena.embodiments import panda_usd_repair as repair

    first = repair.repair_layer_path()
    original = repair.REPAIR_LAYER_VERSION
    try:
        repair.REPAIR_LAYER_VERSION = original + 1
        assert repair.repair_layer_path() != first
    finally:
        repair.REPAIR_LAYER_VERSION = original
