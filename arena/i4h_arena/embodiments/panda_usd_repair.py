# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make the shipped Franka Panda asset usable as a Newton articulation.

The bundle's Panda is a PhysX-era CAD import, and three separate things about
it are incompatible with what this workflow needs. Each pass below fixes one,
and :func:`repaired_panda_asset` bakes all three into a local copy once so the
live stage is never edited while Isaac's physics parser is walking it.

Kept apart from ``franka_catheter`` so it can be tested. These passes need only
``pxr``, while the embodiment imports ``isaaclab.sim`` and therefore cannot be
imported at all without Isaac Sim -- which is why the embodiment's own tests
inspect its source as text rather than importing it.
"""

from __future__ import annotations

import hashlib
import os
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pxr import Usd, UsdPhysics

from i4h_arena.assets.constants import PANDA_USD

#: The asset these passes repair.
FRANKA_PANDA_USD = PANDA_USD

#: The articulation root. Joint direction is defined as "away from" this body,
#: which is what lets the repair below decide which end of a joint is the parent.
_ROOT_BODY_NAME = "panda_link0"

#: Wrist-mounted hardware belonging to this bundle's ultrasound workflow. A
#: catheter drive unit does not carry a depth camera, so this is a body and a
#: joint the articulation is carrying for nothing.
#:
#: It was originally pruned for a sharper reason: the camera's screen is a
#: zero-thickness mesh, and MJWarp rejects one outright as a collider::
#:
#:     ValueError: MuJoCo contact generation does not support planar mesh
#:     collider '.../D405_rigid/D405/Visual/screen/Mesh_082'
#:
#: :func:`strip_collision_geometry` has since made that unreachable, since a
#: mesh that is not a collider cannot be a planar one. The pass stays because
#: keeping the articulation minimal is worth a few lines, but it is no longer
#: what stands between this asset and a model that builds.
_UNUSED_WRIST_HARDWARE = ("D405_rigid",)


def prune_unused_wrist_hardware(root_prims: Iterable[Usd.Prim]) -> int:
    """Deactivate the ultrasound camera rig and whatever pins it to the wrist.

    Deactivating rather than deleting keeps the edit to the layer being
    authored, so the referenced asset is untouched. The joints have to go too:
    a joint left pointing at a deactivated body is a dangling reference, so
    they are collected before anything is deactivated and while they can still
    be resolved.

    Returns the number of prims deactivated.
    """
    deactivated = 0
    for root_prim in root_prims:
        doomed = [prim.GetPath() for prim in Usd.PrimRange(root_prim) if prim.GetName() in _UNUSED_WRIST_HARDWARE]
        if not doomed:
            continue

        attachments = []
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            targets = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
            if any(target.HasPrefix(path) for target in targets for path in doomed):
                attachments.append(prim)

        for prim in attachments:
            prim.SetActive(False)
            deactivated += 1
        for path in doomed:
            root_prim.GetStage().GetPrimAtPath(path).SetActive(False)
            deactivated += 1
    return deactivated


def orient_joints_away_from_root(root_prims: Iterable[Usd.Prim]) -> int:
    """Turn around any joint authored with its parent and child swapped.

    This asset was authored for PhysX, which accepts a joint whose declared
    ``body0`` is the end further from the articulation root and silently
    reinterprets it. Newton refuses outright::

        ValueError: Reversed joints are not supported:
          .../panda_link8/panda_hand_joint, .../panda_hand/FixedJoint

    Two of the wrist fixed joints are wired that way, so the arm cannot be
    spawned into a Newton scene as shipped. Rather than vendoring a corrected
    copy of an 85 MB asset, the joints are turned around on the stage between
    spawning and model building.

    The correct direction is not hardcoded per joint. Treating the joints as
    undirected edges gives the body graph, and breadth-first search from
    ``panda_link0`` gives each body's distance from the root; the parent is
    then whichever end is nearer. That stays correct if the bundle is
    re-exported with a different set of joints reversed.

    Runs against the asset composed on its own stage, never the live one; see
    :func:`repaired_panda_asset`.

    Returns the number of joints rewritten, so a caller can tell the difference
    between "already correct" and "found nothing to inspect".
    """
    joints: list[tuple[UsdPhysics.Joint, Any, Any]] = []
    for root_prim in root_prims:
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            parents = joint.GetBody0Rel().GetTargets()
            children = joint.GetBody1Rel().GetTargets()
            # Joints grounded to the world name only one body; they have no
            # direction to get wrong.
            if len(parents) != 1 or len(children) != 1:
                continue
            joints.append((joint, parents[0], children[0]))

    neighbours: dict[Any, set[Any]] = {}
    for _, parent, child in joints:
        neighbours.setdefault(parent, set()).add(child)
        neighbours.setdefault(child, set()).add(parent)

    depth: dict[Any, int] = {}
    queue: deque[Any] = deque(path for path in neighbours if path.name == _ROOT_BODY_NAME)
    for path in queue:
        depth[path] = 0
    while queue:
        current = queue.popleft()
        for neighbour in neighbours[current]:
            if neighbour not in depth:
                depth[neighbour] = depth[current] + 1
                queue.append(neighbour)

    rewritten = 0
    for joint, parent, child in joints:
        # Bodies the search never reached are not attached to the root through
        # joints, so there is no rooted direction to compare against.
        if parent not in depth or child not in depth:
            continue
        if depth[parent] > depth[child]:
            joint.GetBody0Rel().SetTargets([child])
            joint.GetBody1Rel().SetTargets([parent])
            rewritten += 1
    return rewritten


def strip_collision_geometry(root_prims: Iterable[Usd.Prim]) -> int:
    """Remove every collider from the arm, which nothing in this scene uses.

    The asset is a CAD import with ``CollisionAPI`` applied indiscriminately to
    55 of its 92 meshes -- roughly 284,000 collider faces, most of them on
    hardware that is not the arm: the wrist camera, and an end-effector
    assembly of cooling fans, tool changers, mounts and light pipes.

    Every one of those becomes a ``UsdPhysicsMeshShapeDesc``, and that
    population is what ``_FinalizeCollisionDescs`` spreads across TBB workers,
    each computing world transforms through a shared ``UsdGeomXformCache``
    that is not thread-safe. Removing the colliders is not a way of making the
    race less likely; it leaves the parallel loop with nothing to iterate.

    Nothing is given up, because the catheter drive is kinematic:
    :class:`FlangeMountedIntroducer` carries the wire's entry point in the
    flange frame and the rollers feed it, so the arm never contacts the
    catheter, the patient or itself. Its collision geometry is only ever
    parsed, never queried.

    Inertia does not depend on it either, which is the part worth checking
    before deleting colliders rather than after. Mass properties fall back to
    collision geometry only when ``MassAPI`` is absent, and here every body
    carrying an actuated degree of freedom -- ``panda_link1`` through
    ``panda_link7`` -- authors mass and a diagonal inertia explicitly. The
    three bodies that do not (``panda_hand``, ``TCP``, and the pruned D405)
    are all attached by *fixed* joints, so they weld into their parent rather
    than needing an inertia of their own. Gravity is disabled for this
    articulation as well.

    Returns the number of prims whose colliders were removed.
    """
    stripped = 0
    for root_prim in root_prims:
        for prim in Usd.PrimRange(root_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            # The mesh-specific schema carries the approximation mode and is
            # meaningless without the collider it qualifies, so it goes first.
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            stripped += 1
    return stripped


def anchor_root_to_world(root_prims: Iterable[Usd.Prim]) -> int:
    """Ground the arm's root body so the articulation is fixed-base.

    Every fixed joint this asset ships joins two of its own bodies, so nothing
    ties ``panda_link0`` to the world. Newton decides base type from the root
    joint, and finding none it builds the arm floating, handing it six base
    degrees of freedom it has no business having: the arm is bolted to a cart.

    Those six DOFs are not harmless, because they change the layout of
    everything indexed by degree of freedom. ``is_fixed_base`` turning false
    makes ``num_base_dofs`` six, and IsaacLab's differential IK -- ours is a
    direct copy of it -- shifts its Jacobian columns by that amount to skip
    base columns it expects at the front. Those columns are not at the front
    here, so the shift walks off the arm's joints entirely and the IK steers
    with a Jacobian for the wrong degrees of freedom. It diverges within a
    second, driving joint 1 past twice its limit, and the flange thrashes
    metres from the introducer.

    That lands on the catheter as well, because
    :class:`FlangeMountedIntroducer` carries the wire's entry point in the
    flange frame. A thrashing flange drags the introducer with it, so the wire
    is whipped around by an arm that was asked to hold still, whatever the
    rollers were told to do.

    Grounding the root leaves ``num_base_dofs`` at zero, which restores the
    layout the IK already assumes. ``fix_root_link`` on the spawn config
    cannot do this job: it finds one of the arm's internal fixed joints,
    enables it, and reports success without ever grounding anything.

    Returns the number of articulations grounded, so a caller can distinguish
    "already grounded" from "found nothing to ground".
    """
    anchored = 0
    for root_prim in root_prims:
        stage = root_prim.GetStage()

        root_body = next(
            (prim for prim in Usd.PrimRange(root_prim) if prim.GetName() == _ROOT_BODY_NAME),
            None,
        )
        if root_body is None:
            raise RuntimeError(
                f"Cannot ground the articulation at {root_prim.GetPath()}: it has no body named "
                f"{_ROOT_BODY_NAME!r} to anchor. The asset's root body was renamed, so the base "
                "type and every DoF-indexed quantity would be wrong."
            )

        # A joint naming one body and leaving the other end at the world is a
        # root joint, whichever end it names.
        existing = [
            prim
            for prim in Usd.PrimRange(root_prim)
            if prim.IsA(UsdPhysics.Joint)
            and len(UsdPhysics.Joint(prim).GetBody0Rel().GetTargets())
            + len(UsdPhysics.Joint(prim).GetBody1Rel().GetTargets())
            == 1
        ]
        if len(existing) > 1:
            raise RuntimeError(
                f"The articulation at {root_prim.GetPath()} has {len(existing)} root joints "
                f"({[str(prim.GetPath()) for prim in existing]}). Grounding one would leave the "
                "others contradicting it, and Newton refuses to merge duplicate roots."
            )

        # This asset ships one, but as a generic D6 with every axis free, which
        # is a floating base spelled differently. Retyping it in place keeps the
        # articulation's declared root rather than leaving that joint to fight a
        # second one added beside it. Its local frames are identity, so naming
        # the bodies the other way round moves nothing.
        path = existing[0].GetPath() if existing else root_prim.GetPath().AppendChild("root_joint")
        joint = UsdPhysics.FixedJoint.Define(stage, path)
        # An unset body0 is the world, which is what grounds the arm.
        joint.GetBody0Rel().SetTargets([])
        joint.GetBody1Rel().SetTargets([root_body.GetPath()])
        anchored += 1
    return anchored


#: Bump when the repair passes change what they author, so a stale overlay
#: from an earlier version of this file is never reused.
#:
#: 2: flatten the asset locally and strip its collision geometry.
#: 3: ground the root body so the articulation builds fixed-base.
REPAIR_LAYER_VERSION = 4


def repair_layer_path() -> Path:
    """Where the repaired overlay for the current asset and repairs lives."""
    key = hashlib.sha256(f"{FRANKA_PANDA_USD}|{REPAIR_LAYER_VERSION}".encode()).hexdigest()[:16]
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    # Binary crate rather than ``.usda``: this is a flattened copy of an 85 MB
    # asset, not a handful of overrides.
    return root / "i4h" / "assets" / f"panda_newton_repaired_{key}.usdc"


def author_repair_layer(target: Path) -> None:
    """Flatten the asset into a local layer and repair it there.

    The asset is fetched over HTTPS, and an earlier version of this function
    sublayered it rather than flattening it, which meant the 85 MB download was
    still being composed while the physics parser walked the same prims. This
    resolves the composition once, up front, so spawning reads local bytes and
    the fetch is off the critical path entirely instead of merely reordered.

    Flattening also makes the repairs ordinary edits rather than ``over``
    opinions layered on a remote asset. The remote asset is still never
    modified -- the flattened copy is a separate file under the cache -- but
    ``RemoveAPI`` needs to delete an ``apiSchemas`` entry, which an overlay
    cannot express.

    Writing to a temporary file and renaming makes a concurrent first run safe,
    and means a crash mid-bake cannot leave a half-written layer behind to be
    picked up as valid.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(f"{target.stem}.{os.getpid()}.tmp.usdc")

    source = Usd.Stage.Open(FRANKA_PANDA_USD)
    if source is None:
        raise RuntimeError(f"Could not open the Franka asset at {FRANKA_PANDA_USD}")

    layer = source.Flatten()
    stage = Usd.Stage.Open(layer)

    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(
            f"The Franka asset at {FRANKA_PANDA_USD} names a defaultPrim "
            f"({layer.defaultPrim!r}) that does not compose."
        )

    # Pruning comes first so the discarded camera joint is never a candidate
    # for reorientation, and so its colliders are gone before the strip runs.
    prune_unused_wrist_hardware([root_prim])
    orient_joints_away_from_root([root_prim])
    strip_collision_geometry([root_prim])
    # Last, so the grounding joint is never a candidate for reorientation: it
    # names one body by design, which is exactly what that pass skips.
    anchor_root_to_world([root_prim])

    layer.Export(str(scratch))
    del stage
    os.replace(scratch, target)


def repaired_panda_asset() -> str:
    """Return a local asset the Panda can be spawned from without repairs.

    Three problems share this one answer, which is why the bake does three
    things rather than one.

    The repairs originally ran on the live stage immediately after
    ``spawn_from_usd``, which is unsound: the asset is remote, so its subtree
    is still being fetched and composed when the spawner returns, and USD's
    physics parser walks those same prims across TBB workers. Deactivating a
    prim forces a resync underneath a parser holding prim handles.

    That parser is also where the process died on roughly half of all launches,
    inside ``_FinalizeCollisionDescs<UsdPhysicsMeshShapeDesc>``, with workers
    sharing a ``UsdGeomXformCache`` that is not thread-safe -- surfacing as
    ``malloc(): unaligned tcache chunk detected``, or as a hang, at whatever
    unrelated allocation came next. Moving the repairs off the live stage was
    not enough on its own, because the crash does not need our edits: it needs
    only the arm's 55 collider meshes to give those workers something to race
    over. :func:`strip_collision_geometry` takes that away.

    Flattening handles the third, which is that composing an 85 MB remote asset
    concurrently with the parse is what made the race's window wide enough to
    hit so often.
    """
    target = repair_layer_path()
    if not target.exists():
        author_repair_layer(target)
    return str(target)
