# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep a concave manipulandum concave under Newton.

IsaacLab's Newton cloner replaces every collision mesh with a single convex
hull. For the surgical lift objects that is the wrong shape: the hull of the
peg-transfer ring fills its hole and the hull of the suture needle's arc is a
solid half-disc, so the jaws close on geometry that is not there (NVBug
6928490). PhysX, which these grasps were tuned on, cooks the same meshes with
the convex decomposition their USD authors.

:func:`decompose_bodies` registers body-label suffixes whose collision meshes
are decomposed with CoACD instead. Every other mesh keeps the cloner's convex
hull, so scenes that register nothing are unaffected. Newton's own CoACD
default (``threshold=0.5``) is too coarse to keep the ring's wall; 0.05 is
CoACD's upstream default.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("i4h_arena.newton_meshes")

#: CoACD concavity threshold for registered bodies.
COACD_THRESHOLD = 0.05

_SUFFIXES: set[str] = set()
_keep_static = False
_installed = False


def decompose_bodies(*suffixes: str) -> None:
    """Decompose the collision meshes of bodies whose label ends with any of ``suffixes``."""
    _SUFFIXES.update(suffixes)
    _install()


def keep_static_meshes() -> None:
    """Leave static (bodiless) collision meshes as triangle meshes instead of hulls.

    A static scene authored as one mesh per surface (an operating room, an
    organ bed) becomes solid blocks as hulls, with the robot inside one of
    them. Newton collides convex shapes against static triangle meshes, as
    PhysX does with a static mesh collider.
    """
    global _keep_static
    _keep_static = True
    _install()


def _install() -> None:
    global _installed
    if _installed:
        return
    import newton  # noqa: PLC0415

    newton.ModelBuilder.approximate_meshes = _wrap(newton.ModelBuilder.approximate_meshes)
    _installed = True


def _wrap(original: Any) -> Any:
    def approximate_meshes(self: Any, method: Any = "convex_hull", shape_indices: Any = None, **kwargs: Any) -> Any:
        # Only the cloner's blanket "hull every collision mesh" call is split;
        # explicit requests pass through untouched.
        if method != "convex_hull" or shape_indices is not None or not (_SUFFIXES or _keep_static):
            return original(self, method, shape_indices, **kwargs)
        decompose, hull = _split(self)
        remeshed = set(original(self, method, hull, **kwargs) or ())
        if decompose:
            logger.info("decomposing %d collision mesh(es) of %s with CoACD", len(decompose), sorted(_SUFFIXES))
            remeshed |= set(original(self, "coacd", decompose, threshold=COACD_THRESHOLD, **kwargs) or ())
        return remeshed

    approximate_meshes.__wrapped__ = original
    return approximate_meshes


def _split(builder: Any) -> tuple[list[int], list[int]]:
    """The collision meshes the cloner would hull, split into (decompose, hull); kept static meshes are in neither."""
    from newton import GeoType, ShapeFlags  # noqa: PLC0415

    labels = getattr(builder, "body_label", None) or builder.body_key
    decompose, hull = [], []
    for shape, shape_type in enumerate(builder.shape_type):
        if shape_type != GeoType.MESH or not builder.shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES:
            continue
        body = builder.shape_body[shape]
        if body < 0 and _keep_static:
            continue
        label = str(labels[body]) if body >= 0 else ""
        (decompose if any(label.endswith(suffix) for suffix in _SUFFIXES) else hull).append(shape)
    return decompose, hull


def spawn_without_unreadable_payloads(spawn_func: Any) -> Any:
    """Wrap a USD-file spawner so payloads this USD build cannot read are never loaded.

    ``Props/Organs/organs.usd`` payloads ``Operating_table.obj``, a format with
    no file-format plugin here. Under PhysX the prim simply stays empty, but
    Newton refuses to import a stage with any composition error, so the organ
    scene never builds. A stage load rule leaves each such payload unloaded at
    its destination, so it is never opened and nothing new is authored.
    """

    def spawn(prim_path: str, cfg: Any, *args: Any, **kwargs: Any) -> Any:
        import isaaclab.sim as sim_utils  # noqa: PLC0415
        from pxr import Usd  # noqa: PLC0415

        relatives = unreadable_payload_prims(cfg.usd_path)
        if relatives:
            stage = sim_utils.get_current_stage()
            rules = stage.GetLoadRules()
            for root in _destinations(stage, prim_path):
                for relative in relatives:
                    rules.AddRule(f"{root}/{relative}", Usd.StageLoadRules.NoneRule)
                    logger.info("not loading %s/%s: its payload format has no USD file-format plugin", root, relative)
            stage.SetLoadRules(rules)
        return spawn_func(prim_path, cfg, *args, **kwargs)

    return spawn


def _destinations(stage: Any, prim_path: str) -> list[str]:
    """``prim_path`` under every existing parent its regex parent matches."""
    import re  # noqa: PLC0415

    parent, _, name = prim_path.rpartition("/")
    if ".*" not in parent:
        return [prim_path]
    pattern = re.compile(parent.replace(".*", "[^/]*") + "$")
    return [f"{prim.GetPath()}/{name}" for prim in stage.Traverse() if pattern.match(str(prim.GetPath()))]


def unreadable_payload_prims(usd_path: str) -> list[str]:
    """Prims of ``usd_path``, relative to its default prim, whose payload format has no plugin."""
    from pxr import Sdf, Usd  # noqa: PLC0415

    stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadNone)
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    found: list[str] = []
    # All prims: opened with LoadNone, a prim that carries a payload is unloaded,
    # and the default predicate would skip exactly the prims being looked for.
    for prim in Usd.PrimRange(root, Usd.PrimAllPrimsPredicate):
        for spec in prim.GetPrimStack():
            payloads = spec.payloadList.GetAddedOrExplicitItems()
            if any(
                p.assetPath and Sdf.FileFormat.FindByExtension(p.assetPath.rpartition(".")[2]) is None for p in payloads
            ):
                found.append(str(prim.GetPath().MakeRelativePath(root.GetPath())))
                break
    return found
