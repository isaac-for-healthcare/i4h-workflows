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
    """Leave static scenery's collision meshes as triangle meshes instead of hulls.

    Static scenery is a bodiless mesh, or one on a kinematic body, which
    contacts never move. A scene authored as one mesh per surface (an
    operating room, an organ bed) becomes solid blocks as hulls, with the
    robot inside one of them. The organ bed's only collider is a kinematic
    body's mesh: its hull, 64 x 53 x 26 cm, enclosed the needle and the tool
    tip, and the first step shot the needle out of it (NVBug 6928490). Newton
    collides convex shapes against triangle meshes, as PhysX does with the
    organ bed's SDF mesh.
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
            first_new = len(self.shape_gap)
            remeshed |= set(original(self, "coacd", decompose, threshold=COACD_THRESHOLD, **kwargs) or ())
            _inherit_gap(self, decompose, first_new)
        return remeshed

    approximate_meshes.__wrapped__ = original
    return approximate_meshes


def _inherit_gap(builder: Any, sources: list[int], first_new: int) -> None:
    """Give the convex parts CoACD adds their source shape's contact gap.

    Newton copies the source's other contact settings onto the parts but
    leaves ``gap`` at its builder default, 0.1 m, ten times IsaacLab's
    0.01 m. On the organ bed, the needle's parts then gathered about 100
    speculative contacts a step with the bed's triangle mesh. Newton reported
    one of them at zero distance, several centimetres from the needle, so the
    solver kept it active, and the simulation went NaN every few episodes
    (NVBug 6928490).
    """
    by_label = {builder.shape_label[shape]: shape for shape in sources}
    for shape in range(first_new, len(builder.shape_gap)):
        source = by_label.get(builder.shape_label[shape].rpartition("_convex_")[0])
        if source is not None:
            builder.shape_gap[shape] = builder.shape_gap[source]


def _split(builder: Any) -> tuple[list[int], list[int]]:
    """The collision meshes the cloner would hull, split into (decompose, hull); kept static meshes are in neither."""
    from newton import BodyFlags, GeoType, ShapeFlags  # noqa: PLC0415

    labels = getattr(builder, "body_label", None) or builder.body_key
    decompose, hull = [], []
    for shape, shape_type in enumerate(builder.shape_type):
        if shape_type != GeoType.MESH or not builder.shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES:
            continue
        body = builder.shape_body[shape]
        if _keep_static and (body < 0 or builder.body_flags[body] & BodyFlags.KINEMATIC):
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
