# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coupled Newton cloth physics and ordered Pink IK actions.

Coupled MJWarp (robot) + VBD (cloth) Newton physics for the spread-tablecloth
task.

Also hosts :class:`PinkInverseKinematicsActionOrderedCfg`, a project-local
subclass of ``PinkInverseKinematicsActionCfg`` that forces
``find_joints(..., preserve_order=True)``. Upstream ``PinkInverseKinematicsAction``
doesn't expose this flag on its cfg (unlike ``JointActionCfg`` which does),
and its default (``preserve_order=False``) silently reorders ``hand_joint_names``
to the articulation's own joint ordering. That's a no-op on PhysX because
``H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER`` and
``G1_INSPIRE_HAND_JOINT_NAMES_ARTICULATION_ORDER`` are written in PhysX BFS
order, but Newton uses a per-finger DFS order, so the action-tensor→joint
mapping gets scrambled on Newton (e.g. the value intended for
``left_thumb_CMC_FE`` ends up on ``left_middle_MCP_FE``). Kept co-located with
the Newton physics preset so the whole Newton fix lives in one file; delete
the subclass once IsaacLab exposes ``preserve_order`` on
``PinkInverseKinematicsActionCfg`` upstream.
"""

from __future__ import annotations

from collections import defaultdict

import newton
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage
from isaaclab.utils import configclass
from isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg, NewtonSoftContactCfg, VBDSolverCfg
from pxr import Usd, UsdGeom


def make_newton_physics() -> NewtonCfg:
    """Coupled MJWarp (robot) + VBD (cloth) Newton physics preset."""
    # Higher soft/shape contact stiffness and the
    # stiffer VBD coupling grow the per-substep contact count; give MJWarp
    # headroom so an inspire hand grabbing cloth can't overrun the static
    # buffers (which manifests as a heap corruption at model_init).
    njmax, nconmax = 400, 300
    return NewtonCfg(
        solver_cfg=CoupledMJWarpVBDSolverCfg(
            rigid_solver_cfg=MJWarpSolverCfg(
                njmax=njmax,
                nconmax=nconmax,
                ls_iterations=10,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
                ccd_iterations=20,
            ),
            soft_solver_cfg=VBDSolverCfg(
                iterations=5,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=True,
                particle_rest_shape_contact_exclusion_radius=0.01,
                particle_collision_detection_interval=-1,
            ),
            coupling_mode="two_way",
        ),
        # Isaac Lab 3.0 GA splits the former NewtonModelCfg: body-particle contact
        # stays global, and the rigid shape material becomes the scene's default shape.
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=5.0e4,  # Cloth-hand contact stiffness.
            soft_contact_kd=1.0e-3,
            soft_contact_mu=1.0,
        ),
        default_shape_cfg=NewtonShapeCfg(ke=1.0e4, kd=1.0e-3, mu=0.8),
        num_substeps=20,
        use_cuda_graph=True,
    )


_HAND_COLLISION_FLAGFIX_DONE = False


def _install_newton_hand_collision_flag_fix() -> None:
    """Work around a Newton USD-import gap that makes the G1 Inspire hand clip."""
    global _HAND_COLLISION_FLAGFIX_DONE
    if _HAND_COLLISION_FLAGFIX_DONE:
        return

    _orig = NewtonCoupledMJWarpVBDManager.start_simulation.__func__

    def _patched(cls):
        _orig(cls)
        model = cls.get_model()
        if model is None:
            return
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags = model.shape_flags
        fnp = flags.numpy()
        sb = model.shape_body.numpy()
        blabels = [str(b).lower() for b in model.body_label]
        hand_kw = ("index", "middle", "ring", "pinky", "thumb", "hand", "palm")

        by_body = defaultdict(list)
        for si in range(len(fnp)):
            by_body[int(sb[si])].append(si)

        changed = 0
        for bi, sids in by_body.items():
            if not (0 <= bi < len(blabels)) or not any(k in blabels[bi] for k in hand_kw):
                continue
            if any(int(fnp[si]) & collide for si in sids):
                continue  # body already has a real collider (e.g. H2) -> leave it
            for si in sids:
                fnp[si] = int(fnp[si]) | collide
                changed += 1
        if changed:
            flags.assign(fnp)
            print(f"[g1-hand-flagfix] enabled COLLIDE on {changed} visual-only hand shapes")

    NewtonCoupledMJWarpVBDManager.start_simulation = classmethod(_patched)
    _HAND_COLLISION_FLAGFIX_DONE = True


def select_physics_backend(env_cfg: ManagerBasedRLEnvCfg, backend: str = "newton") -> str:
    """Apply the coupled cloth solver and the recovered H2 table-height adjustment.

    PhysX cannot create a valid view of this surface cloth with the pinned
    runtime. Reject it rather than displaying a cloth that does not simulate.
    """
    if backend != "newton":
        raise ValueError("tablecloth requires the newton physics backend")
    env_cfg.sim.physics = make_newton_physics()
    env_cfg.scene.cloth.spawn.func = spawn_cloth_with_root_deformable_body
    _maybe_apply_h2_table_height_tweak(env_cfg)
    _install_newton_hand_collision_flag_fix()
    return "newton"


_DEFORMABLE_BODY_API = "OmniPhysicsDeformableBodyAPI"


def spawn_cloth_with_root_deformable_body(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the tablecloth with its deformable body declared on the asset root.

    The cloth USD applies the deformable body API to its mesh, which binds a PhysX
    material, and binds its Newton cloth material on the root. Isaac Lab 3 imports a
    Newton deformable from the prim carrying the body API, with the material bound
    there, so the API moves to the root: the mesh and material Isaac Lab up to
    v3.0.0-beta2 simulated, under the asset's own prim path.
    """
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    for child in Usd.PrimRange(prim):
        if child != prim and _DEFORMABLE_BODY_API in child.GetAppliedSchemas():
            child.RemoveAppliedSchema(_DEFORMABLE_BODY_API)
    prim.AddAppliedSchema(_DEFORMABLE_BODY_API)
    return prim


_H2_TABLE_Z_SCALE = 1.5


def spawn_table_scaled_top(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    z = float(getattr(cfg, "child_z_scale", 1.0))
    if z == 1.0:
        return prim

    stage = get_current_stage()
    for env_prim_path in find_matching_prim_paths(prim_path):
        for rel in ("Table256/Collisions", "Table256/Visuals"):
            child = stage.GetPrimAtPath(f"{env_prim_path}/{rel}")
            for op in UsdGeom.Xformable(child).GetOrderedXformOps():
                if op.GetOpName() == "xformOp:scale":
                    op.Set(type(op.Get())(1.0, 1.0, z))
                    break
    return prim


def _maybe_apply_h2_table_height_tweak(env_cfg: ManagerBasedRLEnvCfg) -> None:
    from .h2_spread_tablecloth_env_cfg import H2SpreadTableclothEnvCfg  # circular

    if not isinstance(env_cfg, H2SpreadTableclothEnvCfg):
        return

    z = _H2_TABLE_Z_SCALE
    scene = env_cfg.scene
    scene.table.spawn.func = spawn_table_scaled_top
    scene.table.spawn.child_z_scale = z

    # Table origin at pos.z=0.385 (bottom on ground); tabletop top = 0.77 * z.
    # Cloth sits on the top; cloth_inner keeps its 6 cm hover.
    tx, ty, _ = scene.table.init_state.pos
    scene.table.init_state.pos = (tx, ty, 0.385 * z)
    cloth_top = 0.77 * z
    cx, cy, _ = scene.cloth.init_state.pos
    scene.cloth.init_state.pos = (cx, cy, cloth_top)
    ix, iy, _ = scene.cloth_inner.init_state.pos
    scene.cloth_inner.init_state.pos = (ix, iy, cloth_top + 0.06)


# ---------------------------------------------------------------------------
# Pink IK action: order-preserving variant (see module docstring for rationale).
# ---------------------------------------------------------------------------
class PinkInverseKinematicsActionOrdered(PinkInverseKinematicsAction):
    """PinkInverseKinematicsAction that honors ``cfg.preserve_order`` on ``find_joints``."""

    def _initialize_joint_info(self) -> None:
        preserve_order = bool(getattr(self.cfg, "preserve_order", True))

        self._isaaclab_controlled_joint_ids, self._isaaclab_controlled_joint_names = self._asset.find_joints(
            self.cfg.pink_controlled_joint_names, preserve_order=preserve_order
        )
        self.cfg.controller.joint_names = self._isaaclab_controlled_joint_names
        self._isaaclab_all_joint_ids = list(range(len(self._asset.data.joint_names)))
        self.cfg.controller.all_joint_names = self._asset.data.joint_names

        self._hand_joint_ids, self._hand_joint_names = self._asset.find_joints(
            self.cfg.hand_joint_names, preserve_order=preserve_order
        )

        self._controlled_joint_ids = self._isaaclab_controlled_joint_ids + self._hand_joint_ids
        self._controlled_joint_names = self._isaaclab_controlled_joint_names + self._hand_joint_names


@configclass
class PinkInverseKinematicsActionOrderedCfg(PinkInverseKinematicsActionCfg):
    """Cfg variant that adds a ``preserve_order`` field (mirrors ``JointActionCfg``).

    ``class_type`` is rebound in ``__post_init__`` rather than as a class-level
    default because the parent declares it as a ``ResolvableString`` (``"{DIR}:..."``);
    a plain-class default here can get shadowed during ``configclass`` field
    inheritance. Setting it after ``@configclass`` has processed the class
    guarantees the manager instantiates our subclass.

    Set ``preserve_order=False`` to fall back to upstream behavior; leave True
    (default) whenever ``pink_controlled_joint_names`` / ``hand_joint_names``
    are packed in a specific order that the action tensor depends on,
    otherwise Newton silently reorders the mapping (see module docstring).
    """

    preserve_order: bool = True
    """Whether to keep the input name-list order when resolving joint ids."""

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        self.class_type = PinkInverseKinematicsActionOrdered
