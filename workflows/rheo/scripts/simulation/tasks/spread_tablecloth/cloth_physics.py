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

"""Newton VBD physics + surface-deformable material presets for the cloth task.

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
from isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
from isaaclab_contrib.deformable.newton_manager_cfg import CoupledMJWarpVBDSolverCfg, NewtonModelCfg, VBDSolverCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.hydra import resolve_presets
from pxr import UsdGeom


@configclass
class DeformableNewtonCfg(NewtonCfg):
    """``NewtonCfg`` carrying model-level contact parameters for deformables.

    A distinct class name is required so Isaac Lab's ``_is_kitless_physics``
    check does not match it, which ensures Kit launches for USD deformable
    spawning.
    """

    model_cfg: NewtonModelCfg | None = None


def make_cloth_surface_material() -> NewtonSurfaceDeformableBodyMaterialCfg:
    """Surface-deformable (cloth) material preset.

    ``particle_radius`` was briefly bumped to 8 mm to stop fingertips slipping
    through the mesh gap, but that reshapes VBD's per-particle collision
    tables and reliably crashed model_init with a heap corruption
    (``malloc(): unaligned tcache chunk``). Revert to 5 mm; the finger-through
    behaviour is instead handled by the much stiffer ``soft_contact_ke`` in
    :func:`make_newton_physics`.
    """
    return NewtonSurfaceDeformableBodyMaterialCfg(
        density=50.0,
        particle_radius=0.005,
        tri_ke=5.0e2,
        tri_ka=5.0e2,
        tri_kd=1.0e-3,
        edge_ke=2.0,
        edge_kd=1.0e-3,
    )


def make_cloth_deformable_props() -> NewtonDeformableBodyPropertiesCfg:
    """Newton deformable body props."""
    return NewtonDeformableBodyPropertiesCfg()


def make_newton_physics() -> DeformableNewtonCfg:
    """Coupled MJWarp (robot) + VBD (cloth) Newton physics preset."""
    # Higher soft/shape contact stiffness and the
    # stiffer VBD coupling grow the per-substep contact count; give MJWarp
    # headroom so an inspire hand grabbing cloth can't overrun the static
    # buffers (which manifests as a heap corruption at model_init).
    njmax, nconmax = 400, 300
    return DeformableNewtonCfg(
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
        model_cfg=NewtonModelCfg(
            soft_contact_ke=5.0e4,  # cloth <-> hand (was 5e4: fingertips still tunnel ~1mm at ke=5e4)
            soft_contact_kd=1.0e-3,
            soft_contact_mu=1.0,
            shape_material_ke=1.0e4,
            shape_material_kd=1.0e-3,
            shape_material_mu=0.8,
        ),
        num_substeps=20,
        use_cuda_graph=True,
    )


def make_physx_physics() -> PhysxCfg:
    """Default PhysX backend (the pre-Newton solver path).

    The cloth-specific Newton tuning (coupled MJWarp + VBD, soft contacts) does
    not apply to PhysX; this just selects the stock PhysX solver so the same env
    can run on PhysX for A/B comparison against Newton.
    """
    return PhysxCfg()


@configclass
class TableclothPhysicsCfg(PresetCfg):
    """Switchable physics-backend preset for the spread-tablecloth task.

    Mirrors the upstream ``PhysicsCfg(PresetCfg)`` pattern from Isaac Lab's
    warp-only experimental envs (PR #5974): each field is a named backend, and
    ``default`` chooses the one used when no selection is given.

    Backends:
        * ``newton_mjwarp`` (default) -- coupled MJWarp (robot) + VBD (cloth).
        * ``physx``                   -- stock PhysX solver.

    Select at runtime with ``--physics_backend {newton,physx}`` (the scripts
    call :func:`select_physics_backend`, which resolves this preset on the env).
    """

    newton_mjwarp = make_newton_physics()
    physx = make_physx_physics()
    default = newton_mjwarp


# CLI/alias -> declared preset field name.
_PHYSICS_BACKEND_ALIASES = {
    "newton": "newton_mjwarp",
    "newton_mjwarp": "newton_mjwarp",
    "mjwarp": "newton_mjwarp",
    "physx": "physx",
}


_HAND_COLLISION_FLAGFIX_DONE = False


def _install_newton_hand_collision_flag_fix() -> None:
    """Work around a Newton USD-import gap that makes the G1 Inspire hand clip."""
    global _HAND_COLLISION_FLAGFIX_DONE
    if _HAND_COLLISION_FLAGFIX_DONE:
        return

    _orig = NewtonCoupledMJWarpVBDManager.start_simulation.__func__

    def _patched(cls):
        _orig(cls)
        model = cls._model
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
    """Resolve the physics backend on ``env_cfg``.

    Call this AFTER ``parse_env_cfg`` and BEFORE ``gym.make``. It only swaps
    ``env_cfg.sim.physics`` to the requested backend; gravity and robot props are
    left untouched (the spread-tablecloth task NEEDS gravity for the cloth to
    drape on the table -- zeroing it makes the cloth spring open and the inner
    body float away).

    For the H2 scene this also raises the tabletop 1.55x on both backends via
    :func:`_maybe_apply_h2_table_height_tweak`. G1 is left untouched.
    """
    key = (backend or "newton").strip().lower()
    name = _PHYSICS_BACKEND_ALIASES.get(key)
    if name is None:
        raise ValueError(f"Unknown physics backend {backend!r}. " f"Valid: {sorted(set(_PHYSICS_BACKEND_ALIASES))}.")

    env_cfg.sim.physics = resolve_presets(TableclothPhysicsCfg(), selected=[name])

    _maybe_apply_h2_table_height_tweak(env_cfg)

    if name == "newton_mjwarp":
        _install_newton_hand_collision_flag_fix()

    return name


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
