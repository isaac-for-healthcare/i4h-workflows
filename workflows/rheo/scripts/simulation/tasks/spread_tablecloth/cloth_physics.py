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

Coupled MJWarp (robot) + VBD (cloth) using the official Newton solver/contact
style, no cloth self-contact, and high robot-shape friction for cloth grasping.
World gravity and robot ``disable_gravity`` are left at the env defaults.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim.simulation_cfg import PhysicsCfg
from isaaclab.utils import configclass
from isaaclab_contrib.deformable.newton_manager_cfg import CoupledMJWarpVBDSolverCfg, NewtonModelCfg, VBDSolverCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.hydra import resolve_presets

# High robot-shape friction so the gripper can grasp the cloth (startup material event).
ROBOT_SHAPE_MATERIAL_MU = 100.0


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

    NOTE: this is NOT applied at runtime. The cloth spawns from
    ``assets/cloth_only.usda``, which authors+binds its OWN ``newton:*`` material
    attributes on the template root; isaaclab_contrib reads stiffness from those
    USD attrs (the spawn passes neither ``physics_material`` nor
    ``deformable_props``). To actually change cloth stiffness (tri_ke / edge_ke /
    density / ...), edit ``cloth_only.usda``. This function is kept only as a
    documented mirror of those values.
    """
    return NewtonSurfaceDeformableBodyMaterialCfg(
        density=50.0,
        particle_radius=0.005,
        tri_ke=5.0e2,
        tri_ka=5.0e2,
        tri_kd=1.0e-3,
        edge_ke=2.0,  # mirror of cloth_only.usda newton:edgeKe (higher -> VBD blows up at this dt/iterations)
        edge_kd=1.0e-3,
    )


def make_cloth_deformable_props() -> NewtonDeformableBodyPropertiesCfg:
    """Newton deformable body props."""
    return NewtonDeformableBodyPropertiesCfg()


def make_newton_physics() -> DeformableNewtonCfg:
    """Coupled MJWarp (robot) + VBD (cloth) Newton physics preset."""
    njmax, nconmax = 200, 100
    return DeformableNewtonCfg(
        solver_cfg=CoupledMJWarpVBDSolverCfg(
            rigid_solver_cfg=MJWarpSolverCfg(
                njmax=njmax,
                nconmax=nconmax,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
                ccd_iterations=100,
            ),
            soft_solver_cfg=VBDSolverCfg(
                iterations=10,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=True,
                particle_rest_shape_contact_exclusion_radius=0.01,
                particle_collision_detection_interval=-1,
            ),
            coupling_mode="two_way",
        ),
        model_cfg=NewtonModelCfg(
            soft_contact_ke=1.0e3,
            soft_contact_kd=1.0e-5,
            soft_contact_mu=0.5,
            shape_material_ke=1.0e3,
            shape_material_kd=1.0e-5,
            shape_material_mu=1.0e-4,
        ),
        num_substeps=10,
        # False: CUDA graph + env.reset() after a failed hold-step stalls for minutes.
        use_cuda_graph=False,
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


def select_physics_backend(env_cfg: ManagerBasedRLEnvCfg, backend: str = "newton") -> str:
    """Resolve the physics backend on ``env_cfg``.

    Call this AFTER ``parse_env_cfg`` and BEFORE ``gym.make``. It only swaps
    ``env_cfg.sim.physics`` to the requested backend; gravity and robot props are
    left untouched (the spread-tablecloth task NEEDS gravity for the cloth to
    drape on the table -- zeroing it makes the cloth spring open and the inner
    body float away).

    Args:
        env_cfg: The parsed env cfg to mutate in place.
        backend: ``"newton"`` (default) / ``"newton_mjwarp"`` / ``"mjwarp"`` or
            ``"physx"``.

    Returns:
        The resolved preset field name (``"newton_mjwarp"`` or ``"physx"``).
    """
    key = (backend or "newton").strip().lower()
    name = _PHYSICS_BACKEND_ALIASES.get(key)
    if name is None:
        raise ValueError(
            f"Unknown physics backend {backend!r}. "
            f"Valid: {sorted(set(_PHYSICS_BACKEND_ALIASES))}."
        )

    env_cfg.sim.physics = resolve_presets(TableclothPhysicsCfg(), selected=[name])

    # H2_sharpa-only PhysX root-pose compensation.
    if name == "physx":
        _apply_h2_physx_root_offset(env_cfg)

    return name


# Authored ``H2`` Xform translate baked into the H2_sharpa USD.
_H2_USD_XFORM_Z_OFFSET = 1.05


def _apply_h2_physx_root_offset(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Add the H2_sharpa Xform z-offset to ``robot.init_state.pos`` for PhysX.

    No-op for any robot whose spawn USD is not H2_sharpa.
    """
    try:
        robot_cfg = env_cfg.scene.robot
        usd_path = (robot_cfg.spawn.usd_path or "").lower()
    except AttributeError:
        return
    if "h2_sharpa" not in usd_path and "h2_with_sharpa" not in usd_path:
        return
    try:
        x, y, z = robot_cfg.init_state.pos
    except (AttributeError, TypeError, ValueError):
        return
    robot_cfg.init_state.pos = (x, y, z + _H2_USD_XFORM_Z_OFFSET)
