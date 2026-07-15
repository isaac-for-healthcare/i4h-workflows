# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Solvers module for position-based dynamics simulations.

This module contains implementations of various constraint solvers,
including the Direct Position-Based Solver for Stiff Rods based on
Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods".

Features:
- XPBD (Extended Position-Based Dynamics) framework
- Cosserat rod model for bending and twisting
- Separate stiffness controls (stretch, shear, bend, twist)
- Tip shaping for catheter/guidewire simulation
- BVH-accelerated mesh collision
- Friction models (Coulomb, viscous, static/dynamic)
"""

# Solver implementations (single comment block keeps isort/ruff-format stable —
# interleaving comments between these imports causes the two hooks to oscillate):
#   - newton_xpbd_rod_wrapper: Newton GPU XPBD rod solver (requires Newton build with SolverXPBDRod, e.g. PR #1981)
#   - rod_solver: kept for Isaac Lab/Newton integration (not used by standalone viewport path)
#   - xcath_rod_solver: vessel mesh collision extension — catheter-in-vessel containment + track guidance
#   - xpbd_rod_solver: self-contained XPBD rod solver — no external newton dependency
from .newton_xpbd_rod_wrapper import NewtonXPBDRodSolver, orientations_xyzw_along_polyline
from .rod_data import (
    CollisionMeshConfig,
    FrictionConfig,
    RodConfig,
    RodData,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolverConfig,
    RodTipConfig,
)
from .rod_solver import RodSolver
from .xcath_rod_solver import (
    COLLISION_PROJECTION_STAGE_POST,
    COLLISION_PROJECTION_STAGE_PRE,
    XCathRodSolver,
    compute_signed_distances,
    compute_smooth_vertex_normals,
)
from .xpbd_rod_solver import XPBDRodSolver

__all__ = [
    "RodSolver",
    "NewtonXPBDRodSolver",
    "XPBDRodSolver",
    "XCathRodSolver",
    "COLLISION_PROJECTION_STAGE_POST",
    "COLLISION_PROJECTION_STAGE_PRE",
    "compute_smooth_vertex_normals",
    "compute_signed_distances",
    "orientations_xyzw_along_polyline",
    "RodConfig",
    "RodData",
    "RodMaterialConfig",
    "RodGeometryConfig",
    "RodSolverConfig",
    "RodTipConfig",
    "FrictionConfig",
    "CollisionMeshConfig",
]
