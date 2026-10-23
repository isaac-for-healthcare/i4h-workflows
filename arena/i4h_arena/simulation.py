# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulation defaults i4h scenes keep from before Isaac Lab 3.0.

Isaac Lab 3.0 GA turned on two simulation defaults that change robot dynamics:
``SimulationCfg.use_newton_actuators`` runs explicit actuators, such as the G1's
``IdealPDActuatorCfg`` arms and legs, through a native actuator adapter, and
``PhysxCfg.enable_external_forces_every_iteration`` applies external forces in
every TGS position iteration. i4h policies were trained with both off, and with them
on the G1 cart-push policy fails most episodes, so every i4h scene restores them
before its own tuning runs.
"""

from __future__ import annotations

from typing import Any


def apply_simulation_defaults(env_cfg: Any) -> None:
    """Restore the pre-3.0 actuator path and PhysX external-force application."""
    env_cfg.sim.use_newton_actuators = False
    physics = env_cfg.sim.physics
    if hasattr(physics, "enable_external_forces_every_iteration"):
        physics.enable_external_forces_every_iteration = False
