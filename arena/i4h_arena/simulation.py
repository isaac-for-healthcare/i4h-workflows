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

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def apply_simulation_defaults(env_cfg: Any) -> None:
    """Restore the pre-3.0 actuator path and PhysX external-force application."""
    env_cfg.sim.use_newton_actuators = False
    physics = env_cfg.sim.physics
    if hasattr(physics, "enable_external_forces_every_iteration"):
        physics.enable_external_forces_every_iteration = False


def rebuild_physx_scene(env: Any) -> bool:
    """Replace the PhysX scene with one parsed afresh from the stage (NVBug 6953999).

    A reset writes back every state IsaacLab exposes, yet the GPU PhysX scene
    keeps internal state across it (contact or constraint ordering is the
    suspect). An episode that follows others in one process then drifts from
    the same episode in a fresh process within a few steps, and a balancing G1
    amplifies that to tenths of a radian, so a retry's recording does not
    replay.

    Stopping the timeline invalidates the PhysX views and IsaacLab's assets and
    sensors. Detaching the stage makes the next play parse it into a new PhysX
    scene; without it, play hangs on "Stage ... already attached". Play then
    recreates the views and re-initializes every asset and sensor, as the
    env's construction did. USD keeps everything authored on the stage,
    ``prestartup`` events included; IsaacLab ``startup`` events, which ran once
    against the old scene, do not run again.

    Returns ``False`` and leaves the simulation alone when it does not run on
    Kit PhysX. Raises ``RuntimeError`` if the rebuild fails partway, which
    leaves the simulation stopped.
    """
    unwrapped = getattr(env, "unwrapped", env)
    sim = unwrapped.sim
    try:
        from isaaclab_physx.physics import PhysxManager  # noqa: PLC0415 - needs a running Kit
    except ImportError:  # no Kit PhysX to rebuild
        return False
    manager = getattr(sim, "physics_manager", None)
    if isinstance(manager, str):
        # IsaacLab keeps the configured manager as a lazily resolved "module:Class" string.
        from isaaclab.utils.string import string_to_callable  # noqa: PLC0415

        manager = string_to_callable(manager)
    if not (isinstance(manager, type) and issubclass(manager, PhysxManager)):
        return False
    from omni.physx import get_physx_simulation_interface  # noqa: PLC0415 - needs a running Kit

    started = time.monotonic()
    try:
        sim.stop()
        get_physx_simulation_interface().detach_stage()
        sim.play()
        unwrapped.scene.update(dt=unwrapped.physics_dt)
    except Exception as exc:
        # Playing again would not recover: after a failed detach the stage may
        # still be attached, and play then hangs. Fail here, where the cause is
        # named, rather than at the next reset on invalid PhysX views.
        raise RuntimeError(
            "rebuilding the PhysX scene for a fresh episode failed (NVBug 6953999); "
            "the simulation is stopped and cannot run another episode"
        ) from exc
    logger.info("rebuilt the PhysX scene for a fresh episode in %.1f s", time.monotonic() - started)
    return True


def apply_fresh_physics_per_episode(env_cfg: Any) -> None:
    """Ask Arena's environment for a new PhysX scene before every episode after physics has stepped.

    The environment calls ``env_cfg.fresh_physics_per_episode`` from its reset, through
    ``isaaclab_arena_fresh_physics_per_episode.patch``, before the reset writes the
    episode's initial state. Without that patch the attribute is ignored.
    """
    env_cfg.fresh_physics_per_episode = rebuild_physx_scene
