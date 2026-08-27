# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene entity that surfaces catheter state, without owning its physics.

Newton's manager owns the rod solver. This entity holds what a sensor
legitimately holds: the 3D markers, the proximal insertion and rotation
bookkeeping that gets recorded as virtual joints, and the read path that turns
Newton's particle buffers into a catheter polyline.

Control is forwarded to the solver as per-environment device arrays rather than
scalars, so a batched scene does not serialize on a host copy per step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.configclass import configclass

from i4h_common.types import JointState

from .catheter import CatheterState
from .newton_catheter_physics import require_active_handle
from .newton_providers import NewtonRodCatheterStateProvider


@dataclass(slots=True)
class CatheterAssetData:
    """Runtime state surfaced to Arena adapters and diagnostics."""

    positions_world_m: torch.Tensor | None = None
    insertion_m: torch.Tensor | None = None
    rotation_rad: torch.Tensor | None = None
    command: torch.Tensor | None = None


class XpbdCatheterAsset(SensorBase):
    """Catheter markers, virtual joints, and the Newton read path.

    The rod solver is built and stepped by
    :class:`~catheter_vasculature_solver.isaaclab_integration.xpbd_rod_manager.NewtonXPBDRodManager`,
    so nothing here creates a solver or advances time.
    """

    cfg: XpbdCatheterAssetCfg

    def __init__(self, cfg: XpbdCatheterAssetCfg):
        self._data = CatheterAssetData()
        self._marker: Any = None
        self._provider: NewtonRodCatheterStateProvider | None = None
        super().__init__(cfg)

    @property
    def data(self) -> CatheterAssetData:
        self._refresh_positions()
        return self._data

    def advance(self, commands: torch.Tensor, dt: float) -> None:
        """Forward proximal insertion and rotation rates to the rod solver.

        Newton steps the solver, so this only sets control. Velocities are
        handed over as device arrays; reading them out with ``.item()`` would
        synchronize on every environment every substep.
        """
        command = commands.detach().to(device=self._device, dtype=torch.float32)
        if command.shape != (self._num_envs, 2):
            raise ValueError(f"catheter command must have shape ({self._num_envs}, 2), got {tuple(command.shape)}")
        rod = require_active_handle().rod
        rod.apply_proximal_control_gpu(command[:, 0], command[:, 1], float(dt))

        assert self._data.insertion_m is not None
        assert self._data.rotation_rad is not None
        assert self._data.command is not None
        self._data.insertion_m.add_(command[:, 0] * float(dt))
        self._data.rotation_rad.add_(command[:, 1] * float(dt))
        self._data.command.copy_(command)

    def snapshot(self, num_envs: int) -> CatheterState:
        """Return the catheter polyline in Isaac world coordinates."""
        return self._catheter_provider().snapshot(num_envs)

    def joint_state(self) -> JointState:
        """Represent proximal insertion/rotation as recordable virtual joints."""
        if self._data.insertion_m is None or self._data.rotation_rad is None or self._data.command is None:
            zeros = np.zeros((self._num_envs, 2), dtype=np.float32)
            return JointState(pos=zeros, vel=zeros.copy(), names=("insertion_m", "rotation_rad"))
        pos = torch.stack((self._data.insertion_m, self._data.rotation_rad), dim=-1).detach().cpu().numpy()
        vel = self._data.command.detach().cpu().numpy()
        return JointState(
            pos=pos.astype(np.float32, copy=False),
            vel=vel.astype(np.float32, copy=False),
            names=("insertion_m", "rotation_rad"),
        )

    def reset(self, env_ids=None, env_mask: wp.array | None = None) -> None:
        """Restore the listed environments without rebuilding the solver."""
        super().reset(env_ids=env_ids, env_mask=env_mask)
        if not self.is_initialized:
            return
        self._zero_bookkeeping(env_ids)
        require_active_handle().reset(env_ids)
        self._refresh_positions()

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        self._data.insertion_m = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._data.rotation_rad = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._data.command = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._refresh_positions()

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        del env_mask
        self._refresh_positions()

    def _zero_bookkeeping(self, env_ids) -> None:
        if self._data.insertion_m is None or self._data.rotation_rad is None or self._data.command is None:
            return
        if env_ids is None:
            self._data.insertion_m.zero_()
            self._data.rotation_rad.zero_()
            self._data.command.zero_()
            return
        self._data.insertion_m[env_ids] = 0.0
        self._data.rotation_rad[env_ids] = 0.0
        self._data.command[env_ids] = 0.0

    def _catheter_provider(self) -> NewtonRodCatheterStateProvider:
        """Build the Newton read path once the rod's particle range is known."""
        if self._provider is None:
            from isaaclab_newton.physics import NewtonManager

            handle = require_active_handle()
            self._provider = NewtonRodCatheterStateProvider.from_particle_range(
                NewtonManager.get_state,
                handle.particle_range,
                radius_m=float(self.cfg.radius_m),
            )
        return self._provider

    def _refresh_positions(self) -> None:
        try:
            state = self._catheter_provider().snapshot(self._num_envs)
        except RuntimeError:
            # Before the model is finalized there are no particles to read.
            return
        self._data.positions_world_m = torch.as_tensor(
            state.positions_world_m, device=self._device, dtype=torch.float32
        )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if self._marker is None and debug_vis:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self._marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/FluoroscopyCatheter",
                    markers={
                        "shaft": sim_utils.SphereCfg(
                            # Deliberately larger than the physical radius so a
                            # sub-millimetre wire remains legible in the 3D overview.
                            radius=max(0.003, 2.0 * float(self.cfg.radius_m)),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.85, 0.95)),
                        ),
                        "tip": sim_utils.SphereCfg(
                            radius=max(0.007, 4.0 * float(self.cfg.radius_m)),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.20, 1.0, 0.12),
                                emissive_color=(0.04, 0.55, 0.02),
                            ),
                        ),
                    },
                )
            )
        if self._marker is not None:
            self._marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event: Any) -> None:
        del event
        if self._marker is not None and self._data.positions_world_m is not None:
            translations = self._data.positions_world_m.reshape(-1, 3)
            marker_indices = torch.zeros(translations.shape[0], dtype=torch.int32, device=translations.device)
            # Mark each environment's distal end so the tip stays identifiable
            # when several catheters are on screen at once.
            num_points = self._data.positions_world_m.shape[1]
            if num_points:
                marker_indices[num_points - 1 :: num_points] = 1
            self._marker.visualize(translations=translations, marker_indices=marker_indices)


@configclass
class XpbdCatheterAssetCfg(SensorBaseCfg):
    """Configuration for the catheter scene entity.

    Rod geometry and material live on
    :class:`~i4h_arena.medical.newton_catheter_physics.CatheterRodSpec`, since
    the physics manager owns the solver. What remains here is what this entity
    itself needs: the radius it draws and reports.
    """

    class_type: type[XpbdCatheterAsset] = XpbdCatheterAsset
    radius_m: float = 0.0005
