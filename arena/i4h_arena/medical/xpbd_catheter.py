# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac-managed adapter for the i4h Warp XPBD catheter solver."""

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
from .centerline import sample_polyline


@dataclass(slots=True)
class CatheterAssetData:
    """Runtime state surfaced to Arena adapters and diagnostics."""

    positions_world_m: torch.Tensor | None = None
    insertion_m: torch.Tensor | None = None
    rotation_rad: torch.Tensor | None = None
    command: torch.Tensor | None = None


class XpbdCatheterAsset(SensorBase):
    """Scene entity that owns catheter mechanics and exposes render state.

    The upstream solver remains the authority for the polyline. Isaac's action
    lifecycle only supplies proximal insertion and axial-rotation velocities.
    Vessel collision is intentionally disabled until a patient collision mesh
    is bound in the patient-specific slice.
    """

    cfg: XpbdCatheterAssetCfg

    def __init__(self, cfg: XpbdCatheterAssetCfg):
        self._data = CatheterAssetData()
        self._solver: Any = None
        self._marker: Any = None
        self._guide_path_world_m: np.ndarray | None = None
        self._guide_position_m = 0.0
        super().__init__(cfg)

    @property
    def data(self) -> CatheterAssetData:
        self._refresh_positions()
        return self._data

    def advance(self, commands: torch.Tensor, dt: float) -> None:
        """Advance XPBD once for an Isaac physics substep."""
        if self._solver is None:
            return
        command = commands.detach().to(device=self._device, dtype=torch.float32)
        if command.shape != (1, 2):
            raise ValueError(f"catheter command must have shape (1, 2), got {tuple(command.shape)}")
        push_velocity = float(command[0, 0].item())
        rotation_velocity = float(command[0, 1].item())
        guided = self._guide_path_world_m is not None
        self._solver.apply_proximal_control(0.0 if guided else push_velocity, rotation_velocity, float(dt))
        self._solver.step(float(dt))
        if guided:
            self._advance_guide(push_velocity, float(dt))
        assert self._data.insertion_m is not None
        assert self._data.rotation_rad is not None
        assert self._data.command is not None
        self._data.insertion_m.add_(push_velocity * float(dt))
        self._data.rotation_rad.add_(rotation_velocity * float(dt))
        self._data.command.copy_(command)
        self._refresh_positions()

    def snapshot(self, num_envs: int) -> CatheterState:
        """Return the latest solver polyline in Isaac world coordinates."""
        if num_envs != 1:
            raise ValueError("the vessel-aware catheter rod solver currently supports exactly one Arena environment")
        self._refresh_positions()
        if self._data.positions_world_m is None:
            return CatheterState.empty(1)
        positions = self._data.positions_world_m.detach().cpu().numpy().astype(np.float32, copy=True)
        return CatheterState(
            positions_world_m=positions,
            valid_nodes=np.array([positions.shape[1]], dtype=np.int32),
            radius_m=self.cfg.radius_m,
        )

    def joint_state(self) -> JointState:
        """Represent proximal insertion/rotation as recordable virtual joints."""
        if self._data.insertion_m is None or self._data.rotation_rad is None or self._data.command is None:
            zeros = np.zeros((1, 2), dtype=np.float32)
            return JointState(pos=zeros, vel=zeros.copy(), names=("insertion_m", "rotation_rad"))
        pos = torch.stack((self._data.insertion_m, self._data.rotation_rad), dim=-1).detach().cpu().numpy()
        vel = self._data.command.detach().cpu().numpy()
        return JointState(
            pos=pos.astype(np.float32, copy=False),
            vel=vel.astype(np.float32, copy=False),
            names=("insertion_m", "rotation_rad"),
        )

    def reset(self, env_ids=None, env_mask: wp.array | None = None) -> None:
        super().reset(env_ids=env_ids, env_mask=env_mask)
        if self.is_initialized:
            self._create_solver()

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        self._create_solver()

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        del env_mask
        self._refresh_positions()

    def _create_solver(self) -> None:
        from catheter_vasculature_solver import CathRodSolver, RodConfig

        config = RodConfig()
        config.device = str(self.cfg.solver_device)
        config.geometry.num_segments = int(self.cfg.num_segments)
        config.geometry.rest_length = float(self.cfg.length_m)
        config.geometry.segment_length = float(self.cfg.length_m) / float(self.cfg.num_segments)
        config.geometry.radius = float(self.cfg.radius_m)
        config.solver.num_substeps = int(self.cfg.solver_substeps)
        config.solver.gravity = (0.0, 0.0, 0.0)
        self._solver = CathRodSolver(
            config,
            num_envs=1,
            floor_z=None,
            initial_height=0.0,
            collision_mesh=None,
            track_start=np.zeros(3, dtype=np.float32),
            track_dir=np.asarray(self.cfg.track_direction_world_m, dtype=np.float32),
            track_length=float(self.cfg.length_m + self.cfg.max_insertion_m),
            tip_num_edges=int(self.cfg.tip_num_edges),
            particle_radius=float(self.cfg.radius_m),
            segment_length=float(config.geometry.segment_length),
            track_enabled=self.cfg.guide_path_world_m is None,
            collision_enabled=False,
            track_stiffness=float(self.cfg.track_stiffness),
        )
        if self.cfg.tip_bend_rad != 0.0:
            self._solver.set_tip_bend(float(self.cfg.tip_bend_rad))
        self._data.insertion_m = torch.zeros(1, device=self._device, dtype=torch.float32)
        self._data.rotation_rad = torch.zeros(1, device=self._device, dtype=torch.float32)
        self._data.command = torch.zeros((1, 2), device=self._device, dtype=torch.float32)
        self._guide_position_m = 0.0
        self._guide_path_world_m = (
            np.asarray(self.cfg.guide_path_world_m, dtype=np.float32)
            if self.cfg.guide_path_world_m is not None
            else None
        )
        if self._guide_path_world_m is not None:
            self._advance_guide(0.0, 0.0)
        self._refresh_positions()

    def _advance_guide(self, insertion_velocity_mps: float, dt: float) -> None:
        assert self._guide_path_world_m is not None
        segments = np.linalg.norm(np.diff(self._guide_path_world_m, axis=0), axis=1)
        total_length = float(np.sum(segments))
        maximum = max(0.0, total_length - float(self.cfg.length_m))
        self._guide_position_m = float(np.clip(self._guide_position_m + insertion_velocity_mps * dt, 0.0, maximum))
        distances = self._guide_position_m + np.linspace(0.0, float(self.cfg.length_m), int(self.cfg.num_segments) + 1)
        world_positions = sample_polyline(self._guide_path_world_m, distances)
        local_positions = world_positions - np.asarray(self.cfg.origin_world_m, dtype=np.float32)
        workspace = self._solver._ws
        position_buffer = wp.to_torch(workspace.positions)
        positions = torch.as_tensor(local_positions, device=position_buffer.device, dtype=position_buffer.dtype)
        position_buffer.copy_(positions)
        wp.to_torch(workspace.predicted_positions).copy_(positions)
        if hasattr(workspace, "prev_positions"):
            wp.to_torch(workspace.prev_positions).copy_(positions)
        if hasattr(workspace, "velocities"):
            wp.to_torch(workspace.velocities).zero_()

    def _refresh_positions(self) -> None:
        if self._solver is None:
            return
        positions = self._solver.positions
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
        offset = torch.tensor(self.cfg.origin_world_m, device=positions.device, dtype=positions.dtype)
        self._data.positions_world_m = positions + offset.view(1, 1, 3)

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
            marker_indices[-1] = 1
            self._marker.visualize(translations=translations, marker_indices=marker_indices)


@configclass
class XpbdCatheterAssetCfg(SensorBaseCfg):
    """Configuration for the upstream catheter-vasculature XPBD adapter."""

    class_type: type[XpbdCatheterAsset] = XpbdCatheterAsset
    solver_device: str = "cpu"
    origin_world_m: tuple[float, float, float] = (-0.11, 0.04, 0.08)
    track_direction_world_m: tuple[float, float, float] = (1.0, 0.0, 0.0)
    num_segments: int = 48
    length_m: float = 0.22
    radius_m: float = 0.0005
    max_insertion_m: float = 0.10
    tip_num_edges: int = 8
    tip_bend_rad: float = 0.35
    track_stiffness: float = 0.65
    solver_substeps: int = 4
    guide_path_world_m: tuple[tuple[float, float, float], ...] | None = None
