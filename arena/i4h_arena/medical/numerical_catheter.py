# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Navigation read/drive adapter over the merged Numerical Sim component.

The action terms own physical-rate integration. This sensor only writes absolute
commands, and resolves all numerical views from the current component binding.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils import configclass

from .catheter import CatheterState
from .catheter_drive import _quat_mul, quat_about_axis, quat_to_w_first, quat_to_xyzw
from .xpbd_catheter import XpbdCatheterAsset, XpbdCatheterAssetCfg


class NumericalCatheterAsset(XpbdCatheterAsset):
    def __init__(self, cfg):
        self._component = None
        super().__init__(cfg)

    def bind_component(self, component):
        self._component = component
        self._refresh_positions()

    def _control(self):
        if self._component is None:
            raise RuntimeError("numerical catheter has not been bound")
        return next(s for s in self._component.systems if s.id == "catheter_control")

    def _refresh_positions(self):
        # Isaac may initialize sensors between invalidating the old solver and
        # building the replacement. Never hand a consumer the previous allocation.
        if self._component is not None and (self._component.sim is None or not self._component.sim.physics.valid):
            self._data.positions_world_m = None
            return
        if self._component is not None:
            self._component.sync_views()
            self._data.positions_world_m = self._component.positions("catheter").view(self._num_envs, -1, 3)

    def snapshot(self, num_envs):
        self._refresh_positions()
        points = self._data.positions_world_m
        if points is None:
            return CatheterState.empty(num_envs)
        if num_envs != points.shape[0]:
            raise ValueError("catheter batch does not match sensor")
        return CatheterState(
            points.detach().cpu().numpy().copy(), np.full(num_envs, points.shape[1]), self.cfg.radius_m
        )

    def proximal_frame(self):
        points = self.data.positions_world_m
        if points is None:
            raise RuntimeError("numerical rod not built")
        control = self._control()
        frames = wp.to_torch(control.rod.frame).view(self._num_envs, -1, 4)
        return points[:, 0], quat_to_w_first(frames[:, 0]), points[:, 1] - points[:, 0]

    def advance(self, commands, dt):
        points = self.data.positions_world_m
        tangent = points[:, 1] - points[:, 0]
        tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-12)
        _, quat, _ = self.proximal_frame()
        position = points[:, 0] + tangent * commands[:, :1] * dt
        quat = _quat_mul(quat_about_axis(tangent, commands[:, 1] * dt), quat)
        self.place_proximal(position, quat, commands, dt)

    def place_proximal(self, positions, orientations, commands, dt):
        control = self._control()
        self._record_spent(commands, dt)
        if control.root_mode == "external_pose":
            # Component is authored at the world origin; commands use that frame.
            control.write_root_poses(positions, quat_to_xyzw(orientations).contiguous())
        # Mounted rods retain rigid ownership. Only relative feed/twist/bend are commands.
        target = wp.to_torch(control.absolute_targets)
        target[:, 0] = self._data.insertion_m
        target[:, 1] = self._data.rotation_rad
        self._component.order_inputs()

    def set_tip_bend(self, angles_rad):
        self._tip_bend_rad = angles_rad
        wp.to_torch(self._control().absolute_targets)[:, 2] = angles_rad
        self._component.order_inputs()

    def report_containment(self, points, env_index=0):
        from .catheter_diagnostics import tube_surface_gaps_m
        from .newton_catheter_physics import containment_report

        spec = self.cfg.rod_spec
        if spec.initial_path_world_m is None or spec.lumen_radii_m is None:
            return {}
        report = containment_report(
            points,
            path_world_m=np.asarray(spec.initial_path_world_m),
            lumen_radii_m=np.asarray(spec.lumen_radii_m),
            segment_length_m=spec.segment_length_m,
        )
        if spec.wants_vessel:
            body = self._component.scene.body("lumen")
            wall = self._component.positions("lumen").view(self._num_envs, -1, 3)[env_index].cpu().numpy()
            gaps = tube_surface_gaps_m(
                points,
                wall,
                body.tree.edges[0],
                body.tree.radii[0],
                spec.radius_m,
                open_root=body.tree.root,
                open_root_neighbor=body.tree.root_neighbor,
            )
            report.update(
                live_worst_penetration_mm=float(gaps.max()) * 1000 if gaps.size else 0.0,
                live_samples_outside=int(np.count_nonzero(gaps > 0)),
            )
        return report

    def reset(self, env_ids=None, env_mask=None):
        SensorBase.reset(self, env_ids=env_ids, env_mask=env_mask)
        if self.is_initialized:
            self._zero_bookkeeping(env_ids)
            self._refresh_positions()


@configclass
class NumericalCatheterAssetCfg(XpbdCatheterAssetCfg):
    class_type: type = NumericalCatheterAsset
    rod_spec: object = None
