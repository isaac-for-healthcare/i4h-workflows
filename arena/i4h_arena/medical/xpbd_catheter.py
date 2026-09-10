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

import os
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
from .catheter_drive import quat_to_w_first, quat_to_xyzw
from .newton_catheter_physics import require_active_handle
from .newton_providers import NewtonRodCatheterStateProvider


# The narrowest lumen on the routes we drive is about 3 mm in radius. A marker
# wider than that cannot be drawn inside the vessel at all, so it reads as wall
# perforation exactly where the anatomy is tight -- which is where the question
# of containment is live and where the render most needs to be trusted. Keep the
# inflation that makes a sub-millimetre wire visible, cap it under the lumen.
_SHAFT_MARKER_CAP_M = 0.0015
_TIP_MARKER_CAP_M = 0.003


PROBE_ENV_VAR = "I4H_CATHETER_PROBE"


def probe_interval(environ: Any = None) -> int:
    """Steps between containment reports, from ``I4H_CATHETER_PROBE``.

    Zero when unset or unreadable, because a malformed diagnostic setting should
    not take a simulator run down with it.

    Read once when the asset is built rather than per frame: unset is the common
    case and it reaches this through a raised ``ValueError``, which is not
    something to pay for on every step of a recording.
    """
    raw = (environ if environ is not None else os.environ).get(PROBE_ENV_VAR, "")
    try:
        interval = int(raw)
    except (TypeError, ValueError):
        return 0
    return interval if interval > 0 else 0


def marker_radius_m(radius_m: float, *, inflation: float, cap_m: float) -> float:
    """Marker radius: inflated for legibility, then capped below the lumen.

    Never below the tool's own radius, so a thick catheter is not drawn thinner
    than it really is.
    """
    if not radius_m > 0.0:
        raise ValueError(f"radius_m must be positive, got {radius_m}")
    if not inflation >= 1.0:
        raise ValueError(f"inflation must be at least 1, got {inflation}")
    return max(float(radius_m), min(inflation * float(radius_m), float(cap_m)))


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
        # Last commanded tip shape, for the recorded state. ``None`` until a
        # steer arrives, which reads as an unbent tip.
        self._tip_bend_rad: torch.Tensor | None = None
        self._probe_step = 0
        self._probe_every = probe_interval()
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
        self._record_spent(command, dt)

    def set_tip_bend(self, angles_rad: torch.Tensor) -> None:
        """Shape the distal tip to an absolute bend angle per environment.

        Separate from :meth:`advance` because this is a shape and not a rate:
        the solver holds it as the tip edges' rest curvature, so re-sending the
        same angle is a no-op rather than a further bend.

        The tensor is handed over on the device it arrives on, so steering every
        step does not synchronize.
        """
        angles = angles_rad.detach().to(device=self._device, dtype=torch.float32).reshape(-1)
        if angles.numel() != self._num_envs:
            raise ValueError(f"tip bend must have {self._num_envs} angles, got {angles.numel()}")
        rod = require_active_handle().rod
        steer = getattr(rod, "set_tip_bend", None)
        if steer is None:
            raise RuntimeError(f"rod solver {type(rod).__name__} cannot steer its tip")
        steer(angles)
        # Kept so the shape is recordable. A policy that commands the bend has
        # to be able to see it, and the solver holds it as rest curvature
        # spread over the tip edges rather than as anything readable back.
        self._tip_bend_rad = angles

    def proximal_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """The rod's proximal pose and the direction its first segment runs.

        What a mount needs in order to aim at the wire's entry: where the
        proximal particle is, how it is twisted, and which way the wire leaves
        it. ``None`` until Newton has finalized the model, because before that
        there are no particles and any pose here would be invented.

        Returns:
            ``(position, quaternion, tangent)`` in the world frame with the
            quaternion w-first, matching what
            :meth:`~i4h_arena.medical.catheter_drive.FlangeMountedIntroducer.root_target`
            consumes. The solver stores xyzw, so it is reordered on the way out.
        """
        positions = self.data.positions_world_m
        if positions is None or positions.shape[1] < 2:
            return None
        root_pos = positions[:, 0]
        # Direction to the neighbour rather than a normalized axis: the caller
        # normalizes and has its own fallback for a collapsed first segment,
        # which the containment projection can produce.
        tangent = positions[:, 1] - positions[:, 0]

        orientations = require_active_handle().rod.orientations
        if orientations.ndim == 2:  # single env: (num_points, 4)
            orientations = orientations.unsqueeze(0)
        root_quat = quat_to_w_first(orientations[:, 0].to(device=self._device, dtype=torch.float32))
        return root_pos, root_quat, tangent

    def place_proximal(
        self,
        position_world_m: torch.Tensor,
        quat_world: torch.Tensor,
        commands: torch.Tensor,
        dt: float,
    ) -> None:
        """Hold the wire's proximal end at a pose its mount has chosen.

        The arm-driven counterpart to :meth:`advance`. That method feeds the rod
        at a rate along its own tangent and lets the root's position accumulate;
        this one hands the solver the mount's pose directly, so a hand that
        stalls against a joint limit or is held back by contact stops the wire
        by exactly as much as it stopped itself.

        ``commands`` is still the feed actually spent, because the recorded
        virtual joints have to advance the same way in both drives -- the pose
        says where the root is, not how much wire went in.

        Args:
            position_world_m: ``(N, 3)`` world position for the proximal particle.
            quat_world: ``(N, 4)`` world orientation for it, w-first.
            commands: ``(N, 2)`` insertion and rotation rates actually spent.
            dt: Control-step duration in seconds.
        """
        pos = position_world_m.detach().to(device=self._device, dtype=torch.float32)
        quat = quat_world.detach().to(device=self._device, dtype=torch.float32)
        if pos.shape != (self._num_envs, 3):
            raise ValueError(f"catheter root position must have shape ({self._num_envs}, 3), got {tuple(pos.shape)}")
        if quat.shape != (self._num_envs, 4):
            raise ValueError(
                f"catheter root orientation must have shape ({self._num_envs}, 4), got {tuple(quat.shape)}"
            )
        command = commands.detach().to(device=self._device, dtype=torch.float32)
        if command.shape != (self._num_envs, 2):
            raise ValueError(f"catheter command must have shape ({self._num_envs}, 2), got {tuple(command.shape)}")

        require_active_handle().rod.set_root_pose_gpu(pos, quat_to_xyzw(quat))
        self._record_spent(command, dt)

    def _record_spent(self, command: torch.Tensor, dt: float) -> None:
        """Integrate the spent feed into the recorded virtual joints."""
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
        names = ("insertion_m", "rotation_rad", "tip_bend_rad")
        if self._data.insertion_m is None or self._data.rotation_rad is None or self._data.command is None:
            zeros = np.zeros((self._num_envs, 3), dtype=np.float32)
            return JointState(pos=zeros, vel=zeros.copy(), names=names)
        bend = self._tip_bend_rad
        if bend is None:
            bend = torch.zeros_like(self._data.insertion_m)
        pos = (
            torch.stack((self._data.insertion_m, self._data.rotation_rad, bend.reshape(-1)), dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        # The bend is commanded as a shape rather than a rate, so it has no
        # velocity of its own on this channel; differencing ``pos`` recovers one.
        rates = self._data.command.detach().cpu().numpy()
        vel = np.zeros_like(pos)
        vel[:, : rates.shape[1]] = rates
        return JointState(
            pos=pos.astype(np.float32, copy=False),
            vel=vel.astype(np.float32, copy=False),
            names=names,
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
            if self._tip_bend_rad is not None:
                self._tip_bend_rad = None
            return
        self._data.insertion_m[env_ids] = 0.0
        self._data.rotation_rad[env_ids] = 0.0
        self._data.command[env_ids] = 0.0
        if self._tip_bend_rad is not None:
            # The action term owns the angle and clears its own copy on reset;
            # this is the recorded mirror, cleared for the same envs.
            self._tip_bend_rad[env_ids] = 0.0

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
        self._log_probe()

    def _log_probe(self) -> None:
        """Report containment and chord spread, when the probe is switched on.

        Off by default: the report brings particles to the host, which is a sync
        the hot path should not pay for a diagnostic nobody asked for.
        """
        if self._probe_every <= 0 or self._data.positions_world_m is None:
            return
        self._probe_step += 1
        if self._probe_step % self._probe_every:
            return
        report = require_active_handle().report_containment(
            self._data.positions_world_m.reshape(self._num_envs, -1, 3)[0].detach().cpu().numpy()
        )
        if report is None:
            return
        print(
            f"[catheter probe] step {self._probe_step}  "
            f"worst penetration {report['worst_penetration_mm']:+.2f} mm  "
            f"outside {report['particles_outside']}/{report['num_particles']}  "
            f"chords {report['chord_min_pct']:.0f}-{report['chord_max_pct']:.0f}%",
            flush=True,
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
                            radius=marker_radius_m(self.cfg.radius_m, inflation=3.0, cap_m=_SHAFT_MARKER_CAP_M),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.85, 0.95)),
                        ),
                        "tip": sim_utils.SphereCfg(
                            radius=marker_radius_m(self.cfg.radius_m, inflation=6.0, cap_m=_TIP_MARKER_CAP_M),
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
