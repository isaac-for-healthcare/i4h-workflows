# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-axis catheter embodiment backed by the i4h Warp XPBD solver."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, ClassVar

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from i4h_arena.medical.centerline import ordered_centerline_path
from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_volume import PatientVolume
from i4h_arena.medical.xpbd_catheter import XpbdCatheterAsset, XpbdCatheterAssetCfg
from i4h_common.types import JointState


def reference_initial_catheter_length_m(twin: PatientTwin, *, fallback_m: float) -> float:
    """Match the reference viewport's 15%-to-80% CT-width initialization."""
    metadata_path = twin.artifacts.get("volume_metadata")
    if metadata_path is None:
        return float(fallback_m)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape_zyx = np.asarray(metadata.get("shape_zyx"), dtype=np.float64)
    spacing_zyx_mm = np.asarray(metadata.get("spacing_zyx_mm"), dtype=np.float64)
    if shape_zyx.shape != (3,) or spacing_zyx_mm.shape != (3,):
        raise ValueError("volume metadata must contain three-value shape_zyx and spacing_zyx_mm")
    length_m = 0.65 * float(shape_zyx[2] * spacing_zyx_mm[2]) * 0.001
    if not np.isfinite(length_m) or length_m <= 0.0:
        raise ValueError("volume metadata produces an invalid catheter initialization length")
    return min(float(fallback_m), length_m)


class CatheterVelocityAction(ActionTerm):
    """Proximal insertion velocity and axial rotation rate in SI units."""

    cfg: CatheterVelocityActionCfg

    def __init__(self, cfg: CatheterVelocityActionCfg, env: Any):
        super().__init__(cfg, env)
        if not isinstance(self._asset, XpbdCatheterAsset):
            raise TypeError(f"asset {cfg.asset_name!r} must be XpbdCatheterAsset")
        self._raw_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        self._processed_actions[:, 0] = torch.clamp(
            actions[:, 0], -float(self.cfg.max_insertion_velocity_mps), float(self.cfg.max_insertion_velocity_mps)
        )
        self._processed_actions[:, 1] = torch.clamp(
            actions[:, 1], -float(self.cfg.max_rotation_rate_radps), float(self.cfg.max_rotation_rate_radps)
        )

    def apply_actions(self) -> None:
        self._asset.advance(self._processed_actions, float(self._env.physics_dt))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class CatheterVelocityActionCfg(ActionTermCfg):
    class_type: type[CatheterVelocityAction] = CatheterVelocityAction
    asset_name: str = "catheter"
    max_insertion_velocity_mps: float = 0.030
    max_rotation_rate_radps: float = 1.5


class CArmOrbitAction(ActionTerm):
    """Rotate the complete source-detector assembly about the patient long axis."""

    cfg: CArmOrbitActionCfg

    def __init__(self, cfg: CArmOrbitActionCfg, env: Any):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._angle_rad = torch.full(
            (self.num_envs,), float(cfg.initial_orbit_angle_rad), device=self.device, dtype=torch.float32
        )
        self._root_position = torch.tensor(cfg.isocenter_world_m, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._root_orientation = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self._write_pose()

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def angle_rad(self) -> torch.Tensor:
        return self._angle_rad

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(
            torch.clamp(actions, -float(self.cfg.max_orbit_rate_radps), float(self.cfg.max_orbit_rate_radps))
        )

    def apply_actions(self) -> None:
        self._angle_rad.add_(self._processed_actions[:, 0] * float(self._env.physics_dt))
        self._angle_rad.clamp_(float(self.cfg.min_orbit_angle_rad), float(self.cfg.max_orbit_angle_rad))
        self._write_pose()

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._raw_actions[selected] = 0.0
        self._processed_actions[selected] = 0.0
        self._angle_rad[selected] = float(self.cfg.initial_orbit_angle_rad)
        self._write_pose()

    def joint_state(self) -> JointState:
        return JointState(
            pos=self._angle_rad[:, None].detach().cpu().numpy().astype(np.float32, copy=False),
            vel=self._processed_actions.detach().cpu().numpy().astype(np.float32, copy=False),
            names=("carm_orbit_rad",),
        )

    def set_orbit_angle(self, angle_rad: float) -> float:
        """Set a named projection angle immediately and return the clamped value."""
        selected = float(np.clip(angle_rad, self.cfg.min_orbit_angle_rad, self.cfg.max_orbit_angle_rad))
        self._angle_rad.fill_(selected)
        self._processed_actions.zero_()
        self._write_pose()
        return selected

    def _write_pose(self) -> None:
        half_angle = 0.5 * self._angle_rad
        self._root_orientation[:, 0] = torch.sin(half_angle)
        self._root_orientation[:, 1:3] = 0.0
        self._root_orientation[:, 3] = torch.cos(half_angle)
        self._asset.set_local_poses(
            translations=self._root_position,
            orientations=self._root_orientation,
        )


@configclass
class CArmOrbitActionCfg(ActionTermCfg):
    class_type: type[CArmOrbitAction] = CArmOrbitAction
    asset_name: str = "carm_orbit_root"
    isocenter_world_m: tuple[float, float, float] = (0.0, 0.0, 0.85)
    initial_orbit_angle_rad: float = math.pi / 4.0
    max_orbit_rate_radps: float = 0.6
    min_orbit_angle_rad: float = -math.pi / 6.0
    max_orbit_angle_rad: float = math.pi / 2.0


class CatheterCArmJointStateProvider:
    """Record catheter virtual joints and C-arm angle as one procedure state."""

    def __init__(self, catheter: XpbdCatheterAsset, carm_orbit: CArmOrbitAction) -> None:
        self._catheter = catheter
        self._carm_orbit = carm_orbit

    def joint_state(self) -> JointState:
        catheter = self._catheter.joint_state()
        carm = self._carm_orbit.joint_state()
        return JointState(
            pos=np.concatenate((catheter.pos, carm.pos), axis=-1),
            vel=np.concatenate((catheter.vel, carm.vel), axis=-1),
            names=(*catheter.names, *carm.names),
        )


@configclass
class _CatheterSceneCfg:
    catheter_root = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Catheter",
        spawn=sim_utils.SphereCfg(radius=0.0001, visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.11, 0.04, 0.68)),
    )
    catheter = XpbdCatheterAssetCfg(
        prim_path="{ENV_REGEX_NS}/Catheter",
        update_period=0.0,
        origin_world_m=(-0.11, 0.04, 0.68),
        debug_vis=True,
    )


@configclass
class _ActionsCfg:
    catheter = CatheterVelocityActionCfg()
    carm_orbit = CArmOrbitActionCfg()


class CatheterEmbodiment:
    """Minimal Arena embodiment whose scene entity is an external rod solver."""

    name = "catheter"
    tags: ClassVar[list[str]] = ["embodiment", "medical", "catheter"]

    def __init__(self, patient_twin_manifest: str | None = None) -> None:
        self.scene_config = _CatheterSceneCfg()
        self.action_config = _ActionsCfg()
        if patient_twin_manifest is not None:
            self._align_to_patient_centerline(PatientTwin.load(patient_twin_manifest))

    def _align_to_patient_centerline(self, twin: PatientTwin) -> None:
        patient = PatientVolume.load(twin)
        isocenter = patient.volume_mm_to_world(patient.center_xyz_mm)
        self.action_config.carm_orbit.isocenter_world_m = tuple(float(value) for value in isocenter)
        centerline_path = twin.artifacts.get("centerline_points")
        if centerline_path is None:
            return
        points_patient_mm = np.load(centerline_path)
        edges_path = twin.artifacts.get("centerline_edges")
        if edges_path is None:
            return
        edges = np.load(edges_path)
        radii_path = twin.artifacts.get("centerline_radii")
        radii = np.load(radii_path) if radii_path is not None else None
        path_patient_mm = ordered_centerline_path(
            points_patient_mm,
            edges,
            target_spacing_mm=7.5,
            radii_mm=radii,
        )
        path_world_m = twin.patient_mm_to_world(path_patient_mm)
        path_segments = np.linalg.norm(np.diff(path_world_m, axis=0), axis=1)
        path_length = float(np.sum(path_segments))
        length = reference_initial_catheter_length_m(twin, fallback_m=path_length)
        start = path_world_m[0]
        direction = path_world_m[1] - path_world_m[0]
        direction /= np.linalg.norm(direction)

        origin = tuple(float(value) for value in start)
        track_direction = tuple(float(value) for value in direction)
        self.scene_config.catheter_root.init_state.pos = origin
        self.scene_config.catheter.origin_world_m = origin
        self.scene_config.catheter.track_direction_world_m = track_direction
        self.scene_config.catheter.length_m = length
        self.scene_config.catheter.num_segments = 40
        self.scene_config.catheter.guide_path_world_m = tuple(
            tuple(float(value) for value in point) for point in path_world_m
        )

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_action_cfg(self) -> Any:
        return self.action_config

    def get_observation_cfg(self) -> None:
        return None

    def get_events_cfg(self) -> None:
        return None

    def get_rewards_cfg(self) -> None:
        return None

    def get_curriculum_cfg(self) -> None:
        return None

    def get_commands_cfg(self) -> None:
        return None

    def get_xr_cfg(self) -> None:
        return None

    def get_recorder_term_cfg(self) -> None:
        return None

    def get_termination_cfg(self) -> None:
        return None

    def modify_env_cfg(self, env_cfg: Any) -> Any:
        env_cfg.sim.dt = 1.0 / 120.0
        # The reference catheter viewport advances controls at 30 Hz. Keep the
        # same control period while retaining 120 Hz XPBD substeps.
        env_cfg.decimation = 4
        env_cfg.sim.render_interval = 4
        env_cfg.scene.replicate_physics = False
        return env_cfg
