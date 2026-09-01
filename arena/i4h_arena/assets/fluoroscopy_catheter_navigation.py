# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patient table, C-arm geometry, and fluoroscopy sensor assets."""

from __future__ import annotations

import math
from copy import deepcopy
from itertools import pairwise
from typing import Any

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.asset import Asset

from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.sensors.fluoroscopy import FluoroscopySensorCfg


class ConfigAsset(Asset):
    """Arena wrapper around an Isaac Lab scene config."""

    def __init__(self, name: str, cfg: Any):
        super().__init__(name=name, tags=["scene"])
        self._cfg = cfg

    def get_object_cfg(self) -> tuple[str, Any]:
        return self.name, self._cfg

    def get_event_cfg(self) -> tuple[str, None]:
        return self.name, None


@configclass
class FluoroscopyCatheterNavigationSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1300.0, color=(0.92, 0.96, 1.0)),
    )
    patient_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PatientTable",
        spawn=sim_utils.CuboidCfg(
            size=(1.55, 0.56, 0.06),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.15, 0.17), metallic=0.15),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.74)),
    )
    patient_table_frame = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PatientTableFrame",
        spawn=sim_utils.CuboidCfg(
            size=(1.26, 0.44, 0.10),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.63, 0.65), metallic=0.28),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.67)),
    )
    patient_table_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PatientTableBase",
        spawn=sim_utils.CuboidCfg(
            size=(0.34, 0.30, 0.64),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.49, 0.51), metallic=0.25),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.42, 0.0, 0.39)),
    )
    patient_table_foot = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PatientTableFoot",
        spawn=sim_utils.CuboidCfg(
            size=(0.82, 0.42, 0.07),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.36, 0.38), metallic=0.3),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.42, 0.0, 0.055)),
    )
    patient = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Patient",
        spawn=sim_utils.CuboidCfg(
            size=(0.34, 0.22, 0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.18, 0.12), opacity=0.35),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.85)),
    )
    carm_pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CArm/Support/Pedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.28, 1.18),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.76, 0.78), metallic=0.2),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.72, 0.68, 0.65)),
    )
    carm_floor_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CArm/Support/FloorBase",
        spawn=sim_utils.CuboidCfg(
            size=(0.52, 0.48, 0.10),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.59, 0.61), metallic=0.25),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.72, 0.68, 0.07)),
    )
    carm_boom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CArm/Support/Boom",
        spawn=sim_utils.CuboidCfg(
            size=(0.72, 0.14, 0.14),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.76, 0.78), metallic=0.2),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.36, 0.62, 0.98)),
    )
    carm_orbit_root = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CArm/Orbit",
        spawn=sim_utils.SphereCfg(radius=0.0001, visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.85)),
    )
    xray_source = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CArm/Orbit/Source",
        spawn=sim_utils.CuboidCfg(
            size=(0.17, 0.13, 0.13),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.76, 0.78), metallic=0.15, emissive_color=(0.08, 0.015, 0.0)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.510)),
    )
    detector = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CArm/Orbit/Detector",
        spawn=sim_utils.CuboidCfg(
            size=(0.36, 0.025, 0.36),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.12, 0.16), metallic=0.25),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.510),
            rot=(math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)),
        ),
    )
    detector_backing = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CArm/Orbit/DetectorBacking",
        spawn=sim_utils.CuboidCfg(
            size=(0.42, 0.055, 0.42),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.76, 0.78), metallic=0.15),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.540),
            rot=(math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)),
        ),
    )
    fluoroscopy = FluoroscopySensorCfg(
        prim_path="{ENV_REGEX_NS}/Fluoroscopy",
        update_period=1.0 / 15.0,
        width=1024,
        height=1024,
        backend="synthetic",
        demo_catheter=False,
        dsa=True,
        visual_style="cinematic",
    )


_CARM_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.76, 0.80, 0.82),
    roughness=0.32,
    metallic=0.15,
)
_BEAM_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.0, 0.75, 0.95),
    emissive_color=(0.0, 0.12, 0.20),
    opacity=0.28,
)
_ROOM_WALL_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.82, 0.86, 0.87),
    roughness=0.72,
)
_ROOM_FLOOR_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.62, 0.69, 0.70),
    roughness=0.48,
    metallic=0.04,
)
_ROOM_MAT_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.34, 0.38, 0.40),
    roughness=0.56,
)
_ROOM_METAL_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.66, 0.71, 0.72),
    roughness=0.35,
    metallic=0.32,
)
_ROOM_CABINET_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.70, 0.76, 0.76),
    roughness=0.52,
)
_ROOM_LIGHT_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.94, 0.97, 1.0),
    emissive_color=(0.72, 0.84, 1.0),
    roughness=0.2,
)


def _room_cuboid(
    name: str,
    *,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    material: sim_utils.PreviewSurfaceCfg,
) -> ConfigAsset:
    return ConfigAsset(
        name,
        AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/CathLab/{name}",
            spawn=sim_utils.CuboidCfg(size=size, visual_material=deepcopy(material)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        ),
    )


def _hospital_room_assets() -> list[ConfigAsset]:
    """Build a quiet clinical room around the procedure equipment."""
    assets = [
        _room_cuboid(
            "room_floor",
            size=(5.0, 4.4, 0.02),
            pos=(0.0, 0.0, 0.01),
            material=_ROOM_FLOOR_MATERIAL,
        ),
        _room_cuboid(
            "procedure_floor_inset",
            size=(2.45, 2.05, 0.008),
            pos=(0.0, 0.05, 0.024),
            material=_ROOM_MAT_MATERIAL,
        ),
        _room_cuboid(
            "rear_wall",
            size=(5.0, 0.08, 2.75),
            pos=(0.0, 2.12, 1.375),
            material=_ROOM_WALL_MATERIAL,
        ),
        _room_cuboid(
            "left_wall",
            size=(0.08, 4.4, 2.75),
            pos=(-2.46, 0.0, 1.375),
            material=_ROOM_WALL_MATERIAL,
        ),
        _room_cuboid(
            "ceiling",
            size=(5.0, 4.4, 0.05),
            pos=(0.0, 0.0, 2.77),
            material=_ROOM_WALL_MATERIAL,
        ),
        _room_cuboid(
            "door",
            size=(0.82, 0.025, 2.14),
            pos=(-1.72, 2.07, 1.07),
            material=_ROOM_CABINET_MATERIAL,
        ),
        _room_cuboid(
            "wall_equipment_rail",
            size=(1.30, 0.055, 0.09),
            pos=(-0.55, 2.04, 1.02),
            material=_ROOM_METAL_MATERIAL,
        ),
        _room_cuboid(
            "storage_cabinet",
            size=(0.88, 0.34, 0.82),
            pos=(1.62, 1.88, 0.41),
            material=_ROOM_CABINET_MATERIAL,
        ),
        _room_cuboid(
            "storage_counter",
            size=(0.98, 0.42, 0.055),
            pos=(1.62, 1.84, 0.85),
            material=_ROOM_METAL_MATERIAL,
        ),
    ]
    for index, x_position in enumerate((-1.45, 0.0, 1.45)):
        assets.append(
            _room_cuboid(
                f"ceiling_light_panel_{index}",
                size=(0.82, 0.32, 0.025),
                pos=(x_position, 0.25, 2.73),
                material=_ROOM_LIGHT_MATERIAL,
            )
        )
    return assets


def _quat_z_to_vector_xyzw(vector: np.ndarray) -> tuple[float, float, float, float]:
    """Return an XYZW quaternion that maps local +Z onto ``vector``."""
    direction = np.asarray(vector, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    dot = float(direction[2])
    if dot < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    quat = np.array((-direction[1], direction[0], 0.0, 1.0 + dot), dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(value) for value in quat)


def _capsule_between(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    radius: float,
    material: sim_utils.PreviewSurfaceCfg,
) -> ConfigAsset:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    delta = end_array - start_array
    length = float(np.linalg.norm(delta))
    return ConfigAsset(
        name,
        AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/CArm/Orbit/{name}",
            spawn=sim_utils.CapsuleCfg(radius=radius, height=length, visual_material=deepcopy(material)),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tuple(float(value) for value in (start_array + end_array) * 0.5),
                rot=_quat_z_to_vector_xyzw(delta),
            ),
        ),
    )


def _carm_visual_assets() -> list[ConfigAsset]:
    """Build a recognizable, license-clean C-arm and its projection rays."""
    # Coordinates are local to the orbital frame at the patient isocenter.
    center_z = -0.01
    horizontal_radius = 0.62
    vertical_radius = 0.53
    samples = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 19)
    points = [
        (
            0.0,
            float(horizontal_radius * math.cos(angle)),
            float(center_z + vertical_radius * math.sin(angle)),
        )
        for angle in samples
    ]
    assets = [
        _capsule_between(
            f"carm_arc_segment_{index:02d}",
            start,
            end,
            radius=0.052,
            material=_CARM_MATERIAL,
        )
        for index, (start, end) in enumerate(pairwise(points))
    ]
    assets.extend(
        (
            _capsule_between(
                "carm_source_mount",
                points[0],
                (0.0, 0.0, -0.510),
                radius=0.050,
                material=_CARM_MATERIAL,
            ),
            _capsule_between(
                "carm_detector_mount",
                points[-1],
                (0.0, 0.0, 0.510),
                radius=0.050,
                material=_CARM_MATERIAL,
            ),
        )
    )

    source = (0.0, 0.0, -0.450)
    for index, (corner_x, corner_y) in enumerate(((-0.18, -0.18), (0.18, -0.18), (0.18, 0.18), (-0.18, 0.18))):
        assets.append(
            _capsule_between(
                f"carm_beam_ray_{index}",
                source,
                (corner_x, corner_y, 0.495),
                radius=0.0025,
                material=_BEAM_MATERIAL,
            )
        )
    return assets


def _quat_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a normalized XYZW quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        following = (axis + 1) % 3
        remaining = (axis + 2) % 3
        scale = 2.0 * np.sqrt(1.0 + matrix[axis, axis] - matrix[following, following] - matrix[remaining, remaining])
        quat = np.zeros(4, dtype=np.float64)
        quat[axis] = 0.25 * scale
        quat[following] = (matrix[following, axis] + matrix[axis, following]) / scale
        quat[remaining] = (matrix[remaining, axis] + matrix[axis, remaining]) / scale
        quat[3] = (matrix[remaining, following] - matrix[following, remaining]) / scale
    quat /= np.linalg.norm(quat)
    return tuple(float(value) for value in quat)


def _patient_asset(fallback: AssetBaseCfg, manifest: str | None) -> AssetBaseCfg:
    if manifest is None:
        return fallback
    twin = PatientTwin.load(manifest)
    anatomy = twin.artifacts.get("anatomy_usd")
    if anatomy is None:
        return fallback
    transform = twin.world_from_patient_m
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Patient",
        spawn=sim_utils.UsdFileCfg(usd_path=str(anatomy)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(value) for value in transform[:3, 3]),
            rot=_quat_xyzw(transform[:3, :3]),
        ),
    )


def make_assets(
    *,
    fluoro_backend: str = "synthetic",
    fluoro_device: str = "vulkan",
    patient_twin_manifest: str | None = None,
) -> list[ConfigAsset]:
    """Return the patient/C-arm world and custom image sensor."""
    source = FluoroscopyCatheterNavigationSceneCfg(env_spacing=4.0)
    source.fluoroscopy.backend = fluoro_backend
    source.fluoroscopy.dsa = fluoro_backend == "slang"
    source.fluoroscopy.slang_device_type = fluoro_device
    source.fluoroscopy.patient_twin_manifest = patient_twin_manifest
    source.patient = _patient_asset(source.patient, patient_twin_manifest)
    names = (
        "ground",
        "light",
        "patient_table",
        "patient_table_frame",
        "patient_table_base",
        "patient_table_foot",
        "patient",
        "carm_pedestal",
        "carm_floor_base",
        "carm_boom",
        "carm_orbit_root",
        "xray_source",
        "detector_backing",
        "detector",
        "fluoroscopy",
    )
    assets = [ConfigAsset(name, deepcopy(getattr(source, name))) for name in names]
    assets.extend(_hospital_room_assets())
    assets.extend(_carm_visual_assets())
    return assets
