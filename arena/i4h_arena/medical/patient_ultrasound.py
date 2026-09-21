# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scan-aligned anatomy placement and calibrated ultrasound probe geometry."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

# Measured in the maintained Franka HD3 C3_Assy visual mesh relative to TCP.
# Long aperture is along TCP -Y; positive TCP Z points into the patient.
TCP_FROM_IMAGER = np.eye(4)
TCP_FROM_IMAGER[:3, :3] = Rotation.from_euler("z", -90, degrees=True).as_matrix()
TCP_FROM_IMAGER[:3, 3] = [-0.00033357, -0.00184759, -0.00243516]


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def surface_height(vertices, faces, xy):
    """Highest intersection of a vertical line with a triangle mesh."""
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    v2 = np.asarray(xy) - a[:, :2]
    determinant = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    valid = np.abs(determinant) > 1e-12
    u = np.divide(v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1], determinant, out=np.zeros(len(a)), where=valid)
    v = np.divide(v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1], determinant, out=np.zeros(len(a)), where=valid)
    inside = valid & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
    if not inside.any():
        raise ValueError("Probe target lies outside the skin silhouette")
    z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
    return float(z[inside].max())


@dataclass
class PatientUltrasoundLayout:
    world_from_patient_m: np.ndarray
    skin_vertices_m: np.ndarray
    skin_faces: np.ndarray
    target_position_m: np.ndarray
    table_size_m: tuple
    table_position_m: tuple

    def tcp_target(self, lateral_m=0.0):
        xy = self.target_position_m[:2] + [0, lateral_m]
        contact = np.r_[xy, surface_height(self.skin_vertices_m, self.skin_faces, xy)]
        # The center of the real face is on skin; curved peripheral elements use
        # the simulator's finite gel/contact distance instead of penetrating it.
        rotation = Rotation.from_euler("x", np.pi).as_matrix()
        tcp = contact - rotation @ TCP_FROM_IMAGER[:3, 3]
        return tcp, rotation


def patient_layout(twin):
    from ultrasound_simulator.usd import read_usd_meshes

    meshes = read_usd_meshes(twin.artifacts["anatomy_usd"])
    skin = next((m for m in meshes if m.name == "SOMA"), None)
    liver = next((m for m in meshes if m.name == "liver"), None)
    if skin is None or liver is None:
        raise ValueError("Patient ultrasound needs visible SOMA exterior and liver meshes")
    # Keep supine orientation, with the long axis along the table Y direction,
    # so the complete body stays beside the Franka base at the world origin.
    placement = np.eye(4)
    placement[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    skin_rot = skin.vertices_mm * 0.001 @ placement[:3, :3].T
    liver_center = (liver.vertices_mm.min(0) + liver.vertices_mm.max(0)) * 0.0005
    center_rot = placement[:3, :3] @ liver_center
    placement[:2, 3] = np.array([0.55, -0.05]) - center_rot[:2]
    placement[2, 3] = 0.003 - skin_rot[:, 2].min()
    points = transform_points(skin.vertices_mm * 0.001, placement)
    lower, upper = points.min(0), points.max(0)
    target = transform_points(liver_center, placement)
    target[2] = surface_height(points, skin.faces, target[:2])
    return PatientUltrasoundLayout(
        placement,
        points,
        skin.faces,
        target,
        (float(upper[0] - lower[0] + 0.10), float(upper[1] - lower[1] + 0.10), 0.06),
        (float((lower[0] + upper[0]) / 2), float((lower[1] + upper[1]) / 2), -0.03),
    )


def imager_in_patient(tcp_position_m, tcp_quaternion_xyzw, world_from_patient_m):
    world_from_tcp = np.eye(4)
    world_from_tcp[:3, :3] = Rotation.from_quat(tcp_quaternion_xyzw).as_matrix()
    world_from_tcp[:3, 3] = tcp_position_m
    patient_from_imager = np.linalg.inv(world_from_patient_m) @ world_from_tcp @ TCP_FROM_IMAGER
    return patient_from_imager[:3, 3] * 1000, Rotation.from_matrix(patient_from_imager[:3, :3]).as_euler("xyz")
