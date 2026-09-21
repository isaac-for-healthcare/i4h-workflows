# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from i4h_arena.medical.patient_ultrasound import (
    TCP_FROM_IMAGER,
    PatientUltrasoundLayout,
    imager_in_patient,
    surface_height,
)


def test_contact_and_acoustic_frame_roundtrip():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.2], [1, 0, 0.3], [0, 1, 0.2]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    assert surface_height(vertices, faces, [0.2, 0.2]) == pytest.approx(0.22)
    with pytest.raises(ValueError, match="outside"):
        surface_height(vertices, faces, [2, 2])
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.5, 0.7]).as_matrix()
    world[:3, 3] = [0.1, -0.3, 0.8]
    layout = PatientUltrasoundLayout(world, vertices, faces, np.array([0.2, 0.2, 0.22]), (1, 1, 0.1), (0, 0, 0))
    pos, rot = layout.tcp_target()
    actual_face = pos + rot @ TCP_FROM_IMAGER[:3, 3]
    np.testing.assert_allclose(actual_face, [0.2, 0.2, 0.22])
    native_mm, angles = imager_in_patient(pos, Rotation.from_matrix(rot).as_quat(), world)
    np.testing.assert_allclose(world[:3, :3] @ (native_mm / 1000) + world[:3, 3], actual_face)
    np.testing.assert_allclose(
        world[:3, :3] @ Rotation.from_euler("xyz", angles).as_matrix(), rot @ TCP_FROM_IMAGER[:3, :3], atol=1e-10
    )
    assert (rot @ TCP_FROM_IMAGER[:3, :3] @ [0, 0, 1])[2] == pytest.approx(-1)
