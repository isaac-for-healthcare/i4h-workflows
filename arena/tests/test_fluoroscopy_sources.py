# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider binding must precede observation-manager shape discovery."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from i4h_arena.medical.carm import ReferenceProjectionCArmStateProvider, SceneCArmStateProvider
from i4h_arena.medical.navigation_observation import fluoroscopy_rgb
from i4h_arena.medical.patient_volume import PatientVolume
from i4h_arena.scenes.endoluminal_navigation import EndoluminalNavigationScene


class _Sensor:
    """A lazy image read has the same provider precondition as Slang."""

    def __init__(self, patient_twin, num_envs):
        self.patient_twin = patient_twin
        self.catheter = None
        self.carm = None
        self.rgb = torch.full((num_envs, 8, 12, 3), 127, dtype=torch.uint8)

    @property
    def state_providers_bound(self):
        return self.catheter is not None and self.carm is not None

    def bind_catheter_provider(self, provider):
        self.catheter = provider

    def bind_carm_provider(self, provider):
        self.carm = provider

    @property
    def data(self):
        if not self.state_providers_bound:
            raise RuntimeError("the slang fluoroscopy backend requires a bound C-arm provider")
        self.carm.snapshot(len(self.rgb))
        return SimpleNamespace(output={"rgb": self.rgb})


@pytest.fixture
def patient(monkeypatch):
    twin = SimpleNamespace(coordinate_frame="DICOM_LPS", world_from_patient_m=np.eye(4))
    transform = np.diag([0.001, 0.001, 0.001, 1.0])
    volume = PatientVolume(
        twin=twin,
        mu_volume=np.zeros((8, 8, 8), dtype=np.float32),
        spacing_zyx_mm=(100.0, 100.0, 100.0),
        volume_xyz_mm_to_world_m=transform,
        world_m_to_volume_xyz_mm=np.linalg.inv(transform),
    )
    load = Mock(return_value=volume)
    monkeypatch.setattr(PatientVolume, "load", load)
    return twin, load


def _env(twin, num_envs=1):
    sensor = _Sensor(twin, num_envs)
    catheter = SimpleNamespace(data=SimpleNamespace(positions_world_m=torch.zeros(num_envs, 2, 3)))
    orbit = SimpleNamespace(angle_rad=torch.zeros(num_envs))
    env = SimpleNamespace(
        num_envs=num_envs,
        scene={"fluoroscopy": sensor, "catheter": catheter},
        action_manager=SimpleNamespace(get_term={"carm_orbit": orbit}.__getitem__),
    )
    env.unwrapped = env
    return env


def _make_view(env, monkeypatch):
    spec = SimpleNamespace(objects=(), robots=("robot",), cameras=("fluoroscopy",))
    scene = EndoluminalNavigationScene(spec, SimpleNamespace())
    monkeypatch.setattr(scene, "_joint_state_providers", lambda *args: {})
    return scene.make_view(env)


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("follow_tip", [True, False])
def test_first_observation_binds_patient_geometry_without_a_view(patient, monkeypatch, num_envs, follow_tip):
    twin, load = patient
    monkeypatch.setenv("I4H_CARM_FOLLOW_TIP", str(int(follow_tip)))
    env = _env(twin, num_envs)
    sensor = env.scene["fluoroscopy"]
    with pytest.raises(RuntimeError, match="requires a bound C-arm provider"):
        _ = sensor.data

    rgb = fluoroscopy_rgb(env)

    assert isinstance(sensor.carm, ReferenceProjectionCArmStateProvider)
    assert sensor.catheter is env.scene["catheter"]
    assert (sensor.carm._tip_source is sensor.catheter) == follow_tip
    load.assert_called_once_with(twin)
    assert rgb.shape == (num_envs, 8, 12, 3)
    assert rgb.dtype == torch.uint8
    assert torch.all(rgb == 127)
    rgb.zero_()
    assert torch.all(sensor.rgb == 127), "observation processing must not mutate the sensor buffer"


def test_view_and_repeated_observations_preserve_live_provider_and_pan(patient, monkeypatch):
    twin, load = patient
    monkeypatch.setenv("I4H_CARM_FOLLOW_TIP", "1")
    env = _env(twin)
    fluoroscopy_rgb(env)
    sensor = env.scene["fluoroscopy"]
    provider = sensor.carm
    first_pose = provider.snapshot(1)
    pan = provider._pan_m.copy()
    assert pan[0] > 0.0

    # Move the tip inside the panned field. Recreating the provider would now
    # choose a different pan, losing the frame's hysteresis at view creation.
    env.scene["catheter"].data.positions_world_m[:, -1, 2] = 0.4 - float(pan[0])
    view = _make_view(env, monkeypatch)
    env.action_manager.get_term("carm_orbit").angle_rad.fill_(0.5)
    fluoroscopy_rgb(env)

    assert view.camera("fluoroscopy") is not None
    assert sensor.carm is provider
    load.assert_called_once_with(twin)
    np.testing.assert_allclose(provider._pan_m, pan)
    assert not np.allclose(provider.snapshot(1).source_world_m, first_pose.source_world_m)


def test_view_still_binds_phantom_without_image_observations(monkeypatch):
    env = _env(None)
    monkeypatch.setattr("i4h_arena.medical.fluoroscopy_sources._scene_data_carm_provider", lambda *args: None)
    for name, z in (("xray_source", 0.0), ("detector", 1.0)):
        env.scene[name] = SimpleNamespace(
            get_world_poses=lambda z=z: (np.array([[0.0, 0.0, z]]), np.array([[0.0, 0.0, 0.0, 1.0]]))
        )

    view = _make_view(env, monkeypatch)

    assert isinstance(env.scene["fluoroscopy"].carm, SceneCArmStateProvider)
    assert view.camera("fluoroscopy") is not None
