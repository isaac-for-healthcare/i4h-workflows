# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider binding must precede observation-manager shape discovery."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from i4h_arena.medical.carm import ReferenceProjectionCArmStateProvider, SceneCArmStateProvider
from i4h_arena.medical.navigation_observation import fluoroscopy_rgb
from i4h_arena.medical.patient_twin import PatientTwin
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
def patient():
    # An 8-voxel cube of 100 mm voxels spanning 0-800 mm, so the isocenter is 0.4 m along each axis.
    voxel_to_patient_mm = np.diag([100.0, 100.0, 100.0, 1.0])
    voxel_to_patient_mm[:3, 3] = 50.0
    twin = PatientTwin(
        patient_id="synthetic",
        voxel_to_patient_mm=voxel_to_patient_mm,
        world_from_patient_m=np.eye(4),
        shape_ijk=(8, 8, 8),
        artifacts={},
        source=Path("patient_twin.yaml"),
        patient_mm_from_scan=np.eye(4),
        array_axes="ijk",
    )
    return twin


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
    return scene.make_sensor_view(env)


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("follow_tip", [True, False])
def test_first_observation_binds_patient_geometry_without_a_view(patient, monkeypatch, num_envs, follow_tip):
    twin = patient
    monkeypatch.setenv("I4H_CARM_FOLLOW_TIP", str(int(follow_tip)))
    env = _env(twin, num_envs)
    sensor = env.scene["fluoroscopy"]
    with pytest.raises(RuntimeError, match="requires a bound C-arm provider"):
        _ = sensor.data

    rgb = fluoroscopy_rgb(env)

    assert isinstance(sensor.carm, ReferenceProjectionCArmStateProvider)
    assert sensor.catheter is env.scene["catheter"]
    assert (sensor.carm._tip_source is sensor.catheter) == follow_tip
    assert rgb.shape == (num_envs, 8, 12, 3)
    assert rgb.dtype == torch.uint8
    assert torch.all(rgb == 127)
    rgb.zero_()
    assert torch.all(sensor.rgb == 127), "observation processing must not mutate the sensor buffer"


def test_view_and_repeated_observations_preserve_live_provider_and_pan(patient, monkeypatch):
    twin = patient
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
    np.testing.assert_allclose(provider._pan_m, pan)
    assert not np.allclose(provider.snapshot(1).source_world_m, first_pose.source_world_m)


def test_view_still_binds_phantom_without_image_observations(monkeypatch):
    env = _env(None)
    monkeypatch.setattr("i4h_arena.medical.fluoroscopy_sources._scene_data_carm_provider", lambda *args: None)
    frames = {
        name: SimpleNamespace(get_world_poses=lambda z=z: (np.array([[0.0, 0.0, z]]), np.array([[0.0, 0.0, 0.0, 1.0]])))
        for name, z in (("xray_source", 0.0), ("detector", 1.0))
    }
    # Scene extras are posed through FrameViews over their prims.
    monkeypatch.setattr("i4h_arena.medical.fluoroscopy_sources._frame_view", lambda env, name: frames[name])

    view = _make_view(env, monkeypatch)

    from i4h_arena.medical.fluoroscopy_sources import OrbitCArmStateProvider

    assert isinstance(env.scene["fluoroscopy"].carm, OrbitCArmStateProvider)
    assert isinstance(env.scene["fluoroscopy"].carm._provider, SceneCArmStateProvider)
    assert view.camera("fluoroscopy") is not None


@pytest.mark.parametrize("with_patient", [True, False])
def test_headless_reset_binds_without_views_and_reconstruction_rebinds(patient, monkeypatch, with_patient):
    from i4h_arena.medical.fluoroscopy_sources import initialize_fluoroscopy_sources

    monkeypatch.setattr("i4h_arena.medical.fluoroscopy_sources._scene_data_carm_provider", lambda *args: None)
    frames = {
        name: SimpleNamespace(get_world_poses=lambda z=z: (np.array([[0.0, 0.0, z]]), np.array([[0.0, 0.0, 0.0, 1.0]])))
        for name, z in (("xray_source", 0.0), ("detector", 1.0))
    }
    monkeypatch.setattr("i4h_arena.medical.fluoroscopy_sources._frame_view", lambda env, name: frames[name])
    env = _env(patient if with_patient else None)
    from isaaclab.managers.manager_base import _populate_term_defaults

    params = {}
    _populate_term_defaults(initialize_fluoroscopy_sources, params)
    initialize_fluoroscopy_sources(env, torch.tensor([0]), **params)
    sensor = env.scene["fluoroscopy"]
    assert sensor.catheter is env.scene["catheter"]
    first = sensor.carm
    _ = sensor.data
    initialize_fluoroscopy_sources(env, env_ids=torch.tensor([0]))
    assert sensor.carm is first
    fluoroscopy_rgb(env)
    assert sensor.carm is first
    rebuilt = _env(patient if with_patient else None)
    env.scene, env.action_manager = rebuilt.scene, rebuilt.action_manager
    initialize_fluoroscopy_sources(env, env_ids=torch.tensor([0]))
    assert env.scene["fluoroscopy"].carm is not first
    assert env.scene["fluoroscopy"].catheter is rebuilt.scene["catheter"]
    _ = env.scene["fluoroscopy"].data


def test_nonpatient_adapter_keeps_raw_poses_and_delegates_presets():
    from unittest.mock import Mock

    from i4h_arena.medical.fluoroscopy_sources import OrbitCArmStateProvider

    pose = object()
    raw = SimpleNamespace(snapshot=Mock(return_value=pose))
    orbit = SimpleNamespace(set_orbit_angle=Mock(return_value=0.75))
    provider = OrbitCArmStateProvider(raw, orbit)
    assert provider.snapshot(3) is pose
    raw.snapshot.assert_called_once_with(3)
    assert provider.select_angle(0.75) == 0.75
    orbit.set_orbit_angle.assert_called_once_with(0.75)
