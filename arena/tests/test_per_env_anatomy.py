# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each environment holding its own patient answers as that patient would alone."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from i4h_arena.medical.carm import CArmState, PerPatientCArmStateProvider
from i4h_arena.medical.catheter import CatheterState
from i4h_arena.medical.catheter_drive import LumenClamp, RouteRailedIntroducer
from i4h_arena.medical.navigation_reward import _route_tensors, _segment_radii, project_to_route
from i4h_arena.medical.newton_catheter_physics import CatheterEnvAnatomy, CatheterRodSpec
from i4h_arena.medical.patient_assignment import normalize_patient_twins
from i4h_arena.medical.route_progress import is_per_env_routes, route_for_env

SHORT = ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.10, 0.02, 0.0))
LONG = ((0.0, 0.1, 0.0), (0.0, 0.15, 0.0), (0.03, 0.20, 0.0), (0.06, 0.24, 0.01), (0.10, 0.25, 0.02))
SHORT_RADII = (0.004, 0.003, 0.002)
LONG_RADII = (0.006, 0.005, 0.005, 0.004, 0.003)


def _env(num_envs: int) -> SimpleNamespace:
    return SimpleNamespace(device="cpu", num_envs=num_envs)


def _anatomy(path, radii, length_m: float, name: str = "twin.yaml") -> CatheterEnvAnatomy:
    direction = np.subtract(path[1], path[0])
    return CatheterEnvAnatomy(
        patient_twin_manifest=name,
        origin_world_m=path[0],
        track_direction_world=tuple(direction),
        length_m=length_m,
        initial_path_world_m=path,
        lumen_radii_m=radii,
    )


def test_one_twin_is_shared_and_several_name_each_environment():
    assert normalize_patient_twins(None, 4) is None
    assert normalize_patient_twins("a.yaml", 4) == "a.yaml"
    assert normalize_patient_twins(["a.yaml"], 4) == "a.yaml"
    assert normalize_patient_twins(["a.yaml", "b.yaml"], 2) == ("a.yaml", "b.yaml")
    with pytest.raises(ValueError, match="once per environment"):
        normalize_patient_twins(["a.yaml", "b.yaml"], 3)


def test_routes_are_told_apart_by_nesting_depth():
    assert not is_per_env_routes(SHORT)
    assert is_per_env_routes((SHORT, LONG))
    assert route_for_env((SHORT, LONG), 1) == LONG
    assert route_for_env(SHORT, 1) == SHORT


def test_each_environment_projects_onto_its_own_route():
    points = torch.tensor([[[0.07, 0.01, 0.0], [0.01, 0.0, 0.0]], [[0.04, 0.21, 0.0], [0.09, 0.25, 0.02]]])
    starts, spans, start_arc = _route_tensors(_env(2), (SHORT, LONG))

    arc, lateral, segment = project_to_route(points, starts, spans, start_arc)

    for env_index, route in enumerate((SHORT, LONG)):
        alone = _route_tensors(_env(1), route)
        expected = project_to_route(points[env_index], *alone)
        torch.testing.assert_close(arc[env_index], expected[0])
        torch.testing.assert_close(lateral[env_index], expected[1])
        torch.testing.assert_close(segment[env_index], expected[2])


def test_padding_a_short_route_keeps_its_far_end():
    starts, spans, start_arc = _route_tensors(_env(2), (SHORT, LONG))
    beyond = torch.tensor([[[0.5, 0.5, 0.0]], [[0.0, 0.1, 0.0]]])

    arc, _, _ = project_to_route(beyond, starts, spans, start_arc)

    short_length = float(np.linalg.norm(np.diff(np.asarray(SHORT), axis=0), axis=1).sum())
    assert float(arc[0, 0]) == pytest.approx(short_length, abs=1e-6)
    assert float(arc[1, 0]) == pytest.approx(0.0, abs=1e-6)


def test_navigation_configs_take_routes_of_different_lengths():
    from i4h_arena.envcfg.endoluminal_navigation import (
        navigation_observations_cfg,
        navigation_rewards_cfg,
        navigation_terminations_cfg,
    )

    targets = (SHORT[-1], LONG[-1])
    routes = (SHORT, LONG)

    observations = navigation_observations_cfg(targets, route_world_m=routes)
    rewards = navigation_rewards_cfg(targets, route_world_m=routes, lumen_radii_m=(SHORT_RADII, LONG_RADII))
    terminations = navigation_terminations_cfg(targets, route_world_m=routes)

    assert observations is not None and rewards is not None and terminations is not None
    hash(routes)


def test_lumen_radii_follow_each_environment_route():
    starts, _, _ = _route_tensors(_env(2), (SHORT, LONG))

    radii = _segment_radii(_env(2), (SHORT_RADII, LONG_RADII), int(starts.shape[-2]))

    assert radii.shape == (2, len(LONG) - 1)
    torch.testing.assert_close(radii[0], torch.tensor([0.004, 0.003, 0.002, 0.002]))
    torch.testing.assert_close(radii[1], torch.tensor(LONG_RADII[:-1]))


def test_each_environment_feeds_along_its_own_route():
    shared = RouteRailedIntroducer(
        [torch.tensor(SHORT), torch.tensor(LONG)],
        num_envs=2,
    )
    feed = torch.tensor([1.0, 1.0])
    spent, _ = shared.advance(feed, torch.zeros(2), dt=0.07)
    position, quat = shared.root_target()

    for env_index, route in enumerate((SHORT, LONG)):
        alone = RouteRailedIntroducer(torch.tensor(route), num_envs=1)
        alone_spent, _ = alone.advance(feed[:1], torch.zeros(1), dt=0.07)
        alone_position, alone_quat = alone.root_target()
        torch.testing.assert_close(spent[env_index : env_index + 1], alone_spent)
        torch.testing.assert_close(position[env_index : env_index + 1], alone_position)
        torch.testing.assert_close(quat[env_index : env_index + 1], alone_quat)


def test_feeding_stops_at_each_environment_route_end():
    introducer = RouteRailedIntroducer([torch.tensor(SHORT), torch.tensor(LONG)], num_envs=2)

    for _ in range(20):
        introducer.advance(torch.tensor([1.0, 1.0]), torch.zeros(2), dt=0.1)

    torch.testing.assert_close(introducer.depth_m, introducer.route_lengths_m)
    assert float(introducer.route_lengths_m[0]) < float(introducer.route_lengths_m[1])


def test_one_route_per_environment_must_match_the_environment_count():
    with pytest.raises(ValueError, match="2 routes for 3 environments"):
        RouteRailedIntroducer([torch.tensor(SHORT), torch.tensor(LONG)], num_envs=3)


def test_each_environment_is_held_inside_its_own_lumen():
    clamp = LumenClamp(
        [torch.tensor(SHORT), torch.tensor(LONG)],
        [torch.tensor(SHORT_RADII), torch.tensor(LONG_RADII)],
        margin_m=0.0005,
    )
    points = torch.tensor([[0.06, 0.02, 0.0], [0.02, 0.17, 0.01]])

    held = clamp.project(points)

    for env_index, (route, radii) in enumerate(((SHORT, SHORT_RADII), (LONG, LONG_RADII))):
        alone = LumenClamp(torch.tensor(route), torch.tensor(radii), margin_m=0.0005)
        torch.testing.assert_close(held[env_index : env_index + 1], alone.project(points[env_index : env_index + 1]))


def test_a_lumen_per_environment_needs_one_point_per_environment():
    clamp = LumenClamp(
        [torch.tensor(SHORT), torch.tensor(LONG)],
        [torch.tensor(SHORT_RADII), torch.tensor(LONG_RADII)],
        margin_m=0.0,
    )

    with pytest.raises(ValueError, match="one point per vessel"):
        clamp.project(torch.zeros((3, 3)))


def test_spec_takes_one_anatomy_per_environment():
    anatomies = (_anatomy(SHORT, SHORT_RADII, 0.08, "a.yaml"), _anatomy(LONG, LONG_RADII, 0.12, "b.yaml"))

    spec = CatheterRodSpec(num_envs=2, num_segments=8, tip_length_m=0.02, env_anatomies=anatomies)

    assert spec.per_env_anatomy
    assert spec.env_twin_manifests() == ("a.yaml", "b.yaml")
    np.testing.assert_allclose(spec.env_segment_lengths_m(), [0.01, 0.015])
    positions = spec.env_initial_positions()
    assert positions.shape == (2, 9, 3)
    np.testing.assert_allclose(positions[:, 0], [SHORT[0], LONG[0]], atol=1e-6)
    seeded = np.linalg.norm(np.diff(positions, axis=1), axis=-1).sum(axis=1)
    assert seeded[0] < seeded[1]


def test_spec_rejects_an_anatomy_count_that_is_not_the_environment_count():
    anatomies = (_anatomy(SHORT, SHORT_RADII, 0.08),)

    with pytest.raises(ValueError, match="1 patients for 2 environments"):
        CatheterRodSpec(num_envs=2, tip_length_m=0.02, env_anatomies=anatomies)


def test_spec_tip_has_to_fit_the_shortest_catheter():
    anatomies = (_anatomy(SHORT, SHORT_RADII, 0.03), _anatomy(LONG, LONG_RADII, 0.12))

    with pytest.raises(ValueError, match="tip_length_m"):
        CatheterRodSpec(num_envs=2, tip_length_m=0.05, env_anatomies=anatomies)


class _Provider:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.selected: list[float] = []

    def snapshot(self, num_envs: int) -> CArmState:
        source = np.zeros((num_envs, 3)) + self.offset
        detector = source + np.array([0.0, 0.0, 1.0])
        return CArmState(source, detector, np.tile([1.0, 0.0, 0.0], (num_envs, 1)), (0.3, 0.3))

    def select_angle(self, angle_rad: float) -> float:
        self.selected.append(angle_rad)
        return angle_rad


def test_c_arm_geometry_comes_from_each_environment_patient():
    first, second = _Provider(1.0), _Provider(2.0)
    provider = PerPatientCArmStateProvider([first, second], [np.array([0, 2]), np.array([1])])

    state = provider.snapshot(3)

    np.testing.assert_allclose(state.source_world_m[:, 0], [1.0, 2.0, 1.0])
    assert provider.select_angle(0.5) == 0.5
    assert first.selected == [0.5]


def test_c_arm_patient_groups_must_cover_every_environment_once():
    provider = PerPatientCArmStateProvider([_Provider(1.0)], [np.array([0])])

    with pytest.raises(ValueError, match="once each"):
        provider.snapshot(2)


def test_states_restrict_to_a_patient_group():
    catheter = CatheterState(np.arange(18, dtype=np.float32).reshape(3, 2, 3), np.array([2, 1, 2]))
    carm = _Provider(0.0).snapshot(3)

    taken = catheter.take(np.array([2, 0]))

    np.testing.assert_array_equal(taken.positions_world_m, catheter.positions_world_m[[2, 0]])
    np.testing.assert_array_equal(taken.valid_nodes, [2, 2])
    assert carm.take(np.array([1])).num_envs == 1


def test_synthetic_fluoroscopy_ignores_distinct_patients():
    wp = pytest.importorskip("warp")
    fluoroscopy = pytest.importorskip("i4h_arena.sensors.fluoroscopy", reason="needs Isaac Lab")
    from i4h_arena.medical.synthetic_fluoroscopy import SyntheticFluoroscopyRenderer

    sensor = object.__new__(fluoroscopy.FluoroscopySensor)
    sensor._clear_callbacks = lambda: None
    sensor.cfg = SimpleNamespace(backend="synthetic")
    sensor._num_envs = 2
    sensor._device = "cpu"
    sensor._group_twins = (object(), object())
    sensor._group_env_ids = (np.array([0]), np.array([1]))
    sensor._catheter_provider = None
    sensor._carm_provider = None
    sensor._renderer = SyntheticFluoroscopyRenderer(width=4, height=3)
    sensor._data = fluoroscopy.FluoroscopySensorData(
        output={
            name: torch.zeros((2, 3, 4, channels), dtype=dtype)
            for name, channels, dtype in (
                ("rgb", 3, torch.uint8),
                ("guidance", 3, torch.uint8),
                ("dsa", 3, torch.uint8),
                ("dsa_guidance", 3, torch.uint8),
                ("attenuation", 1, torch.float32),
            )
        },
        frame_id=torch.zeros(2, dtype=torch.int64),
    )

    def _fail(*_args):
        raise AssertionError("the synthetic backend must not build Slang renderers")

    sensor._render_each_patient = _fail
    sensor._update_buffers_impl(wp.array([True, True], dtype=wp.bool, device="cpu"))

    assert sensor._data.frame_id.tolist() == [1, 1]
    assert sensor._data.output["attenuation"].abs().sum() > 0


def test_patient_without_an_anatomy_mesh_shows_none(tmp_path, monkeypatch):
    pxr = pytest.importorskip("pxr", reason="needs USD")
    assets = pytest.importorskip("i4h_arena.assets.fluoroscopy_catheter_navigation", reason="needs Isaac Lab")
    Usd, UsdGeom = pxr.Usd, pxr.UsdGeom

    anatomy = tmp_path / "anatomy.usda"
    mesh_stage = Usd.Stage.CreateNew(str(anatomy))
    mesh_stage.SetDefaultPrim(UsdGeom.Xform.Define(mesh_stage, "/Anatomy").GetPrim())
    mesh_stage.Save()
    stage = Usd.Stage.CreateInMemory()
    paths = ("/World/envs/env_0", "/World/envs/env_1")
    for path in paths:
        UsdGeom.Xform.Define(stage, f"{path}/Patient")
    twins = {
        "with.yaml": SimpleNamespace(artifacts={"anatomy_usd": anatomy}, world_from_patient_m=np.eye(4)),
        "without.yaml": SimpleNamespace(artifacts={}, world_from_patient_m=np.eye(4)),
    }
    monkeypatch.setattr(assets.sim_utils, "get_current_stage", lambda: stage)
    monkeypatch.setattr(assets.PatientTwin, "load", staticmethod(lambda manifest: twins[manifest]))
    env = SimpleNamespace(num_envs=2, scene=SimpleNamespace(env_prim_paths=paths))

    assets.place_patient_anatomies(env, None, ("with.yaml", "without.yaml"))

    shown, hidden = (UsdGeom.Imageable(stage.GetPrimAtPath(f"{path}/Patient")) for path in paths)
    assert shown.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert shown.GetPrim().GetPrimStack()[0].referenceList.prependedItems[0].assetPath == str(anatomy)
    assert hidden.ComputeVisibility() == UsdGeom.Tokens.invisible
