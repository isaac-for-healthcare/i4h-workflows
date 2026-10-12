# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pytest

from i4h_arena.medical.newton_catheter_physics import CatheterRodSpec
from i4h_arena.medical.numerical_catheter import configure_material, navigation_case


def test_material_conversion_and_explicit_runtime():
    spec = CatheterRodSpec()
    cfg = navigation_case(spec, environ={})
    asset = cfg['assets']['catheter']
    rod = next(s for s in cfg['systems'] if s['type'] == 'rod_direct')
    moment = math.pi * spec.radius_m**4 / 4
    assert asset['mass'] == pytest.approx(7800 * math.pi * spec.radius_m**2 * spec.segment_length_m)
    assert rod['bend'] == pytest.approx(4 * 0.1 * moment / spec.segment_length_m**2)
    assert rod['twist'] == pytest.approx(4 * 0.4 * 2 * moment / spec.segment_length_m**2)
    assert rod['inertia'] == 'physical'
    assert cfg['systems'][-1]['tip_mapping'] == 'polyline'
    assert cfg['systems'][-1]['root_mode'] == 'track'
    assert cfg['systems'][-1]['rates'] == [0.060, 1.5, 1.5]


def test_seeded_placement_and_material_are_independent():
    spec = CatheterRodSpec(length_m=0.1, num_segments=30,
                           initial_path_world_m=((0,0,0), (0.07,0,0), (0.07,0.07,0)),
                           track_guidance=True, track_free_distal_length_m=0.03)
    straight = navigation_case(spec, environ={})
    spec.rest_curvature_from_path = True
    seeded = navigation_case(spec, environ={})
    np.testing.assert_array_equal(straight['assets']['catheter']['startup_positions'], seeded['assets']['catheter']['startup_positions'])
    assert not np.any(straight['assets']['catheter']['rest_curvature'])
    assert np.any(seeded['assets']['catheter']['rest_curvature'])
    # Preserve the baseline's int(length / segment) conversion, including float rounding.
    assert seeded['systems'][-1]['distal_free_nodes'] == 8
    assert seeded['systems'][-1]['root_mode'] == 'external_pose'
    spec.drive_body_name = 'hand'
    assert navigation_case(spec, environ={})['systems'][-1]['root_mode'] == 'track'


def test_obsolete_overrides_fail_even_when_empty_or_malformed():
    for value in ['', 'post', 'nonsense']:
        with pytest.raises(ValueError, match='obsolete overrides'):
            navigation_case(CatheterRodSpec(), environ={'I4H_CATHETER_CONTAINMENT': value})
    with pytest.raises(ValueError, match='unsupported legacy'):
        navigation_case(CatheterRodSpec(solver_overrides={'backend': 'scr'}), environ={})


def test_case_runs_and_taper_survives_selected_reset():
    composition = pytest.importorskip('i4h_isaaclab.composition')
    spec = CatheterRodSpec(num_envs=2, device="cpu", num_segments=24, tip_bend_fraction=0.3, vessel_enabled=False)
    cfg = navigation_case(spec, environ={})
    cfg['sim']['cuda_graph'] = False
    sim = composition.load_case(cfg)
    configure_material(sim, spec)
    props = sim.scene.body('catheter').region_properties
    before = props.tables['bend_multiplier'].numpy().copy()
    sim.step()
    sim.reset([0])
    np.testing.assert_array_equal(props.tables['bend_multiplier'].numpy(), before)
    assert before[0, -1] == pytest.approx(0.3)
    assert np.isfinite(sim.scene.q.numpy()).all()


def test_free_distal_nodes_preserve_legacy_discretization():
    spec = CatheterRodSpec(length_m=0.1, num_segments=30, track_free_distal_length_m=0.031)
    assert navigation_case(spec, environ={})['systems'][-1]['distal_free_nodes'] == 9
