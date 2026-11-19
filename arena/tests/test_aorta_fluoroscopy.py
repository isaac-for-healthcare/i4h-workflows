# SPDX-License-Identifier: Apache-2.0
"""Live-state binding, independent detector coordinates, and workflow discovery."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from i4h_arena.medical.catheter import CatheterState
from i4h_arena.medical.synthetic_fluoroscopy import SyntheticFluoroscopyRenderer
from i4h_arena.scenes.endoluminal_aorta_fluoroscopy import AortaStateProvider
from i4h_engine.loader import resolve_workflow


def test_provider_observes_rebound_live_storage_and_snapshots_are_independent():
    component = SimpleNamespace()
    component.q = torch.tensor([[0.0, 0, 1, 9.55, 0, 1]])
    component.commands = torch.zeros((1, 3))
    calls = []
    component.sync_views = lambda: calls.append("sync")
    component.positions = lambda body: component.q
    component.state = lambda: {"catheter_control.targets": component.commands}
    env = SimpleNamespace(num_envs=1, i4h=component)
    env.unwrapped = env
    provider = AortaStateProvider(env)
    first = provider.snapshot(1)
    component.q = torch.tensor([[0.5, 0, 1, 10.05, 0, 1]])
    component.commands = torch.tensor([[0.5, 0.1, 0.2]])
    second = provider.snapshot(1)
    np.testing.assert_allclose(first.positions_world_m[0, -1], [9.55, 0, 1])
    np.testing.assert_allclose(second.positions_world_m[0, -1], [10.05, 0, 1])
    np.testing.assert_allclose(provider.diagnostics()["root_world_m"], [0.5, 0, 1])
    np.testing.assert_allclose(provider.joint_state().pos, [[0.5, 0.1, 0.2]])
    assert len(calls) >= 5
    with pytest.raises(ValueError, match="Expected 1 environments, received 2"):
        provider.snapshot(2)


@pytest.mark.parametrize("x,y", [(9.55, 0.0), (0.5, -1.2), (10.5, 1.2)])
def test_raster_tip_marker_matches_independent_xy_projection_and_drops_z(x, y):
    renderer = SyntheticFluoroscopyRenderer(width=256, height=256, world_bounds_m=(-0.5, 11.0, -2.0, 2.0))
    points = np.asarray([[[0, 0, 1], [x, y, 7]]], dtype=np.float32)
    frame = renderer.render(CatheterState(points, np.array([2])))["guidance"][0]
    ys, xs = np.nonzero(np.all(frame == (245, 255, 245), axis=-1))
    observed = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
    expected = np.array([255 * (x + 0.5) / 11.5, 255 * (2 - y) / 4])
    assert np.linalg.norm(observed - expected) <= 2
    points[..., 2] += 100
    np.testing.assert_array_equal(renderer.render(CatheterState(points, np.array([2])))["guidance"][0], frame)


def test_workflow_has_exactly_the_three_approved_modes():
    from i4h_engine.loader import load_workflow_module

    workflow = load_workflow_module("endoluminal_aorta_fluoroscopy").workflow
    assert set(workflow.modes) == {"idle", "demo", "replay"}
    for mode in workflow.modes:
        resolved = resolve_workflow("endoluminal_aorta_fluoroscopy", mode)
        assert resolved.scene == "endoluminal_aorta_fluoroscopy"
