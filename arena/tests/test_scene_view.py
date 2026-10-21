# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from i4h_arena.adapters.scene_view import ArenaSceneView


class _ArrayView:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float32)

    def numpy(self):
        return self._value


class _XformAsset:
    def get_world_poses(self):
        return _ArrayView([[1.0, 2.0, 3.0]]), _ArrayView([[0.1, 0.2, 0.3, 0.9]])


class _Robot:
    data = SimpleNamespace(
        root_pos_w=np.array([[0.0, -2.4, 0.7923]], dtype=np.float32),
        root_quat_w=np.array([[0.0, 0.0, 0.7071068, 0.7071068]], dtype=np.float32),
        root_lin_vel_w=np.array([[0.0, 0.1, 0.0]], dtype=np.float32),
    )


def test_object_reads_static_xform_world_pose():
    scene = {"tray": _XformAsset()}
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, num_envs=1))

    state = ArenaSceneView(env, objects=("tray",)).object("tray")

    np.testing.assert_allclose(state.pose.pos, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(state.pose.quat, [[0.9, 0.1, 0.2, 0.3]])
    np.testing.assert_array_equal(state.lin_vel, np.zeros((1, 3), dtype=np.float32))


def test_robot_root_reads_floating_base_state():
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"robot": _Robot()}, num_envs=1))

    state = ArenaSceneView(env).robot_root()

    np.testing.assert_allclose(state.pose.pos, [[0.0, -2.4, 0.7923]])
    np.testing.assert_allclose(state.pose.quat, [[0.7071068, 0.0, 0.0, 0.7071068]])
    np.testing.assert_allclose(state.lin_vel, [[0.0, 0.1, 0.0]])


def test_contact_prefers_filtered_force_matrix():
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=np.array([[[[0.0, 0.0, 0.0]]]], dtype=np.float32),
            net_forces_w=np.array([[[0.0, 0.0, 100.0]]], dtype=np.float32),
        )
    )
    scene = SimpleNamespace(sensors={"contact_robot_table": sensor})
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, num_envs=1))

    assert not ArenaSceneView(env).contact("robot", "table").any()


def test_contact_aggregates_filtered_sensor_family():
    quiet = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=np.zeros((1, 1, 1, 3), dtype=np.float32),
            net_forces_w=np.zeros((1, 1, 3), dtype=np.float32),
        )
    )
    touching = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=np.array([[[[0.0, 0.0, 2.0]]]], dtype=np.float32),
            net_forces_w=np.zeros((1, 1, 3), dtype=np.float32),
        )
    )
    scene = SimpleNamespace(
        sensors={
            "contact_robot_table__pelvis": quiet,
            "contact_robot_table__left_hand": touching,
        }
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, num_envs=1))

    assert ArenaSceneView(env).contact("table", "robot").all()


def test_scene_owned_footprint_is_batched():
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={}, num_envs=2))
    view = ArenaSceneView(env, footprint_half_extents={"table": (0.64, 0.4)})

    np.testing.assert_allclose(
        view.footprint_half_extents("table"),
        [[0.64, 0.4], [0.64, 0.4]],
    )


# --------------------------------------------------------------------------- #
# Diagnostics
#
# Optional per-frame measurements of the simulation, supplied by the scene the
# same way joint state providers are. Only the recorder reads them.
# --------------------------------------------------------------------------- #
class _CountingProvider:
    def __init__(self):
        self.calls = 0

    def diagnostics(self):
        self.calls += 1
        return {"arc_excess_mm": float(self.calls)}


def _bare_env():
    return SimpleNamespace(unwrapped=SimpleNamespace(scene={}, num_envs=1))


def test_a_scene_without_a_provider_reports_nothing():
    """The common case, and it has to cost nothing rather than raise."""
    assert ArenaSceneView(_bare_env()).diagnostics() == {}


def test_the_provider_supplies_the_measurements():
    provider = _CountingProvider()

    values = ArenaSceneView(_bare_env(), diagnostics_provider=provider).diagnostics()

    assert values == {"arc_excess_mm": 1.0}


def test_measurements_are_read_once_per_tick():
    """Reading rod state costs a device-to-host copy, so two calls within one
    frame must not pay for it twice -- nor report two different rods."""
    provider = _CountingProvider()
    view = ArenaSceneView(_bare_env(), diagnostics_provider=provider)

    first = view.diagnostics()
    assert view.diagnostics() == first
    assert provider.calls == 1

    view.invalidate()
    assert view.diagnostics() == {"arc_excess_mm": 2.0}
    assert provider.calls == 2


def test_the_caller_cannot_corrupt_the_cached_measurements():
    provider = _CountingProvider()
    view = ArenaSceneView(_bare_env(), diagnostics_provider=provider)

    view.diagnostics()["arc_excess_mm"] = 999.0

    assert view.diagnostics() == {"arc_excess_mm": 1.0}


def test_explicit_reset_hides_only_selected_termination_values_until_a_step():
    values = np.array([True, True, False])
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            num_envs=3,
            termination_manager=SimpleNamespace(active_terms=("success",), get_term=lambda name: values),
        )
    )
    view = ArenaSceneView(env)
    np.testing.assert_array_equal(view.termination("success"), values)
    view.on_reset(np.array([0]))
    np.testing.assert_array_equal(view.termination("success"), [False, True, False])
    # Rendering or waiting for remote inference does not make stale terms current.
    view.invalidate()
    np.testing.assert_array_equal(view.termination("success"), [False, True, False])
    view.on_step()
    np.testing.assert_array_equal(view.termination("success"), values)
    view.on_reset()
    assert not view.termination("success").any()
    # An automatic reset inside step retains the success pulse for the workflow.
    view.on_step()
    np.testing.assert_array_equal(view.termination("success"), values)


class _ProxyArray:
    """Isaac Lab's ProxyArray shape: an explicit ``torch`` property, and a deprecated forwarding bridge."""

    def __init__(self, value):
        import torch

        self._tensor = torch.as_tensor(value)

    @property
    def torch(self):
        return self._tensor

    def __getattr__(self, name):
        raise AssertionError(f"read through the deprecation bridge: .{name}")

    def __getitem__(self, index):
        raise AssertionError("indexed through the deprecation bridge")


def test_isaac_lab_proxy_arrays_are_read_through_their_torch_view():
    from i4h_arena.adapters.scene_view import _np

    values = _np(_ProxyArray([[1.0, 2.0, 3.0]]))

    assert isinstance(values, np.ndarray)
    assert values.tolist() == [[1.0, 2.0, 3.0]]
