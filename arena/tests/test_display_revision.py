# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Frame identity is scoped to a sensor, output, and displayed environment."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from i4h_arena.ui.sensor_image_window import SensorImageWindow
from i4h_arena.ui.sensor_view import SensorView


def sensor():
    return SimpleNamespace(
        display_revision=[1, 1],
        data=SimpleNamespace(output={key: np.zeros((2, 2, 3, 3), dtype=np.uint8) for key in ("rgb", "dsa")}),
    )


def test_poll_keeps_identity_peer_updates_do_not_download_and_replacement_refreshes():
    first = sensor()
    scene = {"image": first}
    view = SensorView(SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)), ("image",))
    frame = view.camera("image")
    for _ in range(4):
        first.display_revision[1] += 1
        view.invalidate(force=False)
        assert view.camera("image") is frame
    first.display_revision[0] += 1
    first.data.output["rgb"][0] = 12
    view.invalidate(force=False)
    newer = view.camera("image")
    assert newer is not frame
    assert newer.to_array().min() == 12
    assert view.camera("image", output="dsa") is not newer
    scene["image"] = sensor()
    assert view.camera("image") is not newer
    prior = view.camera("image")
    view.invalidate()
    assert view.camera("image") is not prior


def test_nonrevision_sensor_keeps_per_step_fallback():
    raw = sensor()
    del raw.display_revision
    view = SensorView(SimpleNamespace(unwrapped=SimpleNamespace(scene={"image": raw})), ("image",))
    frame = view.camera("image")
    assert view.camera("image") is frame
    view.invalidate(force=False)
    assert view.camera("image") is not frame


@pytest.mark.parametrize(
    "method,args,hook,result",
    [
        ("set_sensor_brightness", (2.0,), "set_dsa_brightness", 2.0),
        ("set_sensor_appearance", ("xray",), "set_display_appearance", "xray"),
        ("set_sensor_display_control", ("window_width", 2.0), "set_display_control", 2.0),
        ("recalibrate_sensor_display", (), "recalibrate_display", True),
        ("select_sensor_projection", (0.5,), "select_projection", 0.5),
    ],
)
def test_display_controls_force_invalidation(method, args, hook, result):
    raw = sensor()
    setattr(raw, hook, Mock(return_value=result))
    view = SensorView(SimpleNamespace(unwrapped=SimpleNamespace(scene={"image": raw})), ("image",))
    frame = view.camera("image")
    assert getattr(view, method)("image", *args) == result
    assert view.camera("image") is not frame


def test_repeated_frame_uploads_once_and_keeps_readouts_live():
    raw = sensor()
    view = SensorView(SimpleNamespace(unwrapped=SimpleNamespace(scene={"image": raw})), ("image",))
    window = object.__new__(SensorImageWindow)
    window.name = "image"
    window._view = view
    window._outputs = (("RGB", "rgb"), ("DSA", "dsa"))
    window._output_index = 0
    window._output_combo = None
    window._last_frame = None
    window._provider = Mock()
    window._readout = Mock(return_value="active")
    window._readout_label = SimpleNamespace(text="")
    for _ in range(5):
        window.update()
    assert window._provider.set_data_array.call_count == 1
    assert window._readout.call_count == 5
    window._select_output(1)
    assert window._provider.set_data_array.call_count == 2
    window._projection_presets = ()
    window.on_scene_reset()
    assert window._provider.set_data_array.call_count == 3


def test_sensor_absent_during_construction_or_teardown_returns_no_frame():
    host = SimpleNamespace(scene={})
    view = SensorView(SimpleNamespace(unwrapped=host), ("image",))
    assert view.camera("image") is None
    view.invalidate(force=False)
    host.scene["image"] = sensor()
    assert view.camera("image") is not None
    del host.scene
    assert view.camera("image") is None
    view.invalidate(force=False)


def test_step_polling_preserves_lazy_sampling_without_reading_gpu_frame_counter():
    class LazySensor:
        display_revision = [0]
        pending = False
        renders = 0

        @property
        def data(self):
            if self.pending:
                self.pending = False
                self.renders += 1
                self.display_revision[0] += 1
            return SimpleNamespace(output={"rgb": np.zeros((1, 2, 3, 3), dtype=np.uint8)})

    raw = LazySensor()
    view = SensorView(SimpleNamespace(unwrapped=SimpleNamespace(scene={"image": raw})), ("image",))
    prior = view.camera("image")
    for step in range(10):
        raw.pending = step % 2 == 1
        view.invalidate(force=False)
        frame = view.camera("image")
        assert (frame is prior) == (step % 2 == 0)
        for _ in range(5):
            assert view.camera("image") is frame
        prior = frame
    assert raw.renders == 5


def test_only_displayed_environment_and_rgb_channels_are_transferred(monkeypatch):
    import i4h_arena.ui.sensor_view as module

    raw = sensor()
    raw.data.output["rgb"] = np.arange(8 * 2 * 3 * 4, dtype=np.uint8).reshape(8, 2, 3, 4)
    original = module._np
    shapes = []

    def transfer(value):
        shapes.append(value.shape)
        return original(value)

    monkeypatch.setattr(module, "_np", transfer)
    view = SensorView(SimpleNamespace(unwrapped=SimpleNamespace(scene={"image": raw})), ("image",))
    frame = view.camera("image")
    np.testing.assert_array_equal(frame.to_array(), raw.data.output["rgb"][0, ..., :3])
    assert shapes == [(2, 3, 3)]
    assert view.camera("image") is frame
    assert len(shapes) == 1
