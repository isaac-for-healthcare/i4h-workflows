# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types

import numpy as np
import pytest

from i4h_arena.ui.sensor_image_window import (
    SensorImageWindow,
    control_grid,
    frame_rgba,
    keyboard_event_input_name,
    toggled_output,
)
from i4h_common.types import CameraFrame

# The order the fluoroscopy panel declares its settings in.
FLUOROSCOPY_CONTROLS = (
    "Image mode",
    "Appearance",
    "C-arm view",
    "Velocity (mm/s)",
    "Window level",
    "Window width",
    "DSA brightness",
    "Recalibrate window",
)


def test_rgb8_frame_is_converted_to_kit_rgba() -> None:
    frame = CameraFrame(
        name="fluoroscopy",
        height=1,
        width=2,
        data=bytes((1, 2, 3, 4, 5, 6)),
        encoding="rgb8",
    )

    rgba = frame_rgba(frame)

    np.testing.assert_array_equal(rgba, np.array([[[1, 2, 3, 255], [4, 5, 6, 255]]], dtype=np.uint8))
    assert rgba.flags.c_contiguous


def test_malformed_frame_is_rejected() -> None:
    frame = CameraFrame(name="bad", height=2, width=2, data=b"short", encoding="rgb8")

    with pytest.raises(ValueError, match="expected 12"):
        frame_rgba(frame)


def test_settings_pair_up_two_to_a_row() -> None:
    rows = control_grid(FLUOROSCOPY_CONTROLS)

    assert rows == (
        ("Image mode", "Appearance"),
        ("C-arm view", "Velocity (mm/s)"),
        ("Window level", "Window width"),
        ("DSA brightness", "Recalibrate window"),
    )


def test_an_odd_setting_count_leaves_a_short_trailing_row() -> None:
    assert control_grid(("Image mode", "Appearance", "C-arm view")) == (
        ("Image mode", "Appearance"),
        ("C-arm view",),
    )
    assert control_grid(()) == ()


def test_a_single_column_keeps_one_setting_per_row() -> None:
    assert control_grid(("Image mode", "Appearance"), columns=1) == (("Image mode",), ("Appearance",))


def test_a_grid_needs_at_least_one_column() -> None:
    with pytest.raises(ValueError, match="columns must be positive"):
        control_grid(("Image mode",), columns=0)


def test_dsa_toggle_preserves_guidance_selection() -> None:
    pairs = (("dsa_guidance", "guidance"), ("dsa", "rgb"))

    assert toggled_output("dsa_guidance", pairs) == "guidance"
    assert toggled_output("guidance", pairs) == "dsa_guidance"
    assert toggled_output("dsa", pairs) == "rgb"
    assert toggled_output("rgb", pairs) == "dsa"
    assert toggled_output("depth", pairs) == "depth"


@pytest.mark.parametrize(
    "value",
    ["W", "w", "KeyboardInput.W", "KEY_W", types.SimpleNamespace(name="W")],
)
def test_keyboard_event_input_name_accepts_string_and_named_input(value) -> None:
    assert keyboard_event_input_name(types.SimpleNamespace(input=value)) == "W"


def test_appearance_selection_forwards_the_named_look_to_the_sensor() -> None:
    requested: list[tuple[str, str]] = []
    window = object.__new__(SensorImageWindow)
    window.name = "fluoroscopy"
    window._appearances = (("Fluoroscopy", "fluoro"), ("X-ray", "xray"))  # noqa: SLF001
    window._appearance_index = 0  # noqa: SLF001
    window._appearance_combo = None  # noqa: SLF001
    window._view = types.SimpleNamespace(  # noqa: SLF001
        set_sensor_appearance=lambda name, appearance: requested.append((name, appearance))
    )
    window.update = lambda: None  # type: ignore[method-assign]

    window._select_appearance(1)  # noqa: SLF001

    assert requested == [("fluoroscopy", "xray")]
    assert window._appearance_index == 1  # noqa: SLF001


def test_appearance_selection_tolerates_a_view_without_the_hook() -> None:
    window = object.__new__(SensorImageWindow)
    window.name = "fluoroscopy"
    window._appearances = (("Fluoroscopy", "fluoro"),)  # noqa: SLF001
    window._appearance_index = 0  # noqa: SLF001
    window._appearance_combo = None  # noqa: SLF001
    window._view = types.SimpleNamespace()  # noqa: SLF001
    window.update = lambda: None  # type: ignore[method-assign]

    window._select_appearance(0)  # noqa: SLF001

    assert window._appearance_index == 0  # noqa: SLF001


def _window_with_readout(readout) -> SensorImageWindow:
    window = object.__new__(SensorImageWindow)
    window.name = "fluoroscopy"
    window._readout = readout  # noqa: SLF001
    window._readout_label = types.SimpleNamespace(text="")  # noqa: SLF001
    return window


def test_the_readout_text_lands_on_the_status_line() -> None:
    window = _window_with_readout(lambda: "Target: 27.4 mm away (arrive within 5 mm)")

    window._update_readout()  # noqa: SLF001

    assert window._readout_label.text == "Target: 27.4 mm away (arrive within 5 mm)"  # noqa: SLF001


def test_a_failing_readout_retires_instead_of_ending_the_run() -> None:
    """A teleop recording is expensive; a broken status line must not cost one."""

    def broken() -> str:
        raise RuntimeError("no termination manager yet")

    window = _window_with_readout(broken)

    window._update_readout()  # noqa: SLF001

    assert window._readout is None  # noqa: SLF001
    window._update_readout()  # noqa: SLF001


def test_the_status_line_refreshes_before_the_first_frame_arrives() -> None:
    window = _window_with_readout(lambda: "Target: waiting for the catheter")
    window._outputs = (("RGB", "rgb"),)  # noqa: SLF001
    window._output_index = 0  # noqa: SLF001
    window._view = types.SimpleNamespace(camera=lambda name, output: None)  # noqa: SLF001

    window.update()

    assert window._readout_label.text == "Target: waiting for the catheter"  # noqa: SLF001


def test_a_window_without_a_readout_has_no_status_line() -> None:
    window = object.__new__(SensorImageWindow)
    window.name = "wrist"
    window._readout = None  # noqa: SLF001
    window._readout_label = None  # noqa: SLF001

    window._update_readout()  # noqa: SLF001


def test_scene_reset_restores_default_projection() -> None:
    window = object.__new__(SensorImageWindow)
    window._projection_presets = (("AP", "1", 0.0), ("LAO-45", "2", -0.785))  # noqa: SLF001
    window._projection_default_index = 1  # noqa: SLF001
    selected: list[int] = []
    window._select_projection = selected.append  # type: ignore[method-assign]  # noqa: SLF001

    window.on_scene_reset()

    assert selected == [1]
