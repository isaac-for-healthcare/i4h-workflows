# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docked Kit image window for any camera-compatible :class:`SceneView` sensor."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Sequence
from functools import partial
from typing import Any, TypeVar

import numpy as np

from i4h_common.types import CameraFrame

CONTROL_COLUMNS = 2
_CONTROL_ROW_HEIGHT = 28
_LABEL_WIDTH = 100

T = TypeVar("T")


def control_grid(cells: Sequence[T], columns: int = CONTROL_COLUMNS) -> tuple[tuple[T, ...], ...]:
    """Group settings into rows of ``columns`` so they leave the image more height.

    Settings arrive in the order the window declares them, which is also the order that
    pairs related ones across a row: the two window sliders share a row, as do the two
    combo boxes that choose what is being looked at.
    """
    if columns < 1:
        raise ValueError(f"columns must be positive, got {columns}")
    return tuple(tuple(cells[start : start + columns]) for start in range(0, len(cells), columns))


def frame_rgba(frame: CameraFrame) -> np.ndarray:
    """Decode an RGB8 scene frame into the contiguous RGBA layout Kit expects."""
    if frame.encoding != "rgb8":
        raise ValueError(f"sensor image window requires rgb8, got {frame.encoding!r}")
    expected = frame.height * frame.width * 3
    rgb = np.frombuffer(frame.data, dtype=np.uint8)
    if rgb.size != expected:
        raise ValueError(f"sensor frame contains {rgb.size} bytes; expected {expected}")
    rgb = rgb.reshape(frame.height, frame.width, 3)
    alpha = np.full((frame.height, frame.width, 1), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate((rgb, alpha), axis=-1))


def toggled_output(current: str, pairs: tuple[tuple[str, str], ...]) -> str:
    """Return the opposite output in the pair containing ``current``."""
    for first, second in pairs:
        if current == first:
            return second
        if current == second:
            return first
    return current


def keyboard_event_input_name(event: Any) -> str:
    """Return a Kit keyboard input name across carb API representations."""
    value = event.input
    name = str(getattr(value, "name", value)).rsplit(".", maxsplit=1)[-1]
    return name.removeprefix("KEY_").upper()


class SensorImageWindow:
    """Present a camera-like sensor without coupling its producer to OmniUI."""

    def __init__(
        self,
        *,
        name: str,
        title: str | None = None,
        view: Any,
        outputs: tuple[tuple[str, str], ...] = (("RGB", "rgb"),),
        keyboard_toggles: dict[str, tuple[tuple[str, str], ...]] | None = None,
        projection_presets: tuple[tuple[str, str, float], ...] = (),
        projection_default_index: int = 0,
        appearances: tuple[tuple[str, str], ...] = (),
        display_controls: tuple[Any, ...] = (),
        sliders: tuple[Any, ...] = (),
        controls: dict[str, float] | None = None,
        columns: int = CONTROL_COLUMNS,
        width: int = 560,
        height: int = 620,
    ) -> None:
        from omni import ui

        self.name = name
        self._view = view
        self._outputs = outputs
        self._output_index = 0
        self._output_combo = None
        self._keyboard_toggles = keyboard_toggles or {}
        self._projection_presets = projection_presets
        self._projection_default_index = int(np.clip(projection_default_index, 0, max(0, len(projection_presets) - 1)))
        self._projection_index = self._projection_default_index
        self._projection_combo = None
        self._appearances = appearances
        self._appearance_index = 0
        self._appearance_combo = None
        self._display_controls = display_controls
        self._display_control_sliders: dict[str, Any] = {}
        self._sliders = sliders
        self._controls = controls if controls is not None else {}
        self._control_sliders: dict[str, Any] = {}
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._brightness = 1.0
        self._brightness_slider = None
        self._ui = ui
        self._provider = ui.ByteImageProvider()
        display_title = title or name.replace("_", " ").title()
        self._window = ui.Window(display_title, width=width, height=height)
        quick_cells: list[tuple[str, Any]] = []
        tuning_cells: list[tuple[str, Any]] = []
        if len(outputs) > 1:
            quick_cells.append(("Image mode", self._build_output_cell))
        if appearances:
            quick_cells.append(("Appearance", self._build_appearance_cell))
        if projection_presets:
            quick_cells.append(("C-arm view", self._build_projection_cell))
        for spec in sliders:
            self._controls[spec.control] = float(spec.default) * float(spec.scale)
            quick_cells.append((spec.label, partial(self._build_scene_slider_cell, spec)))
        for spec in display_controls:
            tuning_cells.append((spec.label, partial(self._build_display_control_cell, spec)))
        if any(output.startswith("dsa") for _label, output in outputs):
            tuning_cells.append(("DSA brightness", self._build_brightness_cell))
        if display_controls:
            tuning_cells.append(("Recalibrate window", self._build_recalibrate_cell))
        cells = quick_cells + tuning_cells
        quick_rows = control_grid(quick_cells, columns)
        tuning_rows = control_grid(tuning_cells, columns)
        rows = control_grid(cells, columns)
        self.control_rows = tuple(tuple(label for label, _build in row) for row in rows)

        with self._window.frame, ui.VStack(spacing=6):
            self._build_control_rows(quick_rows, columns)
            if tuning_rows:
                with ui.CollapsableFrame("IMAGE TUNING", collapsed=True, height=0), ui.VStack(spacing=4):
                    self._build_control_rows(tuning_rows, columns)
            ui.ImageWithProvider(self._provider)
        asyncio.ensure_future(self._dock_async(display_title))
        if self._keyboard_toggles or self._projection_presets:
            self._setup_keyboard()

    def _build_output_cell(self) -> None:
        self._ui.Label("Image mode", width=_LABEL_WIDTH)
        self._output_combo = self._ui.ComboBox(0, *(label for label, _output in self._outputs))
        self._output_combo.model.add_item_changed_fn(self._on_output_changed)

    def _build_appearance_cell(self) -> None:
        self._ui.Label("Appearance", width=_LABEL_WIDTH)
        self._appearance_combo = self._ui.ComboBox(
            self._appearance_index,
            *(label for label, _appearance in self._appearances),
        )
        self._appearance_combo.model.add_item_changed_fn(self._on_appearance_changed)

    def _build_projection_cell(self) -> None:
        self._ui.Label("C-arm view", width=_LABEL_WIDTH)
        self._projection_combo = self._ui.ComboBox(
            self._projection_index,
            *(label for label, _key, _angle_rad in self._projection_presets),
        )
        self._projection_combo.model.add_item_changed_fn(self._on_projection_changed)

    def _build_scene_slider_cell(self, spec: Any) -> None:
        slider = self._labelled_slider(spec)
        slider.model.add_value_changed_fn(lambda model, selected=spec: self._on_control_changed(model, selected))
        self._control_sliders[spec.control] = slider

    def _build_display_control_cell(self, spec: Any) -> None:
        slider = self._labelled_slider(spec)
        slider.model.add_value_changed_fn(
            lambda model, selected=spec: self._on_display_control_changed(model, selected)
        )
        self._display_control_sliders[spec.control] = slider

    def _labelled_slider(self, spec: Any) -> Any:
        self._ui.Label(spec.label, width=_LABEL_WIDTH)
        slider = self._ui.FloatSlider(min=spec.minimum, max=spec.maximum, step=spec.step)
        slider.model.set_value(spec.default)
        return slider

    def _build_brightness_cell(self) -> None:
        self._ui.Label("DSA brightness", width=_LABEL_WIDTH)
        self._brightness_slider = self._ui.FloatSlider(min=0.3, max=3.0, step=0.1)
        self._brightness_slider.model.set_value(self._brightness)
        self._brightness_slider.model.add_value_changed_fn(self._on_brightness_changed)

    def _build_recalibrate_cell(self) -> None:
        self._ui.Button("Recalibrate window", clicked_fn=self._recalibrate_display)

    def _build_control_rows(self, rows: Sequence[Sequence[tuple[str, Any]]], columns: int) -> None:
        for row in rows:
            with self._ui.HStack(height=_CONTROL_ROW_HEIGHT, spacing=12):
                for _label, build in row:
                    with self._ui.HStack(spacing=6):
                        build()
                # Keep the columns aligned when a trailing row is short.
                for _ in range(columns - len(row)):
                    self._ui.Spacer()

    def _select_output(self, index: int, *, update_combo: bool = True) -> None:
        self._output_index = index
        if update_combo and self._output_combo is not None:
            self._output_combo.model.get_item_value_model().set_value(index)
        self.update()

    def _on_output_changed(self, model: Any, _item: Any) -> None:
        self._select_output(int(model.get_item_value_model().as_int), update_combo=False)

    def _select_projection(self, index: int, *, update_combo: bool = True) -> None:
        self._projection_index = index
        _label, _key, angle_rad = self._projection_presets[index]
        select = getattr(self._view, "select_sensor_projection", None)
        if callable(select):
            select(self.name, angle_rad)
        if update_combo and self._projection_combo is not None:
            self._projection_combo.model.get_item_value_model().set_value(index)
        self.update()

    def _on_projection_changed(self, model: Any, _item: Any) -> None:
        self._select_projection(int(model.get_item_value_model().as_int), update_combo=False)

    def _select_appearance(self, index: int, *, update_combo: bool = True) -> None:
        self._appearance_index = index
        _label, appearance = self._appearances[index]
        setter = getattr(self._view, "set_sensor_appearance", None)
        if callable(setter):
            setter(self.name, appearance)
        if update_combo and self._appearance_combo is not None:
            self._appearance_combo.model.get_item_value_model().set_value(index)
        self.update()

    def _on_appearance_changed(self, model: Any, _item: Any) -> None:
        self._select_appearance(int(model.get_item_value_model().as_int), update_combo=False)

    def _on_display_control_changed(self, model: Any, spec: Any) -> None:
        setter = getattr(self._view, "set_sensor_display_control", None)
        if callable(setter):
            setter(self.name, spec.control, float(model.as_float))
        self.update()

    def _recalibrate_display(self) -> None:
        """Re-fit the window and return the sliders to the neutral setting it was fitted at."""
        recalibrate = getattr(self._view, "recalibrate_sensor_display", None)
        if callable(recalibrate):
            recalibrate(self.name)
        for spec in self._display_controls:
            slider = self._display_control_sliders.get(spec.control)
            if slider is not None:
                slider.model.set_value(spec.default)
        self.update()

    def on_scene_reset(self) -> None:
        """Restore UI-owned scene controls that have an initial scene pose."""
        if self._projection_presets:
            self._select_projection(self._projection_default_index)
        else:
            self.update()

    def _on_control_changed(self, model: Any, spec: Any) -> None:
        self._controls[spec.control] = float(model.as_float) * float(spec.scale)

    def _setup_keyboard(self) -> None:
        import carb
        import omni.appwindow

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def _on_keyboard_event(self, event: Any, *_args: Any) -> bool:
        import carb

        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            # Kit dispatches to later subscribers while callbacks return True.
            return True
        key = keyboard_event_input_name(event)
        if key in {"-", "MINUS", "NUMPAD_SUBTRACT"}:
            self._adjust_brightness(-0.1)
            return True
        if key in {"=", "+", "EQUAL", "PLUS", "NUMPAD_ADD"}:
            self._adjust_brightness(0.1)
            return True
        for index, (_label, preset_key, _angle_rad) in enumerate(self._projection_presets):
            if key == preset_key:
                self._select_projection(index)
                return True
        pairs = self._keyboard_toggles.get(key)
        if pairs is None:
            # Keep dispatching W/S/A/D/Q/E to the teleop subscriber.
            return True
        current = self._outputs[self._output_index][1]
        selected = toggled_output(current, pairs)
        for index, (_label, output) in enumerate(self._outputs):
            if output == selected:
                self._select_output(index)
                break
        return True

    def _adjust_brightness(self, delta: float) -> None:
        self._set_brightness(self._brightness + delta)

    def _on_brightness_changed(self, model: Any) -> None:
        self._set_brightness(float(model.as_float), update_slider=False)

    def _set_brightness(self, value: float, *, update_slider: bool = True) -> None:
        self._brightness = float(np.clip(value, 0.3, 3.0))
        setter = getattr(self._view, "set_sensor_brightness", None)
        selected = setter(self.name, self._brightness) if callable(setter) else None
        if selected is not None:
            self._brightness = float(selected)
        if update_slider and self._brightness_slider is not None:
            self._brightness_slider.model.set_value(self._brightness)
        self.update()

    async def _dock_async(self, title: str) -> None:
        import omni.kit.app

        image_window = None
        for _ in range(20):
            image_window = self._ui.Workspace.get_window(title)
            if image_window is not None:
                break
            await omni.kit.app.get_app().next_update_async()
        viewport = self._ui.Workspace.get_window("Viewport")
        if image_window is not None and viewport is not None and image_window != viewport:
            image_window.dock_in(viewport, self._ui.DockPosition.RIGHT, 0.4)

    def update(self) -> None:
        _label, output = self._outputs[self._output_index]
        frame = self._view.camera(self.name, output=output)
        if frame is None:
            return
        rgba = frame_rgba(frame)
        self._provider.set_bytes_data(rgba.flatten().data, [frame.width, frame.height])

    def close(self) -> None:
        if self._input is not None and self._keyboard is not None and self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        if self._window is not None:
            self._window.visible = False
            self._window = None
