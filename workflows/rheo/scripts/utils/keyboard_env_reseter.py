# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import weakref


def disable_terminations_and_recorders(env_cfg):
    """Disable all terminations and recorders in the environment configuration.

    This allows the environment to run indefinitely without automatic resets,
    and prevents recorder assertions when manually resetting.
    Matches the approach from replay_demos.py.
    """
    # Disable all recorders (prevents assertion errors on reset)
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = {}
        print("[INFO] Disabled all recorders")

    # Disable all terminations
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = {}
        print("[INFO] Disabled all terminations")


class KeyboardHandler:
    """Handles keyboard input for simulation control.

    Note: Must be instantiated after SimulationAppContext is active.
    """

    def __init__(self):
        # Lazy import to make sure the appwindow is initialized
        import carb.input
        import omni.appwindow

        self.reset_pressed = False
        self._carb_input = carb.input  # Store module reference for use in callbacks
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Callback for keyboard events."""
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            if event.input == self._carb_input.KeyboardInput.R:
                self.reset_pressed = True
                print("\n[INFO] Reset requested via keyboard")
        return True

    def check_reset(self) -> bool:
        """Check if reset was requested and clear the flag."""
        if self.reset_pressed:
            self.reset_pressed = False
            return True
        return False

    def close(self):
        """Unsubscribe from keyboard events."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
