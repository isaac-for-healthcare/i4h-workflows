# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import carb
from isaacsim.sensors.camera import Camera


class CameraEx(Camera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prev_rendering_frame = -1
        self.frame_counter = 0
        self.callback = None

    def set_callback(self, callback):
        self.callback = callback

    def _data_acquisition_callback(self, event: carb.events.IEvent):
        super()._data_acquisition_callback(event)

        if self.callback and self._current_frame["rendering_frame"] != self.prev_rendering_frame:
            self.prev_rendering_frame = self._current_frame["rendering_frame"]

            # rgb annotator is attached to the camera by default in initialize()
            rgba = self.get_rgba()

            if not rgba.shape[0] == 0 and self.callback is not None:
                self.callback(rgba, self.frame_counter)
            self.frame_counter += 1
