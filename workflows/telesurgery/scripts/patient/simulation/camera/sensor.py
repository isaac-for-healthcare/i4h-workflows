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
            rgba = self._current_frame["rgba"]

            if not rgba.shape[0] == 0 and self.callback is not None:
                rgb = rgba[:, :, :3]
                self.callback(rgb, self.frame_counter)
            self.frame_counter += 1
