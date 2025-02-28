import argparse
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg
import numpy as np
import rti.connextdds as dds
from PIL import Image
from rti_dds.schemas.camera_ctrl import CameraCtrlInput
from rti_dds.schemas.camera_info import CameraInfo
from rti_dds.schemas.franka_ctrl import FrankaCtrlInput
from rti_dds.schemas.franka_info import FrankaInfo
from rti_dds.schemas.target_ctrl import TargetCtrlInput
from rti_dds.schemas.target_info import TargetInfo
from rti_dds.schemas.usp_data import UltraSoundProbeData
from rti_dds.schemas.usp_info import UltraSoundProbeInfo
from rti_dds.subscriber import SubscriberWithCallback
from simulation.configs.config import CameraConfig, Topic
from simulation.utils.common import colorize_depth, get_exp_config, list_exp_configs

parser = argparse.ArgumentParser(formatter_class=argparse.MetavarTypeHelpFormatter)
parser.add_argument("-c", "--config", type=str, choices=list_exp_configs(), default="basic")
config = get_exp_config(parser.parse_args().config)


rng = np.random.default_rng(config.random_seed)


class SimulatorApp:
    """A visualization application for robotic ultrasound simulation."""

    # Class-level constants
    DEFAULT_WINDOW_NAME: str = "OV Holoscan - Demo App"
    DEFAULT_RESIZEABLE: bool = True
    DEFAULT_CAMERA_RANGE_START: float = 20.0
    DEFAULT_CAMERA_RANGE_END: float = 200.0
    CAMERA_MODES: List[str] = ["RGB", "DEPTH"]
    TARGET_MOVE_MODES: List[str] = ["Random", "Y-Axis"]
    CAMERA_ZOOM_STEP: float = 0.2
    MOUSE_WHEEL_STEP: float = 5.0
    WINDOW_PADDING: int = 60
    WINDOW_HEIGHT_PADDING: int = 300
    TEXT_WIDTH: int = 530
    BORDER_COLORS = {"room_camera": [255, 0, 0, 255], "wrist_camera": [0, 255, 0, 255], "ultrasound": [0, 0, 255, 255]}

    def __init__(self) -> None:
        """Initialize the SimulatorApp with default settings."""
        self.window_name: str = self.DEFAULT_WINDOW_NAME
        self.window_width: int = (
            config.room_camera.width + config.wrist_camera.width + config.ultrasound.width + self.WINDOW_PADDING
        )
        self.window_height: int = (
            max(config.room_camera.height, config.wrist_camera.height) + self.WINDOW_HEIGHT_PADDING
        )
        self.resizeable: bool = self.DEFAULT_RESIZEABLE

        # Camera range settings
        self.room_camera_range_start: float = (
            min(config.room_camera.range) if config.room_camera.range else self.DEFAULT_CAMERA_RANGE_START
        )
        self.room_camera_range_end: float = (
            max(config.room_camera.range) if config.room_camera.range else self.DEFAULT_CAMERA_RANGE_END
        )

        # Initialize image data arrays
        self._init_image_data()

        # Initialize DDS related attributes
        self._init_dds_attributes()

    def _init_image_data(self) -> None:
        """Initialize image data arrays for cameras and ultrasound."""
        self.room_camera_image_data: np.ndarray = np.zeros(
            (config.room_camera.height, config.room_camera.width, 4), dtype=np.float32
        )
        self.wrist_camera_image_data: np.ndarray = np.zeros(
            (config.wrist_camera.height, config.wrist_camera.width, 4), dtype=np.float32
        )
        self.ultrasound_image_data: np.ndarray = np.zeros(
            (config.ultrasound.height, config.ultrasound.width, 4), dtype=np.float32
        )

    def _init_dds_attributes(self) -> None:
        """Initialize DDS-related attributes."""
        self.dds_writers: Dict[str, dds.DataWriter] = {}
        self.room_camera_fetched: bool = False

        # Stream Data
        self.current_target_position: Optional[List[float]] = None
        self.current_target_orientation: Optional[List[float]] = None
        self.current_ultrasound_position: Optional[List[float]] = None
        self.current_ultrasound_orientation: Optional[List[float]] = None
        self.current_joints_state_positions: Optional[List[float]] = None
        self.current_joints_state_velocities: Optional[List[float]] = None

        # Stoppable Subscribers
        self.sub_room_camera_rgb: Optional[SubscriberWithCallback] = None
        self.sub_room_camera_depth: Optional[SubscriberWithCallback] = None
        self.sub_wrist_camera_rgb: Optional[SubscriberWithCallback] = None
        self.sub_wrist_camera_depth: Optional[SubscriberWithCallback] = None
        self.sub_ultrasound_image: Optional[SubscriberWithCallback] = None

    # Application setup
    def create_app(self) -> None:
        """Create and setup the application window."""
        dpg.create_context()
        dpg.create_viewport(
            title=self.window_name,
            width=self.window_width,
            height=self.window_height,
            resizable=self.resizeable,
        )
        dpg.setup_dearpygui()
        self.create_app_body()
        dpg.show_viewport()

    def create_app_body(self) -> None:
        """Create the main application body with all widgets."""
        with dpg.texture_registry(show=False):
            self.add_texture_widget(config.room_camera, self.room_camera_image_data, "room_camera_image_data")
            self.add_texture_widget(config.wrist_camera, self.wrist_camera_image_data, "wrist_camera_image_data")
            self.add_texture_widget(config.ultrasound, self.ultrasound_image_data, "ultrasound_image_data")

        with dpg.window(tag="Main Window"):
            with dpg.group(horizontal=True):
                # Room Camera
                if config.room_camera and config.room_camera.enabled:
                    self.add_room_camera_widget()

                # Wrist Camera
                if config.wrist_camera and config.wrist_camera.enabled:
                    self.add_wrist_camera_widget()

                # Ultrasound - Widgets
                if config.ultrasound and config.ultrasound.enabled:
                    self.add_ultrasound_widget()

            # Target - Widgets
            if config.target.enabled:
                self.add_target_widget(text_width=self.TEXT_WIDTH)

            # Franka - Widgets
            if config.franka.enabled:
                self.add_franka_widget(text_width=self.TEXT_WIDTH)

        dpg.set_primary_window("Main Window", True)

        # Register Handlers
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.mouse_wheel_event)
            dpg.add_key_press_handler(callback=self.key_press_event)

    # UI components
    def add_texture_widget(self, config, value, tag, format=dpg.mvFormat_Float_rgba) -> None:
        """Create a texture widget for a camera or image."""
        dpg.add_raw_texture(width=config.width, height=config.height, default_value=value, tag=tag, format=format)

    def add_room_camera_widget(self) -> None:
        """Add room camera widget to the GUI."""
        with dpg.group(horizontal=False):
            dpg.add_checkbox(
                label=" Room Camera",
                tag="streaming_room_camera",
                callback=self.on_streaming_room_camera,
                default_value=True,
            )
            dpg.add_image("room_camera_image_data", border_color=self.BORDER_COLORS["room_camera"])
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Ground Truth")
                dpg.add_combo(
                    items=self.CAMERA_MODES,
                    default_value=self.CAMERA_MODES[0],
                    width=100,
                    tag="room_camera_mode",
                    callback=self.on_streaming_room_camera,
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Focal Length")
                dpg.add_slider_float(
                    tag="room_camera_zoom",
                    default_value=self.room_camera_range_start,
                    min_value=self.room_camera_range_start,
                    max_value=self.room_camera_range_end,
                    width=160,
                    callback=self.on_update_focal_length,
                )

    def add_wrist_camera_widget(self) -> None:
        """Add wrist camera widget to the GUI."""
        with dpg.group(horizontal=False):
            dpg.add_checkbox(
                label=" Wrist Camera",
                tag="streaming_wrist_camera",
                callback=self.on_streaming_wrist_camera,
                default_value=True,
            )
            dpg.add_image("wrist_camera_image_data", border_color=self.BORDER_COLORS["wrist_camera"])
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Ground Truth")
                dpg.add_combo(
                    items=self.CAMERA_MODES,
                    default_value=self.CAMERA_MODES[0],
                    width=100,
                    tag="wrist_camera_mode",
                    callback=self.on_streaming_wrist_camera,
                )

    def add_ultrasound_widget(self) -> None:
        """Add ultrasound widget to the GUI."""
        with dpg.group(horizontal=False):
            dpg.add_checkbox(
                label=" Ultrasound Image",
                tag="streaming_ultrasound",
                callback=self.on_streaming_ultrasound,
                default_value=True,
            )
            dpg.add_image("ultrasound_image_data", border_color=self.BORDER_COLORS["ultrasound"])
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Probe Position (world):")
                dpg.add_input_text(
                    tag="ultrasound_position",
                    default_value="[0.0, 0.0, 0.0]",
                    readonly=True,
                    width=200,
                )

    def add_target_widget(self, text_width: int) -> None:
        """Add target control widget to the GUI."""
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text("Target Positions (world):    ")
            dpg.add_input_text(tag="target_position", readonly=True, width=text_width)
        with dpg.group(horizontal=True):
            dpg.add_text("Target Orientations (world): ")
            dpg.add_input_text(tag="target_orientation", readonly=True, width=text_width)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(
                label=" Move Target By",
                tag="move_target",
                callback=self.publish_target_annotations,
            )
            dpg.add_combo(
                self.TARGET_MOVE_MODES, tag="move_target_by", width=100, default_value=self.TARGET_MOVE_MODES[1]
            )

    def add_franka_widget(self, text_width: int) -> None:
        """Add Franka robot control widget to the GUI."""
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text("Franka Joint Positions:      ")
            dpg.add_input_text(tag="joints_state_positions", readonly=True, width=text_width)
        with dpg.group(horizontal=True):
            dpg.add_text("Franka Joint Velocities:     ")
            dpg.add_input_text(tag="joints_state_velocities", readonly=True, width=text_width)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(
                label=" Move Franka By Target", tag="move_franka", callback=self.publish_franka_annotations
            )

    # Event handlers
    def mouse_wheel_event(self, sender: int, app_data: int) -> None:
        """Handle mouse wheel events for camera zoom."""
        new_value = dpg.get_value("room_camera_zoom") + (app_data * self.MOUSE_WHEEL_STEP)
        new_value = max(min(new_value, self.room_camera_range_end), self.room_camera_range_start)
        dpg.set_value("room_camera_zoom", new_value)
        self.on_update_focal_length("room_camera_zoom", None)

    def key_press_event(self, sender: Any, app_data: Any) -> None:
        """Handle keyboard events for camera zoom."""
        if dpg.is_key_down(dpg.mvKey_Plus):
            new_value = min(dpg.get_value("room_camera_zoom") + self.CAMERA_ZOOM_STEP, self.room_camera_range_end)
            dpg.set_value("room_camera_zoom", new_value)
            self.on_update_focal_length("room_camera_zoom", None)
        if dpg.is_key_down(dpg.mvKey_Minus):
            new_value = max(dpg.get_value("room_camera_zoom") - self.CAMERA_ZOOM_STEP, self.room_camera_range_start)
            dpg.set_value("room_camera_zoom", new_value)
            self.on_update_focal_length("room_camera_zoom", None)

    def on_update_focal_length(self, sender: Any, app_data: Any) -> None:
        """Update camera focal length based on UI controls."""
        new_value = dpg.get_value(sender)
        new_value = max(min(new_value, self.room_camera_range_end), self.room_camera_range_start)
        self.publish_camera_annotations({"focal_len": new_value})

    # DDS connection
    def connect_to_dds(self) -> None:
        """Initialize all DDS connections."""
        self.on_streaming_room_camera()
        self.on_streaming_wrist_camera()
        self.on_streaming_ultrasound()

        # Franka
        if config.franka and config.franka.enabled:
            if config.franka.topic_ctrl:
                self.connect_to_dds_publisher(config.franka.topic_ctrl, FrankaCtrlInput)
            if config.franka.topic_info:
                self.connect_to_dds_subscriber(config.franka.topic_info, FrankaInfo, self.on_receive_franka_annotations)

        # Target
        if config.target and config.target.enabled:
            if config.target.topic_ctrl:
                self.connect_to_dds_publisher(config.target.topic_ctrl, TargetCtrlInput)
            if config.target.topic_info:
                self.connect_to_dds_subscriber(config.target.topic_info, TargetInfo, self.on_receive_target_annotations)

        # UltraSound
        if config.ultrasound and config.ultrasound.enabled:
            if config.ultrasound.topic_info:
                self.connect_to_dds_subscriber(
                    config.ultrasound.topic_info,
                    UltraSoundProbeInfo,
                    self.on_receive_ultrasound_annotations,
                )

    def connect_to_dds_publisher(self, topic: Topic, cls) -> None:
        """Connect to a DDS publisher."""
        if self.dds_writers.get(topic.name, None) is None:
            print(f"\nPublishing to topic: {topic}")
            p = dds.DomainParticipant(domain_id=topic.domain_id)
            writer = dds.DataWriter(dds.Topic(p, topic.name, cls))
            self.dds_writers[topic.name] = writer

    def connect_to_dds_subscriber(self, topic: Topic, cls, cb) -> SubscriberWithCallback:
        """Connect to a DDS subscriber."""
        print(f"\nSubscribing to topic: {topic}")
        s = SubscriberWithCallback(cb=cb, domain_id=topic.domain_id, topic=topic.name, cls=cls, period=topic.period)
        s.start()
        return s

    # Streaming control
    def on_streaming_room_camera(self) -> None:
        """Handle room camera streaming control."""
        if self.sub_room_camera_rgb is not None:
            self.sub_room_camera_rgb.stop()
        if self.sub_room_camera_depth is not None:
            self.sub_room_camera_depth.stop()

        if dpg.get_value("streaming_room_camera"):
            self.connect_to_dds_publisher(config.room_camera.topic_ctrl, CameraCtrlInput)

        if dpg.get_value("room_camera_mode") == "RGB":
            self.sub_room_camera_rgb = self.on_streaming_xyz(
                tag="streaming_room_camera",
                sub=self.sub_room_camera_rgb,
                topic=config.room_camera.topic_data_rgb,
                cls=CameraInfo,
                cb=self.on_receive_camera_annotations,
                dv_tag="room_camera_image_data",
                dv_val=self.room_camera_image_data,
            )
        else:
            self.sub_room_camera_depth = self.on_streaming_xyz(
                tag="streaming_room_camera",
                sub=self.sub_room_camera_depth,
                topic=config.room_camera.topic_data_depth,
                cls=CameraInfo,
                cb=self.on_receive_camera_annotations,
                dv_tag="room_camera_image_data",
                dv_val=self.room_camera_image_data,
            )

    def on_streaming_wrist_camera(self) -> None:
        """Handle wrist camera streaming control."""
        if self.sub_wrist_camera_rgb is not None:
            self.sub_wrist_camera_rgb.stop()
        if self.sub_wrist_camera_depth is not None:
            self.sub_wrist_camera_depth.stop()

        if dpg.get_value("wrist_camera_mode") == "RGB":
            self.sub_wrist_camera_rgb = self.on_streaming_xyz(
                tag="streaming_wrist_camera",
                sub=self.sub_wrist_camera_rgb,
                topic=config.wrist_camera.topic_data_rgb,
                cls=CameraInfo,
                cb=self.on_receive_camera_annotations,
                dv_tag="wrist_camera_image_data",
                dv_val=self.wrist_camera_image_data,
            )
        else:
            self.sub_wrist_camera_depth = self.on_streaming_xyz(
                tag="streaming_wrist_camera",
                sub=self.sub_wrist_camera_depth,
                topic=config.wrist_camera.topic_data_depth,
                cls=CameraInfo,
                cb=self.on_receive_camera_annotations,
                dv_tag="wrist_camera_image_data",
                dv_val=self.wrist_camera_image_data,
            )

    def on_streaming_ultrasound(self) -> None:
        """Handle ultrasound streaming control."""
        if self.sub_ultrasound_image is not None:
            self.sub_ultrasound_image.stop()

        self.sub_ultrasound_image = self.on_streaming_xyz(
            tag="streaming_ultrasound",
            sub=self.sub_ultrasound_image,
            topic=config.ultrasound.topic_data,
            cls=UltraSoundProbeData,
            cb=self.on_receive_ultrasound_image,
            dv_tag="ultrasound_image_data",
            dv_val=self.ultrasound_image_data,
        )

    def on_streaming_xyz(self, tag, sub, topic, cls, cb, dv_tag, dv_val) -> Optional[SubscriberWithCallback]:
        """Generic streaming control helper."""
        if not dpg.get_value(tag):
            if sub:
                sub.stop()
                np.multiply(dv_val, 0.0, out=dv_val)
                dpg.set_value(dv_tag, dv_val)
                print(f"{tag}: {topic} => Stopped!")
            return sub

        if dpg.get_value(tag):
            if not sub:
                sub = self.connect_to_dds_subscriber(topic, cls, cb)
            sub.start()
            print(f"{tag}: {topic} => Started!")
        return sub

    # Data receiving
    def on_receive_camera_annotations(self, topic_name: str, s: CameraInfo) -> None:
        """Handle received camera annotations from DDS."""
        if config.room_camera and config.room_camera.enabled:
            if config.room_camera.topic_data_rgb and config.room_camera.topic_data_rgb.name == topic_name:
                return self.on_camera_annotations(s, config.room_camera, True)
            if config.room_camera.topic_data_depth and config.room_camera.topic_data_depth.name == topic_name:
                return self.on_camera_annotations(s, config.room_camera, True)

        if config.wrist_camera and config.wrist_camera.enabled:
            if config.wrist_camera.topic_data_rgb and config.wrist_camera.topic_data_rgb.name == topic_name:
                return self.on_camera_annotations(s, config.wrist_camera, False)
            if config.wrist_camera.topic_data_depth and config.wrist_camera.topic_data_depth.name == topic_name:
                return self.on_camera_annotations(s, config.wrist_camera, False)

    def on_camera_annotations(self, s: CameraInfo, c: CameraConfig, is_room: bool) -> None:
        """Process camera annotations."""
        h = c.height
        w = c.width
        mode = dpg.get_value("room_camera_mode" if is_room else "wrist_camera_mode")
        if mode == "DEPTH":
            img_array = np.frombuffer(s.data, dtype=np.float32).reshape(h, w, 1)
            try:
                img_array = colorize_depth(img_array)
            except:  # noqa: E722
                print(traceback.format_exc())
        else:
            img_array = np.frombuffer(s.data, dtype=np.uint8).reshape(h, w, 4)

        if is_room:
            np.divide(img_array, 255.0, out=self.room_camera_image_data)
            dpg.set_value("room_camera_image_data", self.room_camera_image_data)
            if not self.room_camera_fetched:
                self.room_camera_fetched = True
                dpg.set_value("room_camera_zoom", s.focal_len)
        else:
            np.divide(img_array, 255.0, out=self.wrist_camera_image_data)
            dpg.set_value("wrist_camera_image_data", self.wrist_camera_image_data)

    def on_receive_ultrasound_image(self, topic_name: str, s: UltraSoundProbeData) -> None:
        """Handle received ultrasound image data."""
        if not config.ultrasound or not config.ultrasound.enabled:
            return

        h = config.ultrasound.height
        w = config.ultrasound.width
        img = Image.fromarray(np.frombuffer(s.data, dtype=np.uint8).reshape(h, w)).convert("RGBA")
        img_array = np.array(img)
        np.divide(img_array, 255.0, out=self.ultrasound_image_data)
        dpg.set_value("ultrasound_image_data", self.ultrasound_image_data)

    def on_receive_franka_annotations(self, topic_name: str, s: FrankaInfo) -> None:
        """Handle received Franka robot state data."""
        self.current_joints_state_positions = s.joints_state_positions
        self.current_joints_state_velocities = s.joints_state_velocities
        dpg.set_value("joints_state_positions", [round(p, 4) for p in s.joints_state_positions])
        dpg.set_value("joints_state_velocities", [round(p, 4) for p in s.joints_state_velocities])

    def on_receive_target_annotations(self, topic_name: str, s: TargetInfo) -> None:
        """Handle received target position data."""
        self.current_target_position = s.position
        self.current_target_orientation = s.orientation
        dpg.set_value("target_position", [round(p, 4) for p in s.position])
        dpg.set_value("target_orientation", [round(p, 4) for p in s.orientation])

    def on_receive_ultrasound_annotations(self, topic_name: str, s: UltraSoundProbeInfo) -> None:
        """Handle received ultrasound probe position data."""
        dpg.set_value("ultrasound_position", [round(p, 4) for p in s.position])

    # Data publishing
    def publish_camera_annotations(self, commands: Dict[str, Any]) -> None:
        """Publish camera control commands to DDS."""
        c = config.room_camera
        writer = self.dds_writers.get(c.topic_ctrl.name, None)
        if c.topic_ctrl and writer is not None:
            o = CameraCtrlInput()
            if commands.get("focal_len") is not None:
                o.focal_len = commands["focal_len"]

            print(f"Publishing Data to {c.topic_ctrl} (Camera): {o}")
            writer.write(o)

    def publish_target_annotations(self, sender: Any, app_data: Any, user_data: Any) -> None:
        """Publish target position control commands."""
        field_name = "move_target"
        print(f"Publish Target Annotations (on checkbox): {dpg.get_value(field_name)}")

        if not dpg.get_value(field_name) or not self.current_target_position:
            return

        c = config.target
        writer = self.dds_writers.get(c.topic_ctrl.name, None)
        if not c.topic_ctrl or writer is None:
            return

        def run():
            while dpg.get_value(field_name):
                o = TargetCtrlInput()
                move_target_by = dpg.get_value("move_target_by")
                if move_target_by == "Random":
                    lower_bounds = np.array([-0.1, -0.1, -0.1])
                    upper_bounds = np.array([0.1, 0.1, 0.1])
                    o.position = rng.uniform(lower_bounds, upper_bounds) + self.current_target_position
                else:
                    t = [p for p in self.current_target_position]
                    t[1] = rng.uniform(t[1] - 0.1, t[1] + 0.1)
                    o.position = t
                o.orientation = self.current_target_orientation

                print(f"Publishing Data to Target: {o}")
                writer.write(o)
                time.sleep(3)  # c.topic_in.hz)
            print(f"Publishing Thread ({field_name}) done!")

        thread = threading.Thread(target=run)
        thread.start()
        print(f"Publishing Thread ({field_name}) started!")

    def publish_franka_annotations(self, sender: Any, app_data: Any, user_data: Any) -> None:
        """Publish Franka robot control commands."""
        field_name = "move_franka"
        print(f"Publish Franka Annotations (on checkbox): {dpg.get_value(field_name)}")

        if not dpg.get_value(field_name) or not self.current_joints_state_positions:
            return

        c = config.franka
        writer = self.dds_writers.get(c.topic_ctrl.name, None)
        if not c.topic_ctrl or writer is None:
            return

        def run():
            while dpg.get_value(field_name):
                writer = self.dds_writers[c.topic_ctrl.name]
                o = FrankaCtrlInput()
                o.target_position = self.current_target_position  # Frank movement is always based on local pos
                o.target_orientation = self.current_target_orientation

                print(f"Publishing Data to Target: {o}")
                writer.write(o)
                time.sleep(1)
            print(f"Publishing Thread ({field_name}) done!")

        thread = threading.Thread(target=run)
        thread.start()
        print(f"Publishing Thread ({field_name}) started!")

    @classmethod
    def run_app(cls) -> None:
        """Create and run the simulator application."""
        app = cls()
        app.create_app()
        app.connect_to_dds()
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


SimulatorApp.run_app()
