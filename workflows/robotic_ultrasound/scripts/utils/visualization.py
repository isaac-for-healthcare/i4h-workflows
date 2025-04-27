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

import argparse
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg
import numpy as np
import rti.connextdds as dds
from dds.schemas.camera_ctrl import CameraCtrlInput
from dds.schemas.camera_info import CameraInfo
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from dds.schemas.target_ctrl import TargetCtrlInput
from dds.schemas.target_info import TargetInfo
from dds.schemas.usp_data import UltraSoundProbeData
from dds.schemas.usp_info import UltraSoundProbeInfo
from dds.subscriber import SubscriberWithCallback
from PIL import Image
from simulation.configs.config import CameraConfig, Topic, UltraSoundConfig
from simulation.utils.common import colorize_depth, get_exp_config, list_exp_configs

parser = argparse.ArgumentParser(description="Visualize the robotic ultrasound simulation")
parser.add_argument(
    "-c",
    "--config",
    type=str,
    choices=list_exp_configs(),
    default="basic",
    help="Configuration name to use for visualization. Only `basic` is supported for now.",
)
config = get_exp_config(parser.parse_args().config)


rng = np.random.default_rng(config.random_seed)


class VisualizationApp:
    """A visualization application for robotic ultrasound simulation."""

    # Class-level constants
    DEFAULT_WINDOW_NAME: str = "Robotic Ultrasound Visualization"
    DEFAULT_RESIZEABLE: bool = True
    DEFAULT_CAMERA_RANGE_START: float = 20.0
    DEFAULT_CAMERA_RANGE_END: float = 200.0
    CAMERA_MODES: List[str] = ["RGB", "DEPTH"]
    TARGET_MOVE_MODES: List[str] = ["Random", "Y-Axis"]
    CAMERA_ZOOM_STEP: float = 0.2
    MOUSE_WHEEL_STEP: float = 5.0
    WINDOW_WIDTH_PADDING: int = 80
    WINDOW_HEIGHT_PADDING: int = 300
    TEXT_WIDTH: int = 530
    BORDER_COLORS = {
        "room_camera": [255, 0, 0, 255],
        "wrist_camera": [0, 255, 0, 255],
        "ultrasound": [0, 0, 255, 255],
        "ultrasound_gan": [255, 255, 0, 255], # Yellow for GAN
    }

    def __init__(self) -> None:
        """Initialize the VisualizationApp with default settings."""
        self.window_name: str = self.DEFAULT_WINDOW_NAME

        # Calculate width based on enabled components
        width_components = []
        if config.room_camera and config.room_camera.enabled:
            width_components.append(config.room_camera.width)
        if config.wrist_camera and config.wrist_camera.enabled:
            width_components.append(config.wrist_camera.width)
        if config.ultrasound and config.ultrasound.enabled:
            width_components.append(config.ultrasound.width)
        # Add width for GAN window if enabled
        if hasattr(config, 'ultrasound_gan') and config.ultrasound_gan and config.ultrasound_gan.enabled:
            width_components.append(config.ultrasound_gan.width)

        self.window_width: int = sum(width_components) + self.WINDOW_WIDTH_PADDING

        # Calculate height based on enabled components
        height_components = []
        if config.room_camera and config.room_camera.enabled:
             height_components.append(config.room_camera.height)
        if config.wrist_camera and config.wrist_camera.enabled:
             height_components.append(config.wrist_camera.height)
        # Consider ultrasound heights only if they exist and are enabled
        if config.ultrasound and config.ultrasound.enabled:
             height_components.append(config.ultrasound.height)
        if hasattr(config, 'ultrasound_gan') and config.ultrasound_gan and config.ultrasound_gan.enabled:
             height_components.append(config.ultrasound_gan.height)

        # Ensure height_components is not empty before calling max
        max_image_height = max(height_components) if height_components else 0
        self.window_height: int = max_image_height + self.WINDOW_HEIGHT_PADDING

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
        if config.room_camera and config.room_camera.enabled:
            self.room_camera_image_data: np.ndarray = np.zeros(
                (config.room_camera.height, config.room_camera.width, 3), dtype=np.float32
            )
        if config.wrist_camera and config.wrist_camera.enabled:
            self.wrist_camera_image_data: np.ndarray = np.zeros(
                (config.wrist_camera.height, config.wrist_camera.width, 3), dtype=np.float32
            )
        if config.ultrasound and config.ultrasound.enabled:
            self.ultrasound_image_data: np.ndarray = np.zeros(
                (config.ultrasound.height, config.ultrasound.width, 3), dtype=np.float32
            )
        # Add GAN image data if enabled in config
        if hasattr(config, 'ultrasound_gan') and config.ultrasound_gan and config.ultrasound_gan.enabled:
            self.ultrasound_gan_image_data: np.ndarray = np.zeros(
                (config.ultrasound_gan.height, config.ultrasound_gan.width, 3), dtype=np.float32
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
        # Add subscriber for GAN image
        self.sub_ultrasound_gan_image: Optional[SubscriberWithCallback] = None


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
            if config.room_camera and config.room_camera.enabled:
                self.add_texture_widget(config.room_camera, self.room_camera_image_data, "room_camera_image_data")
            if config.wrist_camera and config.wrist_camera.enabled:
                self.add_texture_widget(config.wrist_camera, self.wrist_camera_image_data, "wrist_camera_image_data")
            if config.ultrasound and config.ultrasound.enabled:
                self.add_texture_widget(config.ultrasound, self.ultrasound_image_data, "ultrasound_image_data")
            # Add texture for GAN ultrasound if enabled
            if hasattr(config, 'ultrasound_gan') and config.ultrasound_gan and config.ultrasound_gan.enabled:
                self.add_texture_widget(config.ultrasound_gan, self.ultrasound_gan_image_data, "ultrasound_gan_image_data")

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

                # Ultrasound GAN - Widgets
                if hasattr(config, 'ultrasound_gan') and config.ultrasound_gan and config.ultrasound_gan.enabled:
                    self.add_ultrasound_gan_widget()

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
    def add_texture_widget(self, cfg: CameraConfig | UltraSoundConfig, value, tag, format=dpg.mvFormat_Float_rgb) -> None:
        """Create a texture widget for a camera or image."""
        # Check if config object has width and height before accessing
        if hasattr(cfg, 'width') and hasattr(cfg, 'height'):
             dpg.add_raw_texture(width=cfg.width, height=cfg.height, default_value=value, tag=tag, format=format)
        else:
             print(f"Warning: Config object for tag '{tag}' lacks width/height attributes. Skipping texture.")

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

    def add_ultrasound_gan_widget(self) -> None:
        """Add ultrasound with GAN widget to the GUI."""
        with dpg.group(horizontal=False):
            dpg.add_checkbox(
                label=" Ultrasound GAN",
                tag="streaming_ultrasound_gan",
                callback=self.on_streaming_ultrasound_gan,
                default_value=True,
            )
            dpg.add_image("ultrasound_gan_image_data", border_color=self.BORDER_COLORS["ultrasound_gan"])
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Probe Position (world):")
                dpg.add_input_text(
                    tag="ultrasound_gan_position",
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
        if config.room_camera and config.room_camera.enabled:
            self.on_streaming_room_camera()
        if config.wrist_camera and config.wrist_camera.enabled:
            self.on_streaming_wrist_camera()
        if config.ultrasound and config.ultrasound.enabled:
            self.on_streaming_ultrasound()
        # Connect GAN ultrasound stream if enabled
        if hasattr(config, 'ultrasound_gan') and config.ultrasound_gan and config.ultrasound_gan.enabled:
            self.on_streaming_ultrasound_gan()

        # Franka
        if config.franka and config.franka.enabled:
            if config.franka.topic_ctrl:
                # NOTE: the Franka topic_ctrl is not used for anything. This is reserved for future use.
                self.connect_to_dds_publisher(config.franka.topic_ctrl, FrankaCtrlInput)
            if config.franka.topic_info:
                self.connect_to_dds_subscriber(config.franka.topic_info, FrankaInfo, self.on_receive_franka_annotations)

        # Target
        if config.target and config.target.enabled:
            if config.target.topic_ctrl:
                # NOTE: the Target topic_ctrl is not used for anything. This is reserved for future use.
                self.connect_to_dds_publisher(config.target.topic_ctrl, TargetCtrlInput)
            if config.target.topic_info:
                # NOTE: the Target topic_info is not used for anything. This is reserved for future use.
                self.connect_to_dds_subscriber(config.target.topic_info, TargetInfo, self.on_receive_target_annotations)

        # Ultrasound
        if config.ultrasound and config.ultrasound.enabled:
            # Getting the probe position
            if config.ultrasound.topic_info:
                self.connect_to_dds_subscriber(
                    config.ultrasound.topic_info,
                    UltraSoundProbeInfo,
                    self.on_receive_ultrasound_annotations,
                )

        # Ultrasound Pose (remains the same)
        # Note: Assuming both simulators use the same pose topic defined in config.ultrasound.topic_info
        if config.ultrasound and config.ultrasound.enabled and config.ultrasound.topic_info:
             # Check if already subscribed by GAN section if GAN uses the same topic
             connect_std_us_pose = True
             if (hasattr(config, 'ultrasound_gan') and config.ultrasound_gan and config.ultrasound_gan.enabled and
                 config.ultrasound_gan.topic_info and config.ultrasound.topic_info.name == config.ultrasound_gan.topic_info.name):
                 print(f"Info: Standard ultrasound ('{config.ultrasound.topic_info.name}') pose topic is the same as GAN pose topic. Using single subscriber.")
                 connect_std_us_pose = False # Avoid duplicate subscription

             if connect_std_us_pose:
                 print(f"Subscribing to standard ultrasound pose: {config.ultrasound.topic_info.name}")
                 self.connect_to_dds_subscriber(
                     config.ultrasound.topic_info,
                     UltraSoundProbeInfo,
                     self.on_receive_ultrasound_annotations, # Callback updates standard US position tag
                 )

        # Ultrasound with GAN Pose
        # Only subscribe if GAN is enabled, has a topic_info, AND that topic is different from standard US pose topic
        if (hasattr(config, 'ultrasound_gan') and config.ultrasound_gan and config.ultrasound_gan.enabled and
            config.ultrasound_gan.topic_info):
            # Check if the topic name is different from the standard ultrasound info topic
            is_different_topic = True
            if (config.ultrasound and config.ultrasound.enabled and config.ultrasound.topic_info and
                config.ultrasound_gan.topic_info.name == config.ultrasound.topic_info.name):
                is_different_topic = False # Already handled above or will be handled by std US sub

            if is_different_topic:
                 print(f"Subscribing to GAN ultrasound pose: {config.ultrasound_gan.topic_info.name}")
                 self.connect_to_dds_subscriber(
                     config.ultrasound_gan.topic_info,
                     UltraSoundProbeInfo,
                     self.on_receive_ultrasound_gan_annotations, # Callback updates GAN US position tag
                 )
            else:
                 # If using the same topic as standard US, ensure the standard callback updates BOTH position tags
                 # Modification needed in on_receive_ultrasound_annotations
                 pass

    def connect_to_dds_publisher(self, topic: Topic, cls) -> None:
        """Connect to a DDS publisher."""
        if self.dds_writers.get(topic.name, None) is None:
            print(f"\nPublishing to topic: {topic}")
            p = dds.DomainParticipant(domain_id=topic.domain_id)
            writer = dds.DataWriter(dds.Topic(p, topic.name, cls))
            self.dds_writers[topic.name] = writer

    def connect_to_dds_subscriber(self, topic: Topic, cls, cb) -> SubscriberWithCallback:
        """Connect to a DDS subscriber."""
        # Add check for valid topic name
        if topic and topic.name:
            print(f"\nSubscribing to topic: {topic}")
            s = SubscriberWithCallback(cb=cb, domain_id=topic.domain_id, topic=topic.name, cls=cls, period=topic.period)
            s.start()
            return s
        else:
            print(f"Warning: Invalid or missing topic name provided for subscriber. Skipping.")
            return None

    # Streaming control
    def on_streaming_room_camera(self) -> None:
        """Handle room camera streaming control."""
        if self.sub_room_camera_rgb is not None:
            self.sub_room_camera_rgb.stop()
        if self.sub_room_camera_depth is not None:
            self.sub_room_camera_depth.stop()

        if dpg.get_value("streaming_room_camera"):
            # NOTE: the ROOM Camera topic_ctrl is not used for anything. This is reserved for future use.
            self.connect_to_dds_publisher(config.room_camera.topic_ctrl, CameraCtrlInput)

        if dpg.get_value("room_camera_mode") == "RGB":
            # Switch to RGB mode and get the RGB image frame
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
            # Switch to DEPTH mode and get the DEPTH image frame
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
            # Switch to RGB mode and get the RGB image frame
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
            # Switch to DEPTH mode and get the DEPTH image frame
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
        """Handle ultrasound streaming images and control."""
        if self.sub_ultrasound_image is not None:
            self.sub_ultrasound_image.stop()

        # Get the ultrasound image
        self.sub_ultrasound_image = self.on_streaming_xyz(
            tag="streaming_ultrasound",
            sub=self.sub_ultrasound_image,
            topic=config.ultrasound.topic_data,
            cls=UltraSoundProbeData,
            cb=self.on_receive_ultrasound_image,
            dv_tag="ultrasound_image_data",
            dv_val=self.ultrasound_image_data,
        )

    def on_streaming_ultrasound_gan(self) -> None:
        """Handle ultrasound with GAN streaming control."""
        # Check existence before stopping
        if hasattr(self, 'sub_ultrasound_gan_image') and self.sub_ultrasound_gan_image is not None:
             self.sub_ultrasound_gan_image.stop()
             self.sub_ultrasound_gan_image = None # Clear it

        if not dpg.get_value("streaming_ultrasound_gan"): return # Added check

        # Get the ultrasound with GAN image
        # Ensure config.ultrasound_gan and its topic_data exist
        if hasattr(config, 'ultrasound_gan') and config.ultrasound_gan and config.ultrasound_gan.topic_data:
            self.sub_ultrasound_gan_image = self.on_streaming_xyz(
                tag="streaming_ultrasound_gan",       # Use new tag
                sub=self.sub_ultrasound_gan_image,  # Use new subscriber variable
                topic=config.ultrasound_gan.topic_data, # Use GAN data topic from config
                cls=UltraSoundProbeData,
                cb=self.on_receive_ultrasound_gan_image, # Use new callback
                dv_tag="ultrasound_gan_image_data",     # Use new texture tag
                dv_val=self.ultrasound_gan_image_data,  # Use new data variable
            )
        else:
            print("Warning: Ultrasound GAN config or topic_data missing. Cannot start stream.")

    def on_streaming_xyz(self, tag, sub, topic, cls, cb, dv_tag, dv_val) -> Optional[SubscriberWithCallback]:
        """Generic streaming control helper."""
        # Add check for valid topic
        if not topic or not topic.name:
             print(f"Warning: Invalid topic provided for tag '{tag}'. Cannot control stream.")
             if sub: sub.stop() # Stop if running
             return None

        if not dpg.get_value(tag):
            if sub:
                sub.stop()
                # Check if dv_val exists before modifying
                if dv_val is not None:
                    np.multiply(dv_val, 0.0, out=dv_val)
                    dpg.set_value(dv_tag, dv_val)
                print(f"{tag}: {topic.name} => Stopped!")
            return None # Return None as subscriber is stopped or wasn't started

        # Only proceed if tag is checked (True)
        # print(f"Attempting to start stream for {tag} on {topic.name}")
        if not sub:
            sub = self.connect_to_dds_subscriber(topic, cls, cb)

        # Check if subscriber was successfully created before starting
        if sub:
            sub.start() # Call start directly. Assume it handles being called if already running.
            print(f"{tag}: {topic.name} => Started!")
        else:
             print(f"Error: Failed to create subscriber for {topic.name}. Cannot start stream.")

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
            img_array = np.frombuffer(s.data, dtype=np.uint8).reshape(h, w, 3)

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
        try:
            # Ensure data length matches expected size
            expected_bytes = h * w * 3 # Assuming uint8 RGB
            if len(s.data) != expected_bytes:
                print(f"Warning: Received ultrasound data size mismatch for {topic_name}. Expected {expected_bytes}, got {len(s.data)}. Skipping frame.")
                return
            img_array = np.frombuffer(s.data, dtype=np.uint8).reshape(h, w, 3)
            # Optional: Convert with PIL if needed, but direct reshape is fine for uint8 RGB
            # img = Image.fromarray(img_array).convert("RGB")
            # img_array = np.array(img)
            np.divide(img_array, 255.0, out=self.ultrasound_image_data, dtype=np.float32)
            dpg.set_value("ultrasound_image_data", self.ultrasound_image_data)
        except Exception as e:
            print(f"Error processing ultrasound image from {topic_name}: {e}")
            print(traceback.format_exc())

    def on_receive_franka_annotations(self, topic_name: str, s: FrankaInfo) -> None:
        """Handle received Franka robot state data."""
        self.current_joints_state_positions = s.joints_state_positions
        # self.current_joints_state_velocities = s.joints_state_velocities
        dpg.set_value("joints_state_positions", [round(p, 4) for p in s.joints_state_positions])
        # dpg.set_value("joints_state_velocities", [round(p, 4) for p in s.joints_state_velocities])

    def on_receive_target_annotations(self, topic_name: str, s: TargetInfo) -> None:
        """Handle received target position data."""
        self.current_target_position = s.position
        self.current_target_orientation = s.orientation
        dpg.set_value("target_position", [round(p, 4) for p in s.position])
        dpg.set_value("target_orientation", [round(p, 4) for p in s.orientation])

    def on_receive_ultrasound_annotations(self, topic_name: str, s: UltraSoundProbeInfo) -> None:
        """Handle received ultrasound probe position data."""
        # Update standard ultrasound position
        formatted_pos = [round(p, 4) for p in s.position]
        dpg.set_value("ultrasound_position", formatted_pos)

        # If GAN uses the same topic, update its position too
        if (hasattr(config, 'ultrasound_gan') and config.ultrasound_gan and config.ultrasound_gan.enabled and
            config.ultrasound_gan.topic_info and config.ultrasound.topic_info and
            config.ultrasound_gan.topic_info.name == config.ultrasound.topic_info.name):
            dpg.set_value("ultrasound_gan_position", formatted_pos)

    # --- New Callback for GAN Image Data ---
    def on_receive_ultrasound_gan_image(self, topic_name: str, s: UltraSoundProbeData) -> None:
        """Handle received ultrasound image data with GAN processing."""
        if not hasattr(config, 'ultrasound_gan') or not config.ultrasound_gan.enabled:
            return
        # Check if data array exists
        if not hasattr(self, 'ultrasound_gan_image_data'):
            print("Warning: ultrasound_gan_image_data not initialized. Skipping update.")
            return

        h = config.ultrasound_gan.height
        w = config.ultrasound_gan.width
        try:
             # Ensure data length matches expected size
            expected_bytes = h * w * 3 # Assuming uint8 RGB output from GAN sim
            if len(s.data) != expected_bytes:
                print(f"Warning: Received ultrasound GAN data size mismatch for {topic_name}. Expected {expected_bytes}, got {len(s.data)}. Skipping frame.")
                return

            img_array = np.frombuffer(s.data, dtype=np.uint8).reshape(h, w, 3)
            # Convert to float32 for DPG texture
            np.divide(img_array, 255.0, out=self.ultrasound_gan_image_data, dtype=np.float32)
            # Update the specific DPG texture value
            dpg.set_value("ultrasound_gan_image_data", self.ultrasound_gan_image_data)
        except Exception as e:
            print(f"Error processing ultrasound GAN image from {topic_name}: {e}")
            print(traceback.format_exc())
    # --- End New Callback ---

    def on_receive_ultrasound_gan_annotations(self, topic_name: str, s: UltraSoundProbeInfo) -> None:
        """Handle received ultrasound with GAN probe position data."""
        formatted_pos = [round(p, 4) for p in s.position]
        dpg.set_value("ultrasound_gan_position", formatted_pos)

    # Data publishing
    def publish_camera_annotations(self, commands: Dict[str, Any]) -> None:
        """Publish camera control commands to DDS."""
        c = config.room_camera
        # Ensure component and topic exist before proceeding
        if not c or not c.enabled or not c.topic_ctrl: return

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
                # Ensure writer still exists (might be cleaned up?)
                current_writer = self.dds_writers.get(c.topic_ctrl.name, None)
                if current_writer is None:
                    print(f"Publishing Thread ({field_name}): Writer lost. Stopping.")
                    break
                # Ensure position/orientation are still valid
                if self.current_target_position is None or self.current_target_orientation is None:
                     print(f"Publishing Thread ({field_name}): Missing target pose. Skipping write.")
                     time.sleep(0.5) # Wait before retrying
                     continue

                o = TargetCtrlInput()
                move_target_by = dpg.get_value("move_target_by")
                current_pos_np = np.array(self.current_target_position) # Make copy for manipulation

                if move_target_by == "Random":
                    lower_bounds = np.array([-0.1, -0.1, -0.1])
                    upper_bounds = np.array([0.1, 0.1, 0.1])
                    o.position = rng.uniform(lower_bounds, upper_bounds) + current_pos_np
                else:
                    t = [p for p in current_pos_np]
                    t[1] = rng.uniform(t[1] - 0.1, t[1] + 0.1)
                    o.position = t
                o.orientation = self.current_target_orientation

                print(f"Publishing Data to Target: {o}")
                current_writer.write(o)
            print(f"Publishing Thread ({field_name}) done!")

        thread = threading.Thread(target=run, daemon=True) # Set daemon=True
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
                 # Ensure writer still exists
                current_writer = self.dds_writers.get(c.topic_ctrl.name, None)
                if current_writer is None:
                    print(f"Publishing Thread ({field_name}): Writer lost. Stopping.")
                    break
                # Re-check target pose validity inside loop
                if self.current_target_position is None or self.current_target_orientation is None:
                    print(f"Publishing Thread ({field_name}): Missing target pose. Skipping Franka write.")
                    # Uncheck the box to stop the loop gracefully
                    dpg.set_value(field_name, False)
                    break

                o = FrankaCtrlInput()
                o.target_position = self.current_target_position  # Frank movement is always based on local pos
                o.target_orientation = self.current_target_orientation

                print(f"Publishing Data to Target: {o}")
                current_writer.write(o)
            print(f"Publishing Thread ({field_name}) done!")

        thread = threading.Thread(target=run, daemon=True) # Set daemon=True
        thread.start()
        print(f"Publishing Thread ({field_name}) started!")

    @classmethod
    def run_app(cls) -> None:
        """Create and run the simulator application."""
        app = cls()
        try:
            app.create_app()
            app.connect_to_dds() # Connect DDS after UI is created
            while dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
        except Exception as e:
             print(f"Error during VisualizationApp execution: {e}")
             print(traceback.format_exc())
        finally:
            # Cleanup DPG context
            if dpg.is_dearpygui_running():
                 dpg.destroy_context()
            # Add cleanup for DDS subscribers if needed (SubscriberWithCallback might handle this)
            print("VisualizationApp finished.")


if __name__ == "__main__":
    VisualizationApp.run_app()
