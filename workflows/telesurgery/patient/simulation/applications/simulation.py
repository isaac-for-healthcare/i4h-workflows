# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import time
from typing import Callable

import numpy as np
from applications.controller import HIDController
from isaacsim import SimulationApp


class Simulation:
    """A simulation environment for IsaacSim that manages a 3D scene with dynamic objects and camera.

    This class sets up a simulation environment with:
    - A dynamic scene containing cubes with different properties
    - A camera for capturing images
    - A dynamic texture system for real-time image updates
    - Physics simulation capabilities

    Args:
        headless (bool): Whether to run the simulation in headless mode (without GUI)
        image_size (tuple[int, int, int]): The dimensions of the camera image (width, height, channels)
        camera_frequency (int): The frequency of the camera
    """

    def __init__(
        self,
        headless: bool,
        image_size: tuple[int, int, int],
        camera_frequency: int,
    ):
        self._headless = headless
        self._image_size = image_size
        self._camera_frequency = camera_frequency
        self._controller = HIDController()
        self._simulation_app = SimulationApp({"headless": self._headless})

        # Any Omniverse level imports must occur after the SimulationApp class is instantiated
        import carb
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import SurfaceGripper
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self._simulation_app.close()
            sys.exit()

        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()

        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"
        add_reference_to_stage(usd_path=gripper_usd, prim_path="/World/UR10/ee_link")

        gripper = SurfaceGripper(
            end_effector_prim_path="/World/UR10/ee_link",
            translate=0.1611,
            direction="x",
        )
        self._ur10 = self._world.scene.add(
            SingleManipulator(
                prim_path="/World/UR10",
                name="my_ur10",
                end_effector_prim_path="/World/UR10/ee_link",
                gripper=gripper,
            )
        )

        self._ur10.set_joints_default_state(positions=self._controller.default_joint_positions)
        self._world.scene.add(
            DynamicCuboid(
                name="cube",
                position=np.array([0.3, 0.3, 0.3]),
                prim_path="/World/Cube",
                scale=np.array([0.0515, 0.0515, 0.0515]),
                size=1.0,
                color=np.array([0, 0, 1]),
            )
        )
        self._world.scene.add_default_ground_plane()
        self._ur10.gripper.set_default_state(opened=True)

        # Set the initial camera view to look at the robot
        set_camera_view(eye=[1.5, 1.5, 1.0], target=[0, 0, 0.5])

        self._world.reset()

        class CameraWithImageCallback(Camera):
            def __init__(self, controller: HIDController, image_size: tuple[int, int, int], *args, **kwargs):
                self._image_callback = None
                self._rendering_frame = -1
                self._controller = controller
                self._image_size = image_size
                self._frame_num = 1
                super().__init__(*args, **kwargs)

            def set_image_callback(self, image_callback: Callable):
                self._image_callback = image_callback

            def _data_acquisition_callback(self, event: carb.events.IEvent):
                super()._data_acquisition_callback(event)

                if self._image_callback is None:
                    return

                # if there is a new frame, push the image
                if self._current_frame["rendering_frame"] != self._rendering_frame:
                    self._rendering_frame = self._current_frame["rendering_frame"]
                    image = self._current_frame["rgba"]
                    time_now = time.time_ns()

                    # TODO: the first frame has a size of 0, remove this code if that is fixed
                    if not image.shape[0] == 0:
                        self._image_callback(
                            {
                                "image": image,
                                "joint_names": self._controller.joint_names,
                                "joint_positions": self._controller.target_joint_positions,
                                "size": self._image_size,
                                "frame_num": self._frame_num,
                                "last_hid_event": self._controller.take_last_hid_event(),
                                "video_acquisition_timestamp": time_now,
                            }
                        )
                        self._frame_num += 1

        # Use the camera from the end effector
        self._camera = CameraWithImageCallback(
            controller=self._controller,
            image_size=self._image_size,
            prim_path="/World/UR10/ee_link/Camera",
            frequency=self._camera_frequency,
            resolution=(self._image_size[1], self._image_size[0]),
        )

        # Resetting the world needs to be called before querying anything related to an articulation specifically.
        # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
        self._world.reset()

        self._world.scene.add_default_ground_plane()

        self._camera.initialize()

    def stop(self):
        """Signals the controller to stop its background threads."""
        self._controller.stop()

    def __del__(self):
        self._simulation_app.close()

    def get_image_size(self):
        """Get the dimensions of the camera image.

        Returns:
            tuple[int, int, int]: The image dimensions (width, height, channels)
        """
        return self._image_size

    def data_ready_callback(self, data):
        """Update the dynamic texture with new image data.

        Args:
            data: The image data to be displayed on the dynamic texture
        """
        self._dynamic_texture.set_bytes_data_from_gpu(
            data, [self._image_size[0], self._image_size[1], self._image_size[2]]
        )

    def hid_event_callback(self, event):
        """Callback for HID events.

        Args:
            event: The HID event to be processed
        """
        self._controller.handle_hid_event(event)

    def run(self, push_data_callback: Callable):
        """Run the simulation loop.

        This method starts the simulation and continuously:
        - Steps the physics simulation
        - Renders the scene
        - Captures and processes camera images
        - Updates the dynamic texture

        Args:
            push_data_callback (Callable): A callback function that receives camera image data

        Raises:
            ValueError: If the image size has an invalid number of channels (must be 4 for RGBA)
        """
        if not self._image_size[2] == 4:
            raise ValueError(f"Invalid image components count: {self._image_size[2]}")

        self._camera.set_image_callback(push_data_callback)

        while self._simulation_app.is_running():
            self._world.step(render=True)

            # Note: You might need SimulationContext if self._world doesn't have get_physics_dt()
            # sim_context = self._simulation_app.get_simulation_context()
            # dt = sim_context.get_physics_dt()
            dt = self._world.get_physics_dt()

            self._world.step(render=True)
            next_move = self._controller.forward(dt)
            self._ur10.set_joint_positions(next_move)

            # Update the robot joint position based on the joystick values
