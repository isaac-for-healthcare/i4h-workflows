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
import json
import os
from dataclasses import dataclass

# import numpy as np
import numpy as np
import rti.connextdds as dds
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, MetadataPolicy, Operator, OperatorSpec
from rti.types import struct
from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets

try:
    import raysim.cuda as rs
except Exception as e:
    raise ImportError(f"Failed to initialize ray_sim_python: {e}\n" "Please check the installation and dependencies.")

# Rest of your imports
from dds.schemas.usp_data import UltraSoundProbeData
from dds.schemas.usp_info import UltraSoundProbeInfo
from dds.subscriber import SubscriberWithQueue


@dataclass
class Pose:
    """3D pose with position and orientation"""

    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # as quaternians from isaac_sim


class UltrasoundSimSubscriber(Operator):
    """
    Subscribes to RTI DDS topics and forwards the received data to the next operator in the pipeline.

    This operator creates a DDS subscriber that listens to a specified topic and emits the received
    data through its output port. If no data is received, it emits a tuple with None and False to
    indicate the absence of data.

    Attributes:
        domain_id: The DDS domain ID to use for communication.
        topic: The name of the DDS topic to subscribe to.
        data_schema: The data schema/type definition for the messages on the topic.
        period: The period at which to check for new data (in seconds).
    """

    def __init__(
        self,
        fragment,
        *args,
        domain_id=0,
        topic="topic_ultrasound_info",
        data_schema: struct = UltraSoundProbeInfo,
        **kwargs,
    ):
        self.domain_id = domain_id
        self.topic = topic
        self.data_schema = data_schema
        self.reader = None
        self.message = None
        self.dp = None
        self.subscriber = None
        self.period = 1 / 30.0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        """
        Sets up the operator's configuration.

        This method is called when the operator is created. It initializes the subscriber
        and starts it to begin listening for incoming data.
        """
        spec.output("output")
        self.subscriber = SubscriberWithQueue(
            domain_id=self.domain_id,
            topic=self.topic,
            cls=self.data_schema,
            period=self.period,
        )
        self.subscriber.start()

    def compute(self, op_input, op_output, context):
        """
        Computes the operator's logic.

        This method is called when the operator is executed. It reads data from the subscriber
        and emits it through the output port. If no data is available, it emits a tuple with None
        and False to indicate the absence of data.

        Args:
            op_input: The input port of the operator.
            op_output: The output port of the operator.
            context: The context of the operator.
        """
        data = self.subscriber.read_data()
        if data is not None:
            op_output.emit((data, True), "output")
        else:
            op_output.emit((None, False), "output")


class Simulator(Operator):
    """
    Ultrasound simulator with careful initialization

    Args:
        fragment: The fragment of the operator.
        out_height: The height of the output image.
        out_width: The width of the output image.
        start_pose: The initial pose of the probe, default is [0, 0, 0, 0, 0, 0].
        config_path: Path to a JSON configuration file with probe parameters.
    """

    def __init__(
        self,
        fragment,
        *args,
        out_height=500,
        out_width=500,
        start_pose=np.zeros((6,)),
        config_path=None,
        **kwargs,
    ):
        # Initialize all critical variables first
        self.out_height = out_height
        self.out_width = out_width
        self.start_pose = start_pose
        self.config_path = config_path
        self.simulator = None
        self.sim_params = None
        self.probe = None
        self.world = None
        self.materials = None
        self.meshes = []  # Store mesh references
        self.last_time_probe_info = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """
        Sets up the operator's configuration.

        This method is called when the operator is created. It initializes the materials
        and world, and adds the meshes to the world.
        """
        spec.output("output")
        spec.input("input")

        # Create materials
        self.materials = rs.Materials()

        # Create world
        self.world = rs.World("water")

        # Add meshes
        self.meshes = []  # Clear and rebuild mesh list

        # Helper function to safely add a mesh
        def add_mesh(filename, material_name):
            """
            Adds a mesh to the world.

            Args:
                filename: The name of the mesh file to add.
                material_name: The name of the material to use for the mesh.
            """
            # Construct the full path to the mesh file
            mesh_dir = robot_us_assets.organs
            mesh_path = os.path.join(mesh_dir, filename)
            try:
                material_idx = self.materials.get_index(material_name)
                mesh = rs.Mesh(mesh_path, material_idx)
                self.world.add(mesh)
                self.meshes.append(mesh)  # Keep reference
                return True
            except Exception as e:
                print(f"Error adding mesh {mesh_path}: {e}")
                return False

        # Add meshes one by one
        mesh_configs = [
            ("Tumor1.obj", "fat"),
            ("Tumor2.obj", "water"),
            ("Liver.obj", "liver"),
            ("Skin.obj", "fat"),
            ("Bone.obj", "bone"),
            ("Vessels.obj", "water"),
            ("Gallbladder.obj", "water"),
            ("Spleen.obj", "liver"),
            # ("Heart.obj", "liver"),
            ("Stomach.obj", "water"),
            ("Pancreas.obj", "liver"),
            ("Small_bowel.obj", "water"),
            ("Colon.obj", "water"),
        ]

        # Count successful mesh additions
        success_count = 0
        for filename, material in mesh_configs:
            if add_mesh(filename, material):
                success_count += 1

        if success_count == 0:
            print("WARNING: No meshes were successfully added to the world!")

        # Create probe position
        position = np.array(self.start_pose[:3], dtype=np.float32)
        rotation = np.array(self.start_pose[3:], dtype=np.float32)

        if hasattr(position, "get"):
            position = position.get()
            rotation = rotation.get()

        initial_pose = rs.Pose(position, rotation)

        # Load probe configuration from JSON file if provided
        default_probe_params = {
            "num_elements": 4096,
            "opening_angle": 73.0,
            "radius": 45.0,
            "frequency": 2.5,
            "elevational_height": 7.0,
            "num_el_samples": 10,
        }

        default_sim_params = {
            "conv_psf": True,
            "buffer_size": 4096,
            "t_far": 180.0,
        }

        probe_params = default_probe_params
        sim_config = default_sim_params

        if self.config_path:
            try:
                with open(self.config_path, "r") as f:
                    config_data = json.load(f)

                    # Load probe parameters
                    if "probe_params" in config_data:
                        # Use values from JSON, falling back to defaults for any missing parameters
                        for key in default_probe_params:
                            if key in config_data["probe_params"]:
                                probe_params[key] = config_data["probe_params"][key]
                        print(f"Loaded probe configuration from {self.config_path}")
                    else:
                        print(f"Warning: No 'probe_params' section found in {self.config_path}. Using defaults.")

                    # Load simulation parameters
                    if "sim_params" in config_data:
                        # Use values from JSON, falling back to defaults for any missing parameters
                        for key in default_sim_params:
                            if key in config_data["sim_params"]:
                                sim_config[key] = config_data["sim_params"][key]
                        print(f"Loaded simulation parameters from {self.config_path}")
                    else:
                        print(f"Warning: No 'sim_params' section found in {self.config_path}. Using defaults.")
            except Exception as e:
                print(f"Error loading configuration from {self.config_path}: {e}")
                print("Using default parameters")
        else:
            print("No config file specified. Using default parameters.")

        # Create ultrasound probe with configured parameters
        self.probe = rs.UltrasoundProbe(
            initial_pose,
            num_elements=probe_params["num_elements"],
            opening_angle=probe_params["opening_angle"],
            radius=probe_params["radius"],
            frequency=probe_params["frequency"],
            elevational_height=probe_params["elevational_height"],
            num_el_samples=probe_params["num_el_samples"],
        )

        # Create simulator
        try:
            self.simulator = rs.RaytracingUltrasoundSimulator(self.world, self.materials)
        except Exception as e:
            print(f"ERROR creating simulator: {e}")
            raise RuntimeError(f"Failed to create simulator: {e}")

        # Create simulation parameters with configured values
        self.sim_params = rs.SimParams()
        self.sim_params.conv_psf = sim_config["conv_psf"]
        self.sim_params.buffer_size = sim_config["buffer_size"]
        self.sim_params.t_far = sim_config["t_far"]
        self.sim_params.b_mode_size = (self.out_height, self.out_width)
        self.sim_params.enable_cuda_timing = True

    def compute(self, op_input, op_output, context):
        """
        Computes the operator's logic.

        This method is called when the operator is executed. It receives probe information
        and processes the probe pose, then runs the simulation and processes the ultrasound image.

        Args:
            op_input: The input port of the operator. In this case, it is the probe information.
            op_output: The output port of the operator. In this case, it is the ultrasound image.
            context: The context of the operator.
        """
        probe_info, receiving = op_input.receive("input")

        # Process the probe pose
        self._process_probe_pose(probe_info, receiving)

        # Run simulation
        b_mode_image = self.simulator.simulate(self.probe, self.sim_params)

        # Process the image
        rgb_data = self._process_ultrasound_image(b_mode_image)
        op_output.emit({"": rgb_data}, "output")

    def _process_probe_pose(self, probe_info, receiving=True):
        """
        Process the probe pose data and update the probe position.

        Args:
            probe_info: The probe information containing position and orientation
            receiving: Boolean indicating if probe info is being received
        """
        # Mock pose if no probe info is received
        if receiving:
            translation = probe_info.position
            rot_euler = probe_info.orientation
            self.last_time_probe_info = probe_info
        else:
            if self.last_time_probe_info is None:
                translation = self.start_pose[:3]
                rot_euler = self.start_pose[3:]
            else:
                translation = self.last_time_probe_info.position
                rot_euler = self.last_time_probe_info.orientation

        # Convert from lists or cupy arrays to numpy arrays
        translation_array = np.array(translation, dtype=np.float32)
        rotation_array = np.array(rot_euler, dtype=np.float32)

        # Create new pose and update probe
        new_pose = rs.Pose(translation_array, rotation_array)
        self.probe.set_pose(new_pose)

    def _process_ultrasound_image(self, b_mode_image):
        """
        Process the ultrasound image for display.

        Args:
            b_mode_image: The B-mode ultrasound image

        Returns:
            RGB image data ready for display
        """
        # Process the image
        min_val, max_val = -60.0, 0.0
        normalized_image = np.clip((b_mode_image - min_val) / (max_val - min_val), 0, 1)
        if np.min(normalized_image) == 1.0:
            print("WARNING: empty frame")
            img_uint8 = np.zeros((self.out_height, self.out_width), dtype=np.uint8)
        else:
            img_uint8 = (normalized_image * 255).astype(np.uint8)

        # Create RGB output
        rgb_data = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

        return rgb_data


class UltrasoundSimPublisher(Operator):
    """
    Transmit incoming stream over RTI-Topic

    Args:
        fragment: The fragment of the operator.
        domain_id: The DDS domain ID to use for communication.
        topic: The name of the DDS topic to publish to.
        data_schema: The data schema/type definition for the messages on the topic.
    """

    def __init__(
        self, fragment, *args, domain_id=0, topic="output_topic", data_schema: struct = UltraSoundProbeData, **kwargs
    ):
        self.domain_id = domain_id
        self.topic = topic
        self.data_schema = data_schema
        self.writer = None
        self.message = None
        self.dp = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        """
        Sets up the operator's configuration.

        This method is called when the operator is created. It initializes the writer
        and creates a DDS domain participant.
        """
        spec.input("input")
        self.message = self.data_schema()
        self.dp = dds.DomainParticipant(domain_id=self.domain_id)
        self.writer = dds.DataWriter(self.dp.implicit_publisher, dds.Topic(self.dp, self.topic, self.data_schema))

    def compute(self, op_input, op_output, context):
        """
        Computes the operator's logic.

        This method is called when the operator is executed. It receives data from the input port
        and publishes it to the DDS topic.

        Args:
            op_input: The input port of the operator. In this case, it is the ultrasound image.
            op_output: The output port of the operator. In this case, it is the ultrasound image.
            context: The context of the operator.
        """
        # Get received data
        data = op_input.receive("input")[""]

        # Check if it's a cupy array and convert to numpy if needed
        if hasattr(data, "get"):
            scan_converted_image_cpu = np.array(data).get()
        else:
            # Already a numpy array, no need for .get()
            scan_converted_image_cpu = np.array(data)

        # Publish the data
        self.message.data = scan_converted_image_cpu.tobytes()
        self.writer.write(self.message)


class StreamingSimulator(Application):
    """
    Streaming simulator application that subscribes to a probe position topic and publishes
    ultrasound data to a DDS topic.

    Args:
        output_topic: The name of the DDS topic to publish ultrasound data to.
        input_topic: The name of the DDS topic to subscribe to for probe position data.
        domain_id: The DDS domain ID to use for communication.
        out_width: The width of the output image.
        out_height: The height of the output image.
        start_pose: The initial pose of the probe.
        config_path: Path to a JSON configuration file with probe parameters.
    """

    def __init__(
        self,
        domain_id=0,
        output_topic="topic_ultrasound_data",
        input_topic="topic_ultrasound_info",
        out_width=224,
        out_height=224,
        start_pose=np.zeros((6,)),
        period=1 / 30.0,
        config_path=None,
    ):
        super().__init__()
        self.domain_id = domain_id
        self.output_topic = output_topic
        self.input_topic = input_topic
        self.out_width = out_width
        self.out_height = out_height
        self.start_pose = start_pose
        self.config_path = config_path
        self.period = period

    def compose(self):
        """
        Composes the application's operators and flows.
        """
        period_ns = int(self.period * 1e9)
        dds_sub = UltrasoundSimSubscriber(
            self,
            domain_id=self.domain_id,
            topic=self.input_topic,
            name="subscriber",
        )

        # Create the simulator
        simulator = Simulator(
            self,
            PeriodicCondition(self, period_ns),
            out_height=self.out_height,
            out_width=self.out_width,
            start_pose=self.start_pose,
            config_path=self.config_path,
            name="simulator",
        )
        simulator.metadata_policy = MetadataPolicy.RAISE
        dds_pub = UltrasoundSimPublisher(self, name="st", domain_id=self.domain_id, topic=self.output_topic)
        # Connect flows
        self.add_flow(dds_sub, simulator)
        self.add_flow(simulator, dds_pub)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="run local test")
    parser.add_argument(
        "--domain_id",
        type=int,
        default=int(os.environ.get("OVH_DDS_DOMAIN_ID", 0)),
        help="domain id. Default is 0.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("OVH_HEIGHT", 224)),
        help="height. Default is 224.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("OVH_WIDTH", 224)),
        help="width. Default is 224.",
    )
    parser.add_argument(
        "--topic_in",
        type=str,
        default="topic_ultrasound_info",
        help="topic name to consume prob pos. Default is topic_ultrasound_info.",
    )
    parser.add_argument(
        "--topic_out",
        type=str,
        default="topic_ultrasound_data",
        help="topic name to publish generated ultrasound data. Default is topic_ultrasound_data.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "path to custom JSON configuration file with probe parameters and simulation parameters. "
            "Default is None."
        ),
    )
    parser.add_argument(
        "--period",
        type=float,
        default=1 / 30.0,
        help="period of the simulation. Default is 1 / 30.0. (30 Hz)",
    )
    args = parser.parse_args()
    app = StreamingSimulator(
        domain_id=args.domain_id,
        output_topic=args.topic_out,
        input_topic=args.topic_in,
        out_width=args.width,
        out_height=args.height,
        period=args.period,
        config_path=args.config,
    )

    app.is_metadata_enabled = True
    app.run()


if __name__ == "__main__":
    main()
