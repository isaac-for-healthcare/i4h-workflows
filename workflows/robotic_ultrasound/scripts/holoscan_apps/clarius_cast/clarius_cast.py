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
import ctypes
import os
import sys
from io import BytesIO

import holoscan
import numpy as np
import rti.connextdds as dds
from dds.schemas.usp_data import UltraSoundProbeData
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.resources import UnboundedAllocator
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))

# load the libcast.so shared library
libcast_handle = ctypes.CDLL(f"{script_dir}/lib/libcast.so", ctypes.RTLD_GLOBAL)._handle
# load the pyclariuscast.so shared library
ctypes.cdll.LoadLibrary(f"{script_dir}/lib/pyclariuscast.so")

sys.path.append(f"{script_dir}/lib")
import pyclariuscast

# The current image
img = None


def processed_image_cb(image, width, height, sz, microns_per_pixel, timestamp, angle, imu):
    """
    Callback function that processes a scan-converted image.

    Parameters:
        image: The processed image data.
        width: The width of the image in pixels.
        height: The height of the image in pixels.
        sz: Full size of the image in bytes.
        microns_per_pixel: Microns per pixel.
        timestamp: The timestamp of the image in nanoseconds.
        angle: Acquisition angle for volumetric data.
        imu: IMU data tagged with the frame.
    """
    bpp = sz / (width * height)

    global img

    if bpp == 4:
        # Handle RGBA
        img = Image.frombytes("RGBA", (width, height), image)
    else:
        # Handle JPEG
        img = Image.open(BytesIO(image))


def raw_image_cb(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    """
    Callback function for raw image data.

    Parameters:
        image: The raw pre scan-converted image data, uncompressed 8-bit or jpeg compressed.
        lines: Number of lines in the data.
        samples: Number of samples in the data.
        bps: Bits per sample.
        axial: Microns per sample.
        lateral: Microns per line.
        timestamp: The timestamp of the image in nanoseconds.
        jpg: JPEG compression size if the data is in JPEG format.
        rf: Flag indicating if the image is radiofrequency data.
        angle: Acquisition angle for volumetric data.
    """
    # Not used in sample app
    return


def spectrum_image_cb(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    """
    Callback function for spectrum image data.

    Parameters:
        image: The spectral image data.
        lines: The number of lines in the spectrum.
        samples: The number of samples per line.
        bps: Bits per sample.
        period: Line repetition period of the spectrum.
        micronsPerSample: Microns per sample for an M spectrum.
        velocityPerSample: Velocity per sample for a PW spectrum.
        pw: Flag that is True for a PW spectrum, False for an M spectrum
    """
    # Not used in sample app
    return


def imu_data_cb(imu):
    """
    Callback function for IMU data.

    Parameters:
        imu: Inertial data tagged with the frame.
    """
    # Not used in sample app
    return


def freeze_cb(frozen):
    """
    Callback function for freeze state changes.

    Parameters:
        frozen: The freeze state of the imaging system.
    """
    if frozen:
        print("\nClarius: Run imaging")
    else:
        print("\nClarius: Stop imaging")
    return


def buttons_cb(button, clicks):
    """
    Callback function for button presses.

    Parameters:
        button: The button that was pressed.
        clicks: The number of clicks performed.
    """
    print(f"button pressed: {button}, clicks: {clicks}")
    return


class NoOp(Operator):
    """A sink operator that takes input and discards them."""

    def setup(self, spec: OperatorSpec):
        """Define input ports."""
        spec.input("input")

    def compute(self, op_input, op_output, context):
        """Receive and discard input frames."""
        op_input.receive("input")


class ClariusCastOp(Operator):
    """
    Operator to interface with a Clarius UltraSound Probe using Clarius Cast APIs.
    Captures processed image data from a Clarius Probe and publishes it via DDS.
    """

    def __init__(self, fragment, *args, ip, port, domain_id, width, height, topic_out, **kwargs):
        """
        Initializes the ClariusCastOp operator.

        Parameters:
            fragment: The fragment this operator belongs to.
            ip: IP address of the Clarius probe.
            port: Port number for Clarius probe.
            domain_id: Domain ID for DDS communication.
            width: Width of the image in pixels.
            height: Height of the image in pixels.
            topic_out: The DDS topic to publish ultrasound data.
        """
        self.ip = ip
        self.port = port
        self.domain_id = domain_id
        self.width = width
        self.height = height
        self.topic_out = topic_out
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the output for this operator."""
        spec.output("output")

    def start(self):
        """Initialize and start the Clarius Cast connection and DDS publisher."""
        # initialize
        path = os.path.expanduser("~/")
        cast = pyclariuscast.Caster(
            processed_image_cb, raw_image_cb, imu_data_cb, spectrum_image_cb, freeze_cb, buttons_cb
        )
        self.cast = cast
        ret = cast.init(path, self.width, self.height)

        if ret:
            print("Initialization succeeded")
            # Use JPEG format
            JPEG = 2
            ret = cast.setFormat(JPEG)
            if ret:
                print("Setting format to JPEG")
            else:
                print("Failed setting format to JPEG")
                # unload the shared library before destroying the cast object
                ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
                cast.destroy()
                exit()

            ret = cast.connect(self.ip, self.port, "research")
            if ret:
                print(f"Connected to {self.ip} on port {self.port}")
            else:
                print("Connection failed")
                # unload the shared library before destroying the cast object
                ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
                cast.destroy()
                exit()
        else:
            print("Initialization failed")
            return

        dp = dds.DomainParticipant(domain_id=self.domain_id)
        topic = dds.Topic(dp, self.topic_out, UltraSoundProbeData)
        self.writer = dds.DataWriter(dp.implicit_publisher, topic)

    def compute(self, op_input, op_output, context):
        """Process the current image and publish it to DDS."""
        global img

        if img is None:
            return

        image = np.array(img)
        d = UltraSoundProbeData()
        d.data = image.tobytes()
        self.writer.write(d)
        out_message = {"image": holoscan.as_tensor(image)}
        op_output.emit(out_message, "output")


class ClariusCastApp(Application):
    """Application for streaming Ultrasound image data using Clarius Cast APIs"""

    def __init__(self, ip, port, domain_id, height, width, topic_out, show_holoviz):
        """
        Initializes the ClariusCastApp application.

        Parameters:
            ip: IP address of the Clarius probe.
            port: Port number for the Clarius probe.
            domain_id: DDS domain ID.
            height: Height of the image in pixels.
            width: Width of the image in pixels.
            topic_out: The DDS topic name for publishing ultrasound data.
            show_holoviz: Flag to indicate if Holoviz should be shown.
        """
        self.ip = ip
        self.port = port
        self.domain_id = domain_id
        self.height = height
        self.width = width
        self.topic_out = topic_out
        self.show_holoviz = show_holoviz
        super().__init__()

    def compose(self):
        """Compose the operators and define the workflow."""
        clarius_cast = ClariusCastOp(
            self,
            name="clarius_cast",
            ip=self.ip,
            port=self.port,
            domain_id=self.domain_id,
            height=self.height,
            width=self.width,
            topic_out=self.topic_out,
        )

        pool = UnboundedAllocator(self, name="pool")
        holoviz = HolovizOp(
            self,
            allocator=pool,
            name="holoviz",
            window_title="Clarius Cast",
            width=self.width,
            height=self.height,
        )
        noop = NoOp(self)

        # Define the workflow
        if self.show_holoviz:
            self.add_flow(clarius_cast, holoviz, {("output", "receivers")})
        else:
            self.add_flow(clarius_cast, noop)


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the Clarius Cast application")
    parser.add_argument("--test", action="store_true", help="show holoviz")
    parser.add_argument(
        "--ip",
        type=str,
        default="192.168.1.1",
        help="IP address of Clarius probe",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5828,
        help="port # for Clarius probe",
    )
    parser.add_argument(
        "--domain_id",
        type=int,
        default=int(os.environ.get("OVH_DDS_DOMAIN_ID", 0)),
        help="domain id",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("OVH_HEIGHT", 480)),
        help="height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("OVH_WIDTH", 640)),
        help="width",
    )
    parser.add_argument(
        "--topic_out",
        type=str,
        default="topic_ultrasound_data",
        help="topic name to publish generated ultrasound data",
    )
    args = parser.parse_args()

    app = ClariusCastApp(args.ip, args.port, args.domain_id, args.height, args.width, args.topic_out, args.test)
    app.run()


if __name__ == "__main__":
    main()
