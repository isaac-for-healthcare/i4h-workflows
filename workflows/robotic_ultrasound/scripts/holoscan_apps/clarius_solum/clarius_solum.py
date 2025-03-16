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
from time import sleep

import holoscan
import numpy as np
import rti.connextdds as dds
from dds.schemas.usp_data import UltraSoundProbeData
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.resources import UnboundedAllocator
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))

# load the libsolum.so shared library
libsolum_handle = ctypes.CDLL(f"{script_dir}/lib/libsolum.so", ctypes.RTLD_GLOBAL)._handle

sys.path.append(f"{script_dir}/lib")
import pysolum

# Is probe connected
connected = False
# Probe is running
running = False
# The current image
img = None


def connect_cb(res, port, status):
    """
    Callback function for connection events.

    Parameters:
        res: The connection result.
        port: UDP port used for streaming.
        status: The status message.
    """
    print(f"Connection: {res} {status}")
    if res == pysolum.CusConnection.ProbeConnected:
        global connected
        connected = True
        print(f"Streaming on port: {port}")


def cert_cb(days_valid):
    """
    Callback function when the certificate is set.

    Parameters:
        days_valid: Number of days the certificate is valid.
    """
    print(f"Certificate valid for {days_valid} days.")


def power_down_cb(res, tm):
    """
    Callback function when the device is powered down.

    Parameters:
        res: The power down reason.
        tm: Time in seconds until probe powers down, 0 for immediate shutdown.
    """
    print(f"Powering down: {res} in {tm} seconds")


def processed_image_cb(image, width, height, size, micros_per_pixel, origin_x, origin_y, fps):
    """
    Callback function when a new processed image is streamed.

    Parameters:
        image: The scan-converted image data.
        width: Image width in pixels.
        height: Image height in pixels.
        size: Full size of the image.
        microns_per_pixel: Microns per pixel.
        origin_x: Image origin in microns (horizontal axis).
        origin_y: Image origin in microns (vertical axis).
        fps: Frames per second.
    """
    bpp = size / (width * height)

    global img

    if bpp == 4:
        img = Image.frombytes("RGBA", (width, height), image).convert("L")
    else:
        img = Image.open(BytesIO(image)).convert("L")


def imu_port_cb(port):
    """
    Callback function for new IMU data streaming port.

    Parameters:
        port: The new IMU data UDP streaming port.
    """
    # Not used in sample code
    return


def imu_data_cb(pos):
    """
    Callback function for new IMU data.

    Parameters:
        pos: Positional information data tagged with the image.
    """
    # Not used in sample code
    return


def imaging_cb(state, imaging):
    """
    Imaging callback function.

    Parameters:
        state: The imaging state.
        imaging: 1 if running, 0 if stopped.
    """
    if imaging == 0:
        print(f"State: {state} imaging: Stopped")
    else:
        global running
        running = True
        print(f"State: {state} imaging: Running")


def error_cb(code, msg):
    """
    Callback function for error events.

    Parameters:
        code: Error code associated with the error.
        msg: The error message.
    """
    print(f"Error: {code}: {msg}")


def buttons_cb(button, clicks):
    """
    Callback function when a button is pressed.

    Parameters:
        button: The button that was pressed.
        clicks: Number of clicks performed.
    """
    print(f"button pressed: {button}, clicks: {clicks}")


class NoOp(Operator):
    """A sink operator that takes input and discards them."""

    def setup(self, spec: OperatorSpec):
        """Define input ports."""
        spec.input("input")

    def compute(self, op_input, op_output, context):
        """Receive and discard input frames."""
        op_input.receive("input")


class ClariusSolumOp(Operator):
    """
    Operator to interface with a Clarius UltraSound Probe using Clarius Solum APIs.
    Captures processed image data from a Clarius Probe and publishes it via DDS.
    """

    def __init__(self, fragment, *args, ip, port, cert, model, app, domain_id, width, height, topic_out, **kwargs):
        """
        Initializes the ClariusSolumOp operator.

        Parameters:
            fragment: The fragment this operator belongs to.
            ip: IP address of the Clarius probe.
            port: Port number for Clarius probe.
            cert: Path to the probe certificate.
            model: The Clarius probe model name.
            app: The ultrasound application to perform.
            domain_id: Domain ID for DDS communication.
            width: Width of the image in pixels.
            height: Height of the image in pixels.
            topic_out: The DDS topic to publish ultrasound data.
        """
        self.ip = ip
        self.port = port
        self.cert = cert
        self.model = model
        self.app = app
        self.domain_id = domain_id
        self.width = width
        self.height = height
        self.topic_out = topic_out
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the output for this operator."""
        spec.output("output")

    def stop(self):
        """Stops imaging on the Clarius probe."""
        print("Clarius: Stop imaging")
        self.solum.stop_imaging()

    def start(self):
        """
        Initializes and starts the Clarius Solum API connection.
        Establishes a connection, loads the application, and starts imaging.
        """
        # initialize
        path = os.path.expanduser("~/")

        solum = pysolum.Solum(
            connect_cb,
            cert_cb,
            power_down_cb,
            processed_image_cb,
            imu_port_cb,
            imu_data_cb,
            imaging_cb,
            buttons_cb,
            error_cb,
        )

        self.solum = solum
        ret = solum.init(path, self.width, self.height)

        if ret:
            global connected

            solum.set_certificate(self.cert)
            ret = solum.connect(self.ip, self.port, "research")

            while not connected:
                sleep(1)

            if ret:
                print(f"Connected to {self.ip} on port {self.port}")
            else:
                print("Connection failed")
                # unload the shared library before destroying the solum object
                ctypes.CDLL("libc.so.6").dlclose(libsolum_handle)
                solum.destroy()
                exit()

            solum.load_application(self.model, self.app)
            sleep(5)
            solum.run_imaging()

            while not running:
                sleep(1)

        else:
            print("Initialization failed")
            return

        dp = dds.DomainParticipant(domain_id=self.domain_id)
        topic = dds.Topic(dp, self.topic_out, UltraSoundProbeData)
        self.writer = dds.DataWriter(dp.implicit_publisher, topic)
        print(f"Creating writer for topic: {self.domain_id}:{self.topic_out}")

    def compute(self, op_input, op_output, context):
        """Processes the current ultrasound image and publishes it via DDS and to the downstream operator."""
        global img

        if img is None:
            return

        image = np.array(img)
        d = UltraSoundProbeData()
        d.data = image.tobytes()
        self.writer.write(d)
        out_message = {"image": holoscan.as_tensor(image)}
        op_output.emit(out_message, "output")


class ClariusSolumApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:
    - ClariusSolumOp
    - HolovizOp

    The ClariusSolumOp reads a video file and sends the frames to the HolovizOp.
    The HolovizOp displays the frames.
    """

    def __init__(self, ip, port, cert, model, app, domain_id, height, width, topic_out, show_holoviz):
        """Initializes the ClariusSolumApp application.

        Parameters:
            ip: IP address of the Clarius probe.
            port: Port number for Clarius probe.
            cert: Path to the probe certificate.
            model: The Clarius probe model name.
            app: The ultrasound application to perform.
            domain_id: Domain ID for DDS communication.
            height: Height of the image in pixels.
            width: Width of the image in pixels.
            topic_out: The DDS topic to publish ultrasound data.
            show_holoviz: Flag to enable visualization.
        """
        self.ip = ip
        self.port = port
        self.cert = cert
        self.model = model
        self.app = app
        self.domain_id = domain_id
        self.height = height
        self.width = width
        self.topic_out = topic_out
        self.show_holoviz = show_holoviz
        super().__init__()

    def compose(self):
        """Compose the operators and define the workflow."""
        clarius_solum = ClariusSolumOp(
            self,
            name="clarius_solum",
            ip=self.ip,
            port=self.port,
            cert=self.cert,
            model=self.model,
            app=self.app,
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
            window_title="Clarius Solum",
            width=self.width,
            height=self.height,
        )
        noop = NoOp(self)

        # Define the workflow
        if self.show_holoviz:
            self.add_flow(clarius_solum, holoviz, {("output", "receivers")})
        else:
            self.add_flow(clarius_solum, noop)


def main():
    """Parse command-line arguments and run the application."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="show holoviz")
    parser.add_argument(
        "--ip",
        type=str,
        default="192.168.68.50",
        help="IP address of Clarius probe",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="port # for Clarius probe",
    )
    parser.add_argument(
        "--cert",
        type=str,
        default=os.environ.get("CLARIUS_CERTIFICATE", f"{cwd}/ClariusOne.cert"),
        help="The required certificate to use Clarius Solum APIs",
    )
    # Only support C3HD3 for now
    parser.add_argument(
        "--model",
        choices=["C3HD3"],
        default="C3HD3",
        help="The model of the Clarius Probe",
    )
    parser.add_argument(
        "--app",
        choices=[
            "abdomen",
            "bladder",
            "cardiac",
            "lung",
            "msk",
            "msk_hip",
            "msk_shoulder",
            "msk_spine",
            "nerve",
            "obgyn",
            "oncoustics_liver",
            "pelvic",
            "prostate",
            "research",
            "superficial",
            "vascular",
        ],
        default="abdomen",
        help="The Ultrasound Application to run",
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

    app = ClariusSolumApp(
        args.ip,
        args.port,
        args.cert,
        args.model,
        args.app,
        args.domain_id,
        args.height,
        args.width,
        args.topic_out,
        args.test,
    )
    app.run()


if __name__ == "__main__":
    main()
