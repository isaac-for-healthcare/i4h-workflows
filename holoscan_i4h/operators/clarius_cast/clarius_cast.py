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

import ctypes
import os
from io import BytesIO

import holoscan
import numpy as np
import rti.connextdds as dds
from dds.schemas.usp_data import UltraSoundProbeData
from holoscan.core import Operator, OperatorSpec
from PIL import Image

# load the libcast.so shared library
libcast_handle = ctypes.CDLL("libcast.so", ctypes.RTLD_GLOBAL)._handle
# load the pyclariuscast.so shared library
ctypes.cdll.LoadLibrary("pyclariuscast.so")

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
