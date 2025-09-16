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

import os
import re
import subprocess
import sys
import time
import unittest
from unittest import skipUnless

import vgamepad as vg

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def requires_rti(func):
    RTI_AVAILABLE = bool(os.getenv("RTI_LICENSE_FILE") and os.path.exists(os.getenv("RTI_LICENSE_FILE")))
    return skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")(func)


@requires_rti
class TestTelesurgeryDataLoop(unittest.TestCase):
    """Test the dataloop of the telesurgery workflow"""

    def setUp(self):
        # Use unique DDS topics and API port for testing
        self.dds_domain_id = "999"  # Use a test domain
        self.camera_topic = "integration_test/telesurgery/camera"
        self.api_port = "18081"
        self.api_host = "127.0.0.1"
        self.processes = []
        self.surgeon_camera_log = "/tmp/surgeon_camera_test.log"
        self.patient_log = "/tmp/patient_sim_test.log"
        self.env = os.environ.copy()
        self.env["PATIENT_IP"] = "127.0.0.1"
        self.env["SURGEON_IP"] = "127.0.0.1"
        self.env["NDDS_DISCOVERY_PEER"] = "127.0.0.1"
        self.env["PYTHONUNBUFFERED"] = "1"
        self.log_files = []

    def start_process(self, cmd, log_file=None, env=None):
        if log_file:
            stdout = open(log_file, "w", encoding="utf-8")
            self.log_files.append(stdout)
        else:
            stdout = subprocess.PIPE
        proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT)
        self.processes.append(proc)
        return proc

    def test_data_loop_with_gamepad(self):
        """Test the dataflow with a surgeon a simulation of a patient and a gamepad"""
        patient_cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "scripts/patient/simulation/main.py"),
            "--domain_id",
            self.dds_domain_id,
            "--topic",
            self.camera_topic,
            "--api_port",
            self.api_port,
            "--width",
            "320",
            "--height",
            "240",
            "--timeline_play",
            "True",
        ]
        self.start_process(patient_cmd, env=self.env, log_file=self.patient_log)

        # Give patient simulation time to start
        time.sleep(60)

        # Start surgeon camera receiver
        surgeon_camera_cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "scripts/surgeon/camera.py"),
            "--domain_id",
            self.dds_domain_id,
            "--topic",
            self.camera_topic,
            "--width",
            "320",
            "--height",
            "240",
        ]
        self.start_process(surgeon_camera_cmd, env=self.env, log_file=self.surgeon_camera_log)

        # Give time for data to flow
        time.sleep(20)

        # Check surgeon process log for evidence of received frames
        with open(self.surgeon_camera_log, encoding="utf-8") as f:
            camera_log = f.read()
        self.assertIn("fps:", camera_log, "Camera data not received on surgeon side")

        # Start surgeon gamepad control (simulate a control command)
        # NOTE: This will require a gamepad or a way to simulate input. For now, we use a virtual gamepad.
        gamepad = vg.VX360Gamepad()
        gamepad.right_joystick(x_value=0, y_value=0)
        gamepad.update()

        surgeon_gamepad_cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "scripts/surgeon/gamepad.py"),
            "--api_host",
            self.api_host,
            "--api_port",
            self.api_port,
        ]
        self.start_process(surgeon_gamepad_cmd)

        # Give time for control command to be sent and processed
        time.sleep(10)

        # Check patient log for evidence of robot control command received
        with open(self.patient_log, encoding="utf-8") as f:
            patient_log = f.read()
        self.assertTrue(
            "Update (" in patient_log or "set_mira" in patient_log, "Robot control command not received on patient side"
        )

        # Check DDS topic and domain ID consistency
        surgeon_dds_match = re.search(r"Reading data from DDS: ([\w/_]+):(\d+)", camera_log)
        self.assertIsNotNone(surgeon_dds_match, "DDS read info not found in surgeon log")
        surgeon_topic, surgeon_domain_id = surgeon_dds_match.groups()

        patient_dds_match = re.search(r"Writing data to DDS: ([\w/_]+):(\d+)", patient_log)
        self.assertIsNotNone(patient_dds_match, "DDS write info not found in patient log")
        patient_topic, patient_domain_id = patient_dds_match.groups()

        self.assertEqual(surgeon_topic, self.camera_topic, "Surgeon topic mismatch")
        self.assertEqual(patient_topic, self.camera_topic, "Patient topic mismatch")
        self.assertEqual(surgeon_topic, patient_topic, "Surgeon and Patient topics do not match")

        self.assertEqual(surgeon_domain_id, self.dds_domain_id, "Surgeon domain ID mismatch")
        self.assertEqual(patient_domain_id, self.dds_domain_id, "Patient domain ID mismatch")
        self.assertEqual(surgeon_domain_id, patient_domain_id, "Surgeon and Patient domain IDs do not match")

    def tearDown(self):
        # Terminate all started processes
        for proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=100)
            except Exception:
                proc.kill()
        # Clean up log files
        if os.path.exists(self.surgeon_camera_log):
            os.remove(self.surgeon_camera_log)
        if os.path.exists(self.patient_log):
            os.remove(self.patient_log)
        for f in self.log_files:
            try:
                f.close()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
