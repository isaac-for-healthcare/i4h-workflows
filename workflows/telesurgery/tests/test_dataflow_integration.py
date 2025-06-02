import unittest
import subprocess
import sys
import time
import os
import vgamepad as vg

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

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

    def start_process(self, cmd, log_file=None, env=None):
        stdout = open(log_file, "w") if log_file else subprocess.PIPE
        proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT)
        self.processes.append(proc)
        return proc

    def test_data_loop_with_gamepad(self):
        """Test the dataflow with a surgeon a simulation of a patient and a gamepad"""
        patient_cmd = [
            sys.executable, os.path.join(BASE_DIR, "scripts/patient/simulation/main.py"),
            "--domain_id", self.dds_domain_id,
            "--topic", self.camera_topic,
            "--api_port", self.api_port,
            "--width", "320", "--height", "240", "--timeline_play", "True"
        ]
        patient_proc = self.start_process(patient_cmd, env=self.env, log_file=self.patient_log)

        # Give patient simulation time to start
        time.sleep(60)

        # Start surgeon camera receiver
        surgeon_camera_cmd = [
            sys.executable, os.path.join(BASE_DIR, "scripts/surgeon/camera.py"),
            "--domain_id", self.dds_domain_id,
            "--topic", self.camera_topic,
            "--width", "320", "--height", "240"
        ]
        surgeon_proc = self.start_process(surgeon_camera_cmd, env=self.env, log_file=self.surgeon_camera_log)

        # Give time for data to flow
        time.sleep(20)

        # Check surgeon process log for evidence of received frames
        with open(self.surgeon_camera_log, "r") as f:
            camera_log = f.read()
        self.assertIn("fps:", camera_log, "Camera data not received on surgeon side")

        # Start surgeon gamepad control (simulate a control command)
        # NOTE: This will require a gamepad or a way to simulate input. For now, we use a virtual gamepad.
        gamepad = vg.VX360Gamepad()
        gamepad.right_joystick(x_value=0, y_value=0)
        gamepad.update()

        surgeon_gamepad_cmd = [
            sys.executable, os.path.join(BASE_DIR, "scripts/surgeon/gamepad.py"),
            "--api_host", self.api_host,
            "--api_port", self.api_port
        ]
        gamepad_proc = self.start_process(surgeon_gamepad_cmd)

        # Give time for control command to be sent and processed
        time.sleep(10)

        # Check patient log for evidence of robot control command received
        with open(self.patient_log, "r") as f:
            patient_log = f.read()
        self.assertTrue(
            "Update (" in patient_log or "set_mira" in patient_log,
            "Robot control command not received on patient side"
        )

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

if __name__ == "__main__":
    unittest.main() 