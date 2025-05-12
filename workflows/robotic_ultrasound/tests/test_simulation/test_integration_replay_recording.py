import unittest
import os
import h5py
import numpy as np
import shutil

import unittest
import subprocess
import threading
import time
import signal
# from helpers import run_with_monitoring
from parameterized import parameterized
from pathlib import Path

TESTS_PATH = Path(__file__).parents[1].as_posix()

TEST_DATA_DIR = os.path.join(TESTS_PATH, "temp_test_hdf5_data_dir")
FAKE_HDF5_FILENAME = "data_0.hdf5"
ORGAN_POS = np.array([[0.69, -0.11, 0.09, -0.21, -0.00, 0.00, 0.98]]*6)
ROBOT_JOINT_POS = np.array([[0.23, -0.75, -0.22, -2.47, -0.13, 1.73, 0.077]]*6)
ROBOT_JOINT_VEL = np.array([[0.39, 2.32, -1.74, 2.28, 0.53, -1.36, -1.74]]*6)
ACTIONS_DATA = np.array([[0.36, -0.15, -0.11, -0.01, 0.016, 0.012]]*6)

TORSO_OBS_KEY = "observations/torso_obs"
JOINT_STATE_KEY = "abs_joint_pos"
JOINT_VEL_KEY = "observations/joint_vel"
ACTION_KEY_IN_HDF5 = "action"

def monitor_output(process, found_event, target_line=None):
    """Monitor process output for target_line and set event when found."""
    try:
        if target_line:
            for line in iter(process.stdout.readline, ""):
                if target_line in line:
                    found_event.set()
                    break
    except (ValueError, IOError):
        # Handle case where stdout is closed
        pass

def run_with_monitoring(command, timeout_seconds, target_line=None):
    # Start the process with pipes for output
    env = os.environ.copy()
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        text=True,
        bufsize=1,  # Line buffered
        preexec_fn=os.setsid if os.name != "nt" else None,  # Create a new process group on Unix
        env=env,
    )

    # Event to signal when target line is found
    found_event = threading.Event()

    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_output, args=(process, found_event, target_line))
    monitor_thread.daemon = True
    monitor_thread.start()

    target_found = False

    try:
        # Wait for either timeout or target line found
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if target_line and found_event.is_set():
                target_found = True
                break

            # Check if process has already terminated
            if process.poll() is not None:
                break

            time.sleep(0.1)

        # If we get here, either timeout occurred or process ended
        if process.poll() is None:  # Process is still running
            print(f"Sending SIGINT after {timeout_seconds} seconds...")

            if os.name != "nt":  # Unix/Linux/MacOS
                # Send SIGINT to the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            else:  # Windows
                process.send_signal(signal.CTRL_C_EVENT)

            # Give the process some time to handle the signal and exit gracefully
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process didn't terminate after SIGINT, force killing...")
                if os.name != "nt":  # Unix/Linux/MacOS
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:  # Windows
                    process.kill()

    except Exception as e:
        print(f"Error during process execution: {e}")
        if process.poll() is None:
            process.kill()

    finally:
        # Ensure we close all pipes and terminate the process
        try:
            # Try to get any remaining output, but with a short timeout
            remaining_output, _ = process.communicate(timeout=2)
            if remaining_output:
                print(remaining_output)
        except subprocess.TimeoutExpired:
            # If communicate times out, force kill the process
            process.kill()
            process.communicate()

        # If the process is somehow still running, make sure it's killed
        if process.poll() is None:
            process.kill()
            process.communicate()

        # Check if target was found
        if not target_found and found_event.is_set():
            target_found = True

    return process.returncode, target_found


TEST_CASES = [
    (
        f"python -u -m simulation.environments.state_machine.replay_recording {TEST_DATA_DIR} --headless --enable_camera",
        300,
        "Completed all episodes",
    ),
    (
        f"python -u -m simulation.environments.state_machine.replay_recording {TEST_DATA_DIR}/data_0.hdf5 --headless --enable_camera",
        300,
        "Completed all episodes",
    ),
]



class TestReplayRecording(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        cls.fake_hdf5_path = os.path.join(TEST_DATA_DIR, FAKE_HDF5_FILENAME)
        print(f"Creating fake HDF5 file at: {cls.fake_hdf5_path}")
        with h5py.File(cls.fake_hdf5_path, "w") as f:
            base_path = "data/demo_0"
            f.create_dataset(f"{base_path}/{TORSO_OBS_KEY}", data=ORGAN_POS)
            f.create_dataset(f"{base_path}/{JOINT_STATE_KEY}", data=ROBOT_JOINT_POS)
            f.create_dataset(f"{base_path}/{JOINT_VEL_KEY}", data=ROBOT_JOINT_VEL)
            f.create_dataset(f"{base_path}/{ACTION_KEY_IN_HDF5}", data=ACTIONS_DATA)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DATA_DIR):
            shutil.rmtree(TEST_DATA_DIR)


    @parameterized.expand(TEST_CASES)
    def test_policy_eval(self, command, timeout, target_line):
        # Run and monitor command
        _, found_target = run_with_monitoring(command, timeout, target_line)
        self.assertTrue(found_target)


if __name__ == "__main__":
    unittest.main() 