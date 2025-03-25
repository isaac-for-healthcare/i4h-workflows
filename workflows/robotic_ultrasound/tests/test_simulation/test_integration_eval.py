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
import signal
import subprocess
import threading
import time
import unittest

from parameterized import parameterized


def monitor_output(process, found_event, target_line=None):
    """Monitor process output for target_line and set event when found."""
    try:
        if target_line:
            for line in iter(process.stdout.readline, ""):
                if target_line in line:
                    found_event.set()
                    break  # todo
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


SM_CASES = [
    (
        "python -u workflows/robotic_ultrasound/scripts/simulation/environments/state_machine/pi0_policy/eval.py --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 --enable_camera     --repo_id i4h/sim_liver_scan --headless",
        120,
        "Resetting the environment.",
    ),
]

class TestSurgerySM(unittest.TestCase):
    @parameterized.expand(SM_CASES)
    def test_surgery_sm(self, command, timeout, target_line):
        # Run and monitor command
        exit_code, found_target = run_with_monitoring(command, timeout, target_line)
        self.assertTrue(found_target)


if __name__ == "__main__":
    unittest.main()
