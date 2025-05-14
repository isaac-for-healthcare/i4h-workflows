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

import hashlib
import importlib
import os
import pathlib
import signal
import subprocess
import threading
import time
from unittest import skipUnless


def get_md5_checksum(output_dir, model_name, md5_checksum_lookup):
    for key, value in md5_checksum_lookup.items():
        if key.startswith(model_name):
            print(f"Verifying checkpoint {key}...")
            file_path = os.path.join(output_dir, key)
            # File must exist
            if not pathlib.Path(file_path).exists():
                print(f"Checkpoint {key} does not exist.")
                return False
            # File must match give MD5 checksum
            with open(file_path, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            if file_md5 != value:
                print(f"MD5 checksum of checkpoint {key} does not match.")
                return False
    print(f"Model checkpoints for {model_name} exist with matched MD5 checksums.")
    return True


def requires_rti(func):
    RTI_AVAILABLE = bool(os.getenv("RTI_LICENSE_FILE") and os.path.exists(os.getenv("RTI_LICENSE_FILE")))
    return skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")(func)


def requires_cosmos_transfer1(func):
    # check if cosmos-transfer1 is installed
    spec = importlib.util.find_spec("cosmos_transfer1")
    COSMOS_TRANSFER1_AVAILABLE = spec is not None
    return skipUnless(
        COSMOS_TRANSFER1_AVAILABLE,
        "cosmos-transfer1 is not installed. "
        "Please install it using "
        "`python tools/install_deps.py --workflow robotic_ultrasound/cosmos_transfer1`",
    )(func)


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
