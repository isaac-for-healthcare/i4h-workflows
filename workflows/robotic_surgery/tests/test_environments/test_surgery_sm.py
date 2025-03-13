import os
import signal
import subprocess
import threading
import time
import unittest
from parameterized import parameterized

SCRIPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")


def monitor_output(process, target_line, found_event):
    """Monitor process output for target_line and set event when found."""
    try:
        for line in iter(process.stdout.readline, ""):
            if target_line in line:
                found_event.set()
    except (ValueError, IOError):
        # Handle case where stdout is closed
        pass


def run_with_monitoring(command, timeout_seconds, target_line=None):
    # Start the process with pipes for output
    env = os.environ.copy()
    env["PYTHONPATH"] = SCRIPTS_PATH
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        text=True,
        bufsize=1,  # Line buffered
        preexec_fn=os.setsid if os.name != "nt" else None,  # Create a new process group on Unix
        env=env,
        cwd=SCRIPTS_PATH,
    )

    # Event to signal when target line is found
    found_event = threading.Event()

    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_output, args=(process, target_line, found_event))
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
    ("python simulation/scripts/environments/state_machine/lift_block_sm.py --headless", 20, "Environment stepped"),
    ("python simulation/scripts/environments/state_machine/lift_needle_sm.py --headless", 20, "Environment stepped"),
    ("python simulation/scripts/environments/state_machine/reach_dual_psm_sm.py --headless", 20, "Environment stepped"),
    ("python simulation/scripts/environments/state_machine/reach_psm_sm.py --headless", 20, "Environment stepped"),
    ("python simulation/scripts/environments/state_machine/reach_star_sm.py --headless", 20, "Environment stepped"),
]

class TestSurgerySM(unittest.TestCase):

    @parameterized.expand(SM_CASES)
    def test_surgery_sm(self, command, timeout, target_line):
        # Run and monitor command
        exit_code, found_target = run_with_monitoring(command, timeout, target_line)
        self.assertTrue(found_target)


if __name__ == "__main__":
    unittest.main()
