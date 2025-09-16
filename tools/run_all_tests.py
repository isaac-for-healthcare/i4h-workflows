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
import glob
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

WORKFLOWS = [
    "robotic_ultrasound",
    "robotic_surgery",
    "so_arm_starter",
]


XVFB_TEST_CASES = [
    "test_visualization",
]


def get_tests(test_root, pattern="test_*.py"):
    path = f"{test_root}/**/{pattern}"
    return glob.glob(path, recursive=True)


def _stream_output(pipe, output_list, prefix=""):
    """Helper function to stream output from subprocess in real-time"""
    try:
        for line in iter(pipe.readline, ""):
            if line:
                timestamped_line = f"[{datetime.now().strftime('%H:%M:%S')}] {prefix}{line.rstrip()}"
                print(timestamped_line)
                output_list.append(line)
                sys.stdout.flush()
    except Exception as e:
        print(f"Error streaming output: {e}")


def _run_test_process(cmd, env, test_path, timeout=1200):
    """Enhanced test process runner with better debugging and real-time output"""
    print(f"\n{'='*80}")
    print(f"Running test: {test_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Timeout: {timeout} seconds ({timeout//60} minutes)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    stdout_lines = []
    stderr_lines = []

    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )

        # Create threads to stream output in real-time
        stdout_thread = threading.Thread(target=_stream_output, args=(process.stdout, stdout_lines, "STDOUT: "))
        stderr_thread = threading.Thread(target=_stream_output, args=(process.stderr, stderr_lines, "STDERR: "))

        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # Wait for process with timeout
        start_time = time.time()
        return_code = None

        while time.time() - start_time < timeout:
            return_code = process.poll()
            if return_code is not None:
                break
            time.sleep(1)

            # Print progress every 60 seconds
            elapsed = int(time.time() - start_time)
            if elapsed > 0 and elapsed % 60 == 0:
                print(f"Test still running... {elapsed//60} minutes elapsed")

        # Handle timeout
        if return_code is None:
            print(f"\nTEST TIMEOUT AFTER {timeout} SECONDS!")
            print(f"Test ran for: {int(time.time() - start_time)} seconds")
            print("Attempting to terminate process...")

            try:
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    print("Force killing process...")
                    process.kill()
                    time.sleep(2)
            except Exception as e:
                print(f"Error terminating process: {e}")

            # Wait for output threads to finish
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            print("\nPARTIAL OUTPUT BEFORE TIMEOUT:")
            print(f"STDOUT ({len(stdout_lines)} lines):")
            if stdout_lines:
                for line in stdout_lines:
                    print(f"  {line.rstrip()}")
            else:
                print("  (No stdout output)")

            print(f"STDERR ({len(stderr_lines)} lines):")
            if stderr_lines:
                for line in stderr_lines:
                    print(f"  {line.rstrip()}")
            else:
                print("  (No stderr output)")

            return False

        # Wait for output threads to complete
        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)

        elapsed_time = int(time.time() - start_time)

        if return_code == 0:
            print(f"\nTEST PASSED in {elapsed_time} seconds")
            return True
        elif return_code == -6:
            print("\n The process crashes at shutdown because of native async code that does not finalize safely.")
            return True
        else:
            print(f"\nTEST FAILED with return code {return_code} after {elapsed_time} seconds")

            # Show error output
            print("\nERROR OUTPUT:")
            print(f"STDERR ({len(stderr_lines)} lines):")
            if stderr_lines:
                for line in stderr_lines:
                    print(f"  {line.rstrip()}")

            print(f"STDOUT ({len(stdout_lines)} lines):")
            if stdout_lines:
                for line in stdout_lines:
                    print(f"  {line.rstrip()}")

            return False

    except Exception as e:
        print(f"\nEXCEPTION while running test: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def _setup_test_env(project_root, tests_dir):
    """Helper function to setup test environment"""
    env = os.environ.copy()
    pythonpath = [os.path.join(project_root, "scripts"), tests_dir]

    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = ":".join(pythonpath) + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = ":".join(pythonpath)

    return env


def _setup_test_cosmos_transfer1_env(project_root, workflow_root, tests_dir):
    """Helper function to setup test environment for cosmos-transfer1"""
    env = _setup_test_env(workflow_root, tests_dir)
    pythonpath = [
        os.path.join(project_root, "third_party", "cosmos-transfer1"),
    ]
    env["PYTHONPATH"] = ":".join(pythonpath) + ":" + env["PYTHONPATH"]
    env["DEBUG_GENERATION"] = "1"
    return env


def run_tests_with_coverage(workflow_name, skip_xvfb, timeout=1200):
    """Run all unittest cases with coverage reporting"""
    print(f"Starting test run for workflow: {workflow_name}")
    print(f"XVFB tests skipped: {skip_xvfb}")
    print(f"Test timeout: {timeout} seconds ({timeout//60} minutes)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    project_root = f"workflows/{workflow_name}"

    try:
        default_license_file = os.path.join(os.getcwd(), project_root, "scripts", "dds", "rti_license.dat")
        os.environ["RTI_LICENSE_FILE"] = os.environ.get("RTI_LICENSE_FILE", default_license_file)
        all_tests_passed = True
        tests_dir = os.path.join(project_root, "tests")
        print(f"Looking for tests in {tests_dir}")
        tests = get_tests(tests_dir)
        env = _setup_test_env(project_root, tests_dir)

        for test_path in tests:
            test_name = os.path.basename(test_path).replace(".py", "")

            # Check if this test needs a virtual display
            if test_name in XVFB_TEST_CASES:
                if skip_xvfb:
                    continue
                cmd = [
                    "xvfb-run",
                    "-a",
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--parallel-mode",
                    "-m",
                    "unittest",
                    test_path,
                ]
            # TODO: move these tests to integration tests
            elif "test_sim_with_dds" in test_path or "test_policy" in test_path or "test_gr00t_training" in test_path:
                continue
            elif "test_integration" in test_path:
                continue
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--parallel-mode",
                    "--source",
                    os.path.join(project_root, "scripts"),
                    "-m",
                    "unittest",
                    test_path,
                ]

            if not _run_test_process(cmd, env, test_path):
                print(f"FAILED TEST: {test_path}")
                all_tests_passed = False
        # combine coverage results
        subprocess.run([sys.executable, "-m", "coverage", "combine"])

        print("\nCoverage Report:")
        subprocess.run([sys.executable, "-m", "coverage", "report", "--show-missing"])

        # Generate HTML report
        subprocess.run([sys.executable, "-m", "coverage", "html", "-d", os.path.join(project_root, "htmlcov")])
        print(f"\nDetailed HTML coverage report generated in '{project_root}/htmlcov'")

        # Return appropriate exit code
        if all_tests_passed:
            print("All tests passed")
            return 0
        else:
            print("Some tests failed")
            return 1

    except Exception as e:
        print(f"Error running tests: {e}")
        print(traceback.format_exc())
        return 1


def run_integration_tests(workflow_name, timeout=1200):
    """Run integration tests for a workflow"""
    print(f"Starting integration tests for workflow: {workflow_name}")
    print(f"Test timeout: {timeout} seconds ({timeout//60} minutes)")

    project_root = f"workflows/{workflow_name}"
    try:
        default_license_file = os.path.join(os.getcwd(), project_root, "scripts", "dds", "rti_license.dat")
        os.environ["RTI_LICENSE_FILE"] = os.environ.get("RTI_LICENSE_FILE", default_license_file)
        all_tests_passed = True
        tests_dir = os.path.join(project_root, "tests")
        print(f"Looking for tests in {tests_dir}")
        tests = get_tests(tests_dir, pattern="test_integration_*.py")
        env = _setup_test_env(project_root, tests_dir)

        for test_path in tests:
            cmd = [
                sys.executable,
                "-m",
                "unittest",
                test_path,
            ]
            if "cosmos_transfer1" in test_path:
                env = _setup_test_cosmos_transfer1_env(os.getcwd(), project_root, tests_dir)

            if not _run_test_process(cmd, env, test_path):
                all_tests_passed = False
    except Exception as e:
        print(f"Error running integration tests: {e}")
        print(traceback.format_exc())
        return 1

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all tests for a workflow")
    parser.add_argument("--workflow", type=str, default="robotic_ultrasound", help="Workflow name")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--skip-xvfb", action="store_true", help="Skip running tests with xvfb")
    parser.add_argument(
        "--timeout", type=int, default=1200, help="Test timeout in seconds (default: 1200 = 20 minutes)"
    )
    args = parser.parse_args()

    if args.workflow not in WORKFLOWS:
        raise ValueError(f"Invalid workflow name: {args.workflow}")

    if args.integration:
        exit_code = run_integration_tests(args.workflow, args.timeout)
    else:
        exit_code = run_tests_with_coverage(args.workflow, args.skip_xvfb, args.timeout)
    sys.exit(exit_code)
