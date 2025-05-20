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
import traceback

WORKFLOWS = [
    "robotic_ultrasound",
    "robotic_surgery",
]


XVFB_TEST_CASES = [
    "test_visualization",
]


def get_tests(test_root, pattern="test_*.py"):
    path = f"{test_root}/**/{pattern}"
    return glob.glob(path, recursive=True)


def _run_test_process(cmd, env, test_path):
    """Helper function to run a test process and handle its output"""
    print(f"Running test: {test_path}")

    try:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=600)
        # Filter out extension loading messages
        filtered_stdout = "\n".join(
            [line for line in stdout.split("\n") if not ("[ext:" in line and "startup" in line)]
        )
        filtered_stderr = "\n".join(
            [line for line in stderr.split("\n") if not ("[ext:" in line and "startup" in line)]
        )

        # Print filtered output
        if filtered_stdout.strip():
            print(filtered_stdout)
        if filtered_stderr.strip():
            print(filtered_stderr)

        return process.returncode == 0

    except subprocess.TimeoutExpired:
        print("❌ Test run timed out! ", cmd)
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


def run_tests_with_coverage(workflow_name, skip_xvfb):
    """Run all unittest cases with coverage reporting"""
    print(f"Running tests with xvfb skipped: {skip_xvfb}")
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
            elif "test_sim_with_dds" in test_path or "test_policy" in test_path:
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


def run_integration_tests(workflow_name):
    """Run integration tests for a workflow"""
    project_root = f"workflows/{workflow_name}"
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

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all tests for a workflow")
    parser.add_argument("--workflow", type=str, default="robotic_ultrasound", help="Workflow name")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--skip-xvfb", action="store_true", help="Skip running tests with xvfb")
    args = parser.parse_args()

    if args.workflow not in WORKFLOWS:
        raise ValueError(f"Invalid workflow name: {args.workflow}")

    if args.integration:
        exit_code = run_integration_tests(args.workflow)
    else:
        exit_code = run_tests_with_coverage(args.workflow, args.skip_xvfb)
    sys.exit(exit_code)
