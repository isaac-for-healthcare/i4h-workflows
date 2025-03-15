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
    "test_orientation",
    "test_transform_matrix",
    "test_visualization",
]


def get_tests(test_root):
    path = f"{test_root}/**/test_*.py"
    return glob.glob(path, recursive=True)


def run_tests_with_coverage(workflow_name):
    """Run all unittest cases with coverage reporting"""
    project_root = f"workflows/{workflow_name}"

    try:
        # TODO: add license file to secrets
        default_license_file = os.path.join(os.getcwd(), project_root, "scripts", "dds", "rti_license.dat")
        os.environ["RTI_LICENSE_FILE"] = os.environ.get("RTI_LICENSE_FILE", default_license_file)
        all_tests_passed = True
        tests_dir = os.path.join(project_root, "tests")
        print(f"Looking for tests in {tests_dir}")
        tests = get_tests(tests_dir)

        for test_path in tests:
            test_name = os.path.basename(test_path).replace(".py", "")
            print(f"\nRunning test: {test_path}")

            # add project root to pythonpath
            env = os.environ.copy()
            pythonpath = [os.path.join(project_root, "scripts"), tests_dir]

            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = ":".join(pythonpath) + ":" + env["PYTHONPATH"]
            else:
                env["PYTHONPATH"] = ":".join(pythonpath)

            # Check if this test needs a virtual display
            if test_name in XVFB_TEST_CASES:  # virtual display for GUI tests
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
            # TODO: remove this as integration tests
            elif "test_sim_with_dds" in test_path or "test_pi0" in test_path:
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

            process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

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

            result = process
            if result.returncode != 0:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all tests for a workflow")
    parser.add_argument("--workflow", type=str, default="robotic_ultrasound", help="Workflow name")
    args = parser.parse_args()

    if args.workflow not in WORKFLOWS:
        raise ValueError(f"Invalid workflow name: {args.workflow}")

    exit_code = run_tests_with_coverage(args.workflow)
    sys.exit(exit_code)
