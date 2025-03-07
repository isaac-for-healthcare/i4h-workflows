import os
import subprocess
import sys
import traceback

PROJECT_ROOTS = [
    "workflows/robotic_ultrasound",
]


def run_tests_with_coverage(project_root):
    """Run all unittest cases with coverage reporting"""
    try:
        # TODO: add license file to secrets
        os.environ["RTI_LICENSE_FILE"] = os.path.join(os.getcwd(), project_root, "scripts/dds/rti_license.dat")
        all_tests_passed = True
        tests_dir = os.path.join(project_root, "tests")

        print(f"Looking for tests in {tests_dir}")
        for name in os.listdir(tests_dir):
            test_dir = os.path.join(tests_dir, name)
            if os.path.isdir(test_dir):
                for test_file in os.listdir(test_dir):
                    if test_file.startswith("test_") and test_file.endswith(".py"):
                        test_path = os.path.join(test_dir, test_file)
                        print(f"\nRunning test: {test_path}")

                        # add project root to pythonpath
                        env = os.environ.copy()
                        pythonpath = [os.path.join(project_root, "scripts"), tests_dir]

                        if "PYTHONPATH" in env:
                            env["PYTHONPATH"] = ":".join(pythonpath) + ":" + env["PYTHONPATH"]
                        else:
                            env["PYTHONPATH"] = ":".join(pythonpath)

                        if "test_visualization" in test_path:  # virtual display for GUI tests
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
                            pass
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

                        process = subprocess.Popen(
                            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                        )
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
    exit_code = 0
    for project_root in PROJECT_ROOTS:
        result = run_tests_with_coverage(project_root)
        if result != 0:
            exit_code = result
    sys.exit(exit_code)
