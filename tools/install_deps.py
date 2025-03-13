import argparse
import os
import subprocess
import sys


def install_dependencies(workflow_name: str = "robotic_ultrasound"):
    """Install project dependencies from requirements.txt"""
    if workflow_name not in ["robotic_ultrasound", "robotic_surgery"]:
        raise ValueError(f"Invalid workflow name: {workflow_name}")

    try:
        # Install test dependencies
        apt_cmd = ["apt-get", "install", "-y", "xvfb", "x11-utils", "cmake", "build-essential", "pybind11-dev"]
        # check if the user is root
        if os.geteuid() != 0:
            apt_cmd.insert(0, "sudo")
        subprocess.check_call(apt_cmd)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage", "parameterized"])

        # Install workflow dependencies
        dir = os.path.dirname(os.path.abspath(__file__))
        if workflow_name == "robotic_ultrasound":
            subprocess.check_call(["./env_setup_robot_us.sh"], cwd=dir)
            subprocess.check_call(["./build.sh"], cwd=dir)
        elif workflow_name == "robotic_surgery":
            subprocess.check_call(["./env_setup_robot_surgery.sh"], cwd=dir)

        print("Dependencies installed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install project dependencies")
    parser.add_argument("--workflow", type=str, default="robotic_ultrasound", help="Workflow name")
    args = parser.parse_args()
    install_dependencies(args.workflow)
