import os
import subprocess
import sys


def install_dependencies(workflow_name: str = "robotic_ultrasound"):
    """Install project dependencies from requirements.txt"""
    try:
        # Install test dependencies
        apt_cmd = ["apt-get", "install", "-y", "xvfb", "x11-utils", "cmake", "build-essential"]
        # check if the user is root
        if os.geteuid() != 0:
            apt_cmd.insert(0, "sudo")
        subprocess.check_call(apt_cmd)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage", "parameterized"])

        # Install workflow dependencies
        if workflow_name == "robotic_ultrasound":
            dir = os.path.dirname(os.path.abspath(__file__))
            subprocess.check_call(["./env_setup_robot_us.sh"], cwd=dir)

        print("Dependencies installed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    install_dependencies()
