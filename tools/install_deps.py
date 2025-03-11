import subprocess
import sys


def install_dependencies():
    """Install project dependencies from requirements.txt"""
    try:
        subprocess.check_call(["nvidia-smi"])
        # Install dependencies
        subprocess.check_call(["sudo", "apt-get", "install", "-y", "xvfb", "x11-utils", "cmake", "build-essential"])
        # Install specific version of gcc for conda env
        subprocess.check_call(["conda", "install", "-c", "conda-forge", "gcc=13.3.0"])
        # Test dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage", "parameterized", "dearpygui"])
        # Install IsaacSim
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "rti.connext",
                "isaacsim==4.2.0.2",
                "isaacsim-extscache-physics==4.2.0.2",
                "isaacsim-extscache-kit==4.2.0.2",
                "isaacsim-extscache-kit-sdk==4.2.0.2",
                "pyrealsense",
                "holoscan==2.9.0",
                "--extra-index-url",
                "https://pypi.nvidia.com",
            ]
        )
        # Install IsaacLab
        subprocess.check_call(["git", "clone", "-b", "v1.4.1", "git@github.com:isaac-sim/IsaacLab.git"])
        subprocess.check_call(
            ["sed", "-i", "s/rsl-rl/rsl-rl-lib/g", "IsaacLab/source/extensions/omni.isaac.lab_tasks/setup.py"]
        )
        subprocess.check_call(["./isaaclab.sh", "--install"], cwd="./IsaacLab")
        # Install OpenPI
        subprocess.check_call(["./tools/install_openpi_with_isaac_4.2.sh"])
        print("Dependencies installed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    install_dependencies()
