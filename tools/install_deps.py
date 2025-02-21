import subprocess
import sys

def install_dependencies():
    """Install project dependencies from requirements.txt"""
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "coverage"])
        print("Dependencies installed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_dependencies()
