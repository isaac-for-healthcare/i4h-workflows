import pyrealsense2 as rs


def count_devices():
    # Create a context object to get access to connected devices
    context = rs.context()
    devices = context.query_devices()
    return len(devices)

def list_devices():
    # Create a context object to get access to connected devices
    context = rs.context()
    devices = context.query_devices()

    # Enumerate and list all connected devices
    if len(devices) == 0:
        print("No RealSense devices connected.")
    else:
        for i, device in enumerate(devices):
            print(f"Device {i}: {device}")

if __name__ == "__main__":
    list_devices()
