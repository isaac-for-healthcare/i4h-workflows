# RealSense Camera

## Setup

Make sure the following pip packages are available:
```
pip install pyrealsense2
pip install holoscan
```

## Building Application

The `clarius_cast` and `clarius_solum` applications need to built using cmake.  This
will download the necessary header files and libraries from Clarius repo, and build the
python bindings in the case of the Clarius Solum app.

To build the applications run the following commands from this directory:

```
cmake -B build -S .
cmake --build build
```

## Running Application

To run realsense camera
```
python3 holoscan_apps/realsense/camera.py
```
