# Holoscan Apps
This folder contains Holoscan applications for robotic ultrasound.

## Building the applications

Run the following command to build the Holoscan SDK applications.

```
cmake -B build -S .
cmake --build build
```

## Running the applications

After building the applications, you can run them with the instructions below.

### RealSense Camera
```
python3 holoscan_apps/realsense/camera.py
```

### Clarius Cast
```
phthon3 holoscan_apps/clarius_cast/clarius_cast.py
```

### Clarius Solumn
```
python3 holoscan_apps/clarius_solumn/clarius_solumn.py
```
