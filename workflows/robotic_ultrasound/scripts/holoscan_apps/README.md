# Holoscan Apps
This folder contains Holoscan applications for robotic ultrasound.

## Running Applications

Each application can be run using the i4h script. The script builds operators as needed and uses the docker environment to pull the dependencies.

Here's how to run each application:

### Clarius Cast
```bash
./i4h run clarius_cast
```

### Clarius Solum
```bash
./i4h run clarius_solum
```

### RealSense
```bash
./i4h run realsense
```

Note: Make sure you have the required dependencies installed and the appropriate hardware connected before running these applications. The applications may require specific configuration files or environment variables to be set up properly.
