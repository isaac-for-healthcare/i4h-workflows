# Ultrasound Raytracing Simulation

This example implements a standalone ultrasound raytracing simulator that generates realistic ultrasound images
based on 3D meshes. The simulator uses Holoscan framework and DDS communication for realistic performance.

## Quick Start

Please ensure you have the [NVIDIA OptiX Raytracing with Python Bindings](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/blob/v0.3.0/ultrasound-raytracing/README.md) to setup the `raysim` module correctly.

```bash
python -m simulation.examples.ultrasound_raytracing
```

**Expected Behavior:**
- Terminal messages showing the progress of the simulation, e.g. `[info] Timing Simulation took 1.389568 ms`.


To visualize the ultrasound images, please check out the [Visualization Utility](../../utils/README.md), but it's not recommended to use it without [moving the probe with robotic arm](../../../README.md#policy--ultrasound-simulation), which receives `topic_ultrasound_info` from the scripts such as [sim_with_dds.py](../environments/sim_with_dds.py) or [Tele-op](../environments/teleoperation/teleop_se3_agent.py).


## Configuration

Optionally, the simulator supports customization through JSON configuration files. You need to create your configuration file following the structure below:

```json
{
    "probe_type": "curvilinear",
    "probe_params": {
        "num_elements": 256,
        "sector_angle": 73.0,
        "radius": 45.0,
        "frequency": 2.5,
        "elevational_height": 7.0,
        "num_el_samples": 1,
        "f_num": 1.0,
        "speed_of_sound": 1.54,
        "pulse_duration": 2.0
    },
    "sim_params": {
        "conv_psf": true,
        "buffer_size": 4096,
        "t_far": 180.0
    }
}
```

### Probe Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| num_elements | Number of elements in the ultrasound probe | 256 |
| sector_angle | Beam sector angle in degrees | 73.0 |
| radius | Radius of the ultrasound probe in mm | 45.0 |
| frequency | Ultrasound frequency in MHz | 2.5 |
| elevational_height | Height of the elevation plane in mm | 7.0 |
| num_el_samples | Number of samples in the elevation direction | 1 |
| f_num | F-number (unitless) | 1.0 |
| speed_of_sound | Speed of sound in mm/us | 1.54 |
| pulse_duration | Pulse duration in cycles | 2.0 |

### Simulation Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| conv_psf | Whether to use convolution point spread function | true |
| buffer_size | Size of the simulation buffer | 4096 |
| t_far | Maximum time/distance for ray tracing (in units relevant to the simulation) | 180.0 |

### Minimal Configuration Example

You only need to specify the parameters you want to change - any omitted parameters will use their default values:

```json
{
    "probe_type": "curvilinear",
    "probe_params": {
        "frequency": 3.5,
        "radius": 55.0
    },
    "sim_params": {
        "t_far": 200.0
    }
}
```

## Command Line Arguments

| Argument | Description | Default Value |
|----------|-------------|---------------|
| --domain_id | Domain ID for DDS communication | 0 |
| --height | Input image height | 224 |
| --width | Input image width | 224 |
| --topic_in | Topic name to consume probe position | topic_ultrasound_info |
| --topic_out | Topic name to publish generated ultrasound data | topic_ultrasound_data |
| --config | Path to custom JSON configuration file with probe parameters and simulation parameters | None |
| --period | Period of the simulation (in seconds) | 1/30.0 (30 Hz) |
| --probe_type | Type of ultrasound probe to use ("curvilinear") | "curvilinear" |
