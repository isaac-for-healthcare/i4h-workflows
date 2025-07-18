# SO-ARM 101 Simulation Package

This package provides keyboard and real robot teleoperation for SO-ARM 101 robot simulation.

## Setup

### 1. Set PYTHONPATH

```bash
export PYTHONPATH="/path/to/i4h-workflows/workflows/soarm101_sim_to_real/scripts:$PYTHONPATH"
```

### 2. For Real Robot Teleoperation

Find and set up serial port permissions:

```bash
# Find available ports
cd /path/to/lerobot
python lerobot/find_port.py

# Set permissions for leader arm (required)
sudo chmod 666 /dev/ttyACM0

# Set permissions for follower arm (optional)
sudo chmod 666 /dev/ttyACM1
```

## Usage

### Keyboard Teleoperation

Control the virtual SO-ARM 101 robot using keyboard keys. The robot is positioned on top of a Seattle Lab Table for realistic interaction.

```bash
cd i4h-workflows/workflows/soarm101_sim_to_real/scripts/simulation
python scripts/teleoperation.py --num_envs 1
```

**Keyboard Controls:**
- **I/K**: shoulder_pan (base rotation)
- **J/L**: shoulder_lift (shoulder elevation)  
- **U/O**: elbow_flex (elbow flexion)
- **Z/X**: wrist_flex (wrist flexion)
- **C/V**: wrist_roll (wrist rotation)
- **B/N**: gripper (open/close)
- **SPACE**: Reset to default position
- **ESC**: Exit the application

**Usage Tips:**
- Press and hold keys to move joints continuously
- Use step size parameter to control movement speed: `--step_size 0.02`
- The robot will respect joint limits based on real robot calibration data
- Joint status is displayed in the console during operation

**Command Line Options:**
```bash
python scripts/teleoperation.py --help
```
- `--num_envs`: Number of environments to spawn (default: 1)
- `--step_size`: Step size for joint movements (default: 0.02)
- `--device`: Device for simulation (default: auto-detected)

### Real Robot Teleoperation

Use a physical SO-ARM 101 robot to control the virtual robot in real-time.

**Step 1:** Start the hardware driver
```bash
cd i4h-workflows/workflows/soarm101_sim_to_real/scripts/simulation
python communication/host_soarm_driver.py --port 8888
```

**Step 2:** Start Isaac Lab simulation (in a new terminal)
```bash
cd i4h-workflows/workflows/soarm101_sim_to_real/scripts/simulation
python scripts/real_to_sim_teleoperation.py --port 8888
```
