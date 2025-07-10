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

Control the virtual robot using keyboard keys:

```bash
cd i4h-workflows/workflows/soarm101_sim_to_real/scripts/simulation
python scripts/teleoperation.py --num_envs 1
```

**Keyboard Controls:**
- I/K: shoulder_pan (base rotation)
- J/L: shoulder_lift (shoulder elevation)  
- U/O: elbow_flex (elbow flexion)
- Z/X: wrist_flex (wrist flexion)
- C/V: wrist_roll (wrist rotation)
- B/N: gripper (open/close)
- SPACE: Reset to default position
- ESC: Exit

### Real Robot Teleoperation

Use a physical SO-ARM 101 robot to control the virtual robot:

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

Move the real robot and watch the virtual robot follow in real-time!
