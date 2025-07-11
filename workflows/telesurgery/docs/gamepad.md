# 🎮 MIRA Gamepad Control

A Python interface for controlling MIRA using a generic gamepad from [Virtual Incision](https://virtualincision.com/) team.

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows-lightgrey)

</div>

## ✨ Features

- 🎮 Xbox controller support
- 🤖 Dual-arm control in both Cartesian and Polar modes
- 🔄 Mode switching with lightring feedback
- 📷 Camera position control
- 🛠️ Tool roll, grasp, and homing

## 🚀 Quick Start

### Prerequisites

- Xbox controller or compatible gamepad


## 🎮 Controls

### Mode Indication (Light ring on physical MIRA robot)
- 🔵 Blue Light Ring: Cartesian Mode
- 🟢 Green Light Ring: Polar Mode
- 🔴 Red Light Ring: Controller Disconnected

### Basic Controls

| Control | Action |
|---------|--------|
| View Button | Switch between Cartesian/Polar modes |
| Start Button | Reset to default pose |
| D-Pad | Control camera position |
| X Button (West) | Toggle left gripper **(Only works after a successful tool homing)** |
| B Button (East) | Toggle right gripper **(Only works after a successful tool homing)** |
| Hold X+B (3s) | Home tools |

### Arm Movement

#### Cartesian Mode
| Control | Action |
|---------|--------|
| Left Stick | Left arm XY movement |
| Right Stick | Right arm XY movement |
| LT/LB | Left arm Z-axis |
| RT/RB | Right arm Z-axis |
| LT+LB + Left Stick-x | Left arm elbow angle |
| RT+RB + Right Stick-x | Right arm elbow angle |

#### Polar Mode
| Control | Action |
|---------|--------|
| Left Stick | Left arm Polar XY (Keeps sweep plane constant) |
| Right Stick | Right arm Polar XY (Keeps sweep plane constant) |
| LT/LB | Left arm sweep plane |
| RT/RB | Right arm sweep plane |
| LT+LB + Left Stick-x | Left arm elbow angle |
| RT+RB + Right Stick-x | Right arm elbow angle |

### Camera Control
| D-Pad Direction | Movement |
|-----------------|----------|
| Up/Down | North/South |
| Left/Right | East/West |
