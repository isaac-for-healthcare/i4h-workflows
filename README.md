# Isaac for Healthcare - Workflows

<p align="center" style="display: flex; justify-content: center; gap: 10px;">
  <img src="./docs/source/surgery.png" alt="Robotic Surgery" style="width: 45%; height: auto; aspect-ratio: 16/9; object-fit: cover;" />
  <img src="./docs/source/ultrasound.jpg" alt="Robotic Ultrasound" style="width: 45%; height: auto; aspect-ratio: 16/9; object-fit: cover;" />
</p>

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/isaac-for-healthcare/i4h-workflows/tree/main/docs)
[![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](./CONTRIBUTING.md)

## Table of Contents
- [Overview](#overview)
- [Available Workflows](#available-workflows)
- [Tutorials](#tutorials)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Overview

This repository contains **healthcare robotics workflows** - complete, end-to-end implementations that demonstrate how to build, simulate, and deploy robotic systems for specific medical applications using the [Nvidia Isaac for Healthcare](https://github.com/isaac-for-healthcare) platform.

### What are Workflows?

![Key features](./docs/source/key_features.jpg)

Workflows are comprehensive reference implementations that showcase the complete development pipeline from simulation to real-world deployment. Each workflow includes digital twin environments, AI model training capabilities, and deployment frameworks for specific healthcare robotics applications.

### Available Workflows
This repository currently includes four main workflows:

- **[Robotic Surgery](./workflows/robotic_surgery/README.md)** - Physics-based surgical robot simulation framework with photorealistic rendering for developing autonomous surgical skills. Supports da Vinci Research Kit (dVRK), dual-arm configurations, and STAR surgical arms. This workflow enables researchers and medical device companies to train AI models for surgical assistance, validate robot behaviors safely, and accelerate development through GPU-parallelized reinforcement learning. Includes pre-built surgical subtasks like suture needle manipulation and precise reaching tasks.
- **[Robotic Ultrasound](./workflows/robotic_ultrasound/README.md)** - Comprehensive autonomous ultrasound imaging system featuring physics-accurate sensor simulation through GPU-accelerated raytracing technology that models ultrasound wave propagation, tissue interactions, and acoustic properties in real-time. The raytracing ultrasound simulator generates photorealistic B-mode images by simulating acoustic wave physics, enabling synthetic data generation for training AI models without requiring physical ultrasound hardware. Supports multiple AI policies (PI0, GR00T N1), distributed communication via RTI DDS, and Holoscan deployment for clinical applications. This workflow enables medical imaging researchers, ultrasound device manufacturers, and healthcare AI developers to train robotic scanning protocols, validate autonomous imaging algorithms, and accelerate development through GPU-accelerated simulation before clinical deployment.
- **[Telesurgery](./workflows/telesurgery/README.md)** - Real-time remote surgical operations framework supporting both simulated and physical environments with low-latency video streaming, haptic feedback, and distributed control systems. This workflow features H.264/HEVC hardware-accelerated video encoding, RTI DDS communication, cross-platform deployment (x86/AARCH64), and seamless sim-to-real transition. Designed for surgical robotics companies, medical device manufacturers, and telemedicine providers to develop remote surgical capabilities, validate teleoperation systems, and deploy scalable telesurgery solutions across different network conditions.
- **[SO-ARM Starter](./workflows/so_arm_starter/README.md)** - Surgical assistant robotics system featuring SO-ARM101 manipulator control with complete data collection, policy training, and deployment pipeline. Implements GR00T N1.5 diffusion policy for autonomous surgical instrument handling, precise tool positioning, and workspace organization using dual RGB camera streams (640x480@30fps room/wrist views) and 6-DOF joint state feedback. Features physics-based simulation environments, imitation learning from both real-world and simulated demonstrations, real-time policy inference generating 16-step action sequences, and RTI DDS middleware for distributed robot communication.

Each workflow provides complete simulation environments, training datasets, pre-trained models, and deployment tools to accelerate your healthcare robotics development.

Please see [What's New](./docs/source/whatsnew_0_3_0.md) for details on our milestone releases.

### Tutorials

Get started with our comprehensive tutorials that guide you through key aspects of the framework:

#### Tutorials
- [Bring Your Own Patient](./tutorials/assets/bring_your_own_patient/README.md)
- [Medical Data Conversion (CT-to-USD)](./tutorials/assets/CT_to_USD/README.md)
- [Bring Your Own Robot](./tutorials/assets/bring_your_own_robot/)
- [Bring Your Own Operating Room](./tutorials/assets/bring_your_own_or/README.md)
- [Bring Your Own XR Device](./tutorials/assets/bring_your_own_xr/README.md)
- [Sim2Real Transition](./tutorials/sim2real/README.md)
- [Telesurgery Latency Benchmarking](./tutorials/benchmarking/telesurgery_latency_benchmarking.md)

## Repository Structure

```
i4h-workflows/
├── docs/                 # Documentation and guides
├── tutorials/            # Tutorial materials
│   ├── assets/             # Asset-related tutorials
│   └── sim2real/           # Sim2Real transition tutorials
├── workflows/            # Main workflow implementations
│   ├── so_arm_starter/     # SO-ARM Starter workflow
│   ├── robotic_surgery/    # Robotic surgery workflow
│   ├── robotic_ultrasound/ # Robotic ultrasound workflow
│   └── telesurgery/        # Telesurgery workflow
```

## Contributing

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone. Contributions can be made through:

- Bug reports
- Feature requests
- Code contributions
- Documentation improvements
- Tutorial additions

Please check our [contribution guidelines](./CONTRIBUTING.md) for detailed information on:
- Development setup
- Code style guidelines
- Pull request process
- Issue reporting
- Documentation standards

## Support

For support and troubleshooting:

1. Check the [documentation](https://github.com/isaac-for-healthcare/i4h-workflows/tree/main/docs)
2. Search existing [issues](https://github.com/isaac-for-healthcare/i4h-workflows/issues)
3. [Submit a new issue](https://github.com/isaac-for-healthcare/i4h-workflows/issues/new) for:
   - Bug reports
   - Feature requests
   - Documentation improvements
   - General questions

## License

The Isaac for Healthcare framework is under [Apache 2.0](./LICENSE).

## Acknowledgement

The [Robotic Surgery workflow](./workflows/robotic_surgery/) initiated from the [ORBIT-Surgical](https://orbit-surgical.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@inproceedings{ORBIT-Surgical,
  author={Yu, Qinxi and Moghani, Masoud and Dharmarajan, Karthik and Schorp, Vincent and Panitch, William Chung-Ho and Liu, Jingzhou and Hari, Kush and Huang, Huang and Mittal, Mayank and Goldberg, Ken and Garg, Animesh},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  title={ORBIT-Surgical: An Open-Simulation Framework for Learning Surgical Augmented Dexterity},
  year={2024},
  pages={15509-15516},
  doi={10.1109/ICRA57147.2024.10611637}
}

@inproceedings{SuFIA-BC,
  author={Moghani, Masoud and Nelson, Nigel and Ghanem, Mohamed and Diaz-Pinto, Andres and Hari, Kush and Azizian, Mahdi and Goldberg, Ken and Huver, Sean and Garg, Animesh},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  title={SuFIA-BC: Generating High Quality Demonstration Data for Visuomotor Policy Learning in Surgical Subtasks},
  year={2025},
}
```
