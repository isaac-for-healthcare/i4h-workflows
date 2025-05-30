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
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Overview

**[Nvidia Isaac for Healthcare](https://github.com/isaac-for-healthcare)** is a comprehensive 3-computer solution for healthcare robotics, enabling developers to leverage NVIDIA technologies for:

- Accelerating development and testing of AI algorithms with synthetic data generated in digital twin
- Accelerating development, training and testing of robotic policies in digital twin environments
- Enabling development, training and testing of systems with hardware-in-the-loop (HIL)
- Seamlessly transitioning from simulation to physical systems (Sim2Real)

## Key Features

![Key features](./docs/source/key_features.jpg)

Our framework provides powerful capabilities for healthcare robotics development:

### Digital Twin & Simulation
- **Digital prototyping** of next-gen healthcare robotic systems, sensors and instruments
- **Synthetic data generation (SDG)** for training AI models, augmented with real data
- **Hardware-in-the-loop (HIL)** testing and evaluation of AI models in digital twin environments

### AI & Robotics Development
- **Data collection** for training robotic policies through imitation learning
- **XR and haptics-enabled teleoperation** of robotic systems in digital twin
- **GPU-accelerated training** of reinforcement and imitation learning algorithms
- **Augmented dexterity training** for robot-assisted surgery

### Deployment & Training
- **Sim2Real deployment** on Nvidia Holoscan
- **Continuous testing (CT)** of robotic systems through HIL digital twin systems
- **Interactive training** experiences for clinicians/users (pre-op planning, post-op evaluations/insights)

Please see [What's New](./docs/source/whatsnew_0_1_0.md) for details on our milestone releases.

## Getting Started

For everything you need to get started, including detailed tutorials and step-by-step guides, follow these links to learn more about:

### Workflows
- [Robotic ultrasound](./workflows/robotic_ultrasound/README.md)
- [Robotic surgery](./workflows/robotic_surgery/README.md)
- [Telesurgery](./workflows/telesurgery/README.md)

### Tutorials
- [Bring your own patient](./tutorials/assets/bring_your_own_patient/README.md)
- [Bring your own robot](./tutorials/assets/bring_your_own_robot/README.md)
- [Sim2Real Transition](./tutorials/sim2real/README.md)

## Project Structure

```
i4h-workflows/
├── docs/                  # Documentation and guides
├── workflows/            # Main workflow implementations
│   ├── robotic_surgery/  # Robotic surgery workflow
│   └── robotic_ultrasound/ # Robotic ultrasound workflow
├── tutorials/            # Tutorial materials
│   ├── assets/          # Asset-related tutorials
│   └── sim2real/        # Sim2Real transition tutorials
├── tests/               # Test suite
└── examples/            # Example implementations
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
