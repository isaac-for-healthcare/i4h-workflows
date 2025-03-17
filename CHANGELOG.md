# Isaac for Healthcare - Workflows v0.1 Release Notes

We're excited to announce the initial release of Isaac for Healthcare Workflows, a domain-specific framework built on top of NVIDIA Isaac Sim/Lab and NVIDIA Omniverse. This release provides foundational components for healthcare robotics simulation and development.

## Key Features

### Robotic Ultrasound Workflow
- Simulation environment for robotic ultrasound procedures
- Integration with Raytracing Ultrasound Simulator
- Policy runner for executing trained models
- Holoscan applications for real-time processing
- Training scripts for reinforcement learning
- Visualization utilities

### Robotic Surgery Workflow
- Simulation environment for surgical robotics
- Support for dVRK and STAR robot arm control
- Suture needle manipulation tasks
- Peg block manipulation tasks
- Reinforcement learning capabilities

## System Requirements
- Ubuntu 22.04
- NVIDIA GPU with ray tracing capability (GPUs without RT Cores like A100, H100 are not supported)
- NVIDIA Driver Version >= 555
- 50GB of disk space

## Dependencies
- IsaacSim 4.1.0.0/4.2.0.2
- IsaacLab 1.2.0/1.4.1
- Python 3.10
- Additional libraries as specified in the setup scripts

## Getting Started
Please refer to the workflow-specific README files for detailed setup instructions:
- [Robotic Ultrasound Workflow](./workflows/robotic_ultrasound/README.md)
- [Robotic Surgery Workflow](./workflows/robotic_surgery/README.md)

## Known Limitations
- This is an initial release focused on establishing core functionality
- Some features may require further refinement
- Documentation is still being developed

## Future Work
- Enhanced documentation and tutorials
- Additional healthcare robotics workflows
- Improved integration with physical hardware
- Expanded asset library

## License
The Isaac for Healthcare framework is under [Apache 2.0](./LICENSE). 