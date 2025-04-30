Changelog
=========

[0.1.0] – 2025-04-30
--------------------

🚀 Initial Release
~~~~~~~~~~~~~~~~~

Isaac for Healthcare (I4H) Workflows is now available, providing a modular and extensible foundation for building and testing AI-enabled healthcare robotics applications.

Added
~~~~~

Overview
--------
- Initial release of Isaac for Healthcare Workflows
- Support for digital prototyping of healthcare robotic systems
- Simulated Data Collection capabilities for AI model training
- Hardware-in-the-loop (HIL) testing framework with NVIDIA Holoscan
- GPU-accelerated training for reinforcement and imitation learning

Robotic Surgery
---------------
- Digital twin capabilities for surgical robotics
- State Machine and Data Collection system
- Simulation tasks including reaching, dual-arm reaching, suture needle lift, and STAR arm reaching
- Reinforcement learning policy training and evaluation capabilities
- Operating Room (OR) scene setup with surgical tools and robots

Robotic Ultrasound Workflow
---------------------------
- Digital twin capabilities for ultrasound robotics
- DDS-based communication system
- Integration with Clarius Cast and Clarius Solum Ultrasound Apps
- Integration with Ultrasound raytracing simulator
- Camera RGB and depth data support and visualization tools
- Simulation of policy-driven ultrasound probe movements, state-machine driven data collection and teleoperation

Tutorials and Documentation
---------------------------
- Installation and environment setup guides
- Bring Your Own Robot (BYOR) tutorial with example CAD files
- Bring Your Own Patient (BYOP) tutorial
- Comprehensive workflow documentation
- Sim2Real deployment guide

Development Tools
-----------------
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code formatting
- Unit test framework with coverage reporting
- License header checking
- Code style enforcement (isort, ruff)
- Asset management system

Technical Details
~~~~~~~~~~~~~~~~~

- Compatible with IsaacSim 4.5.0 and IsaacLab 2.0.2
- Python 3.10 support
- Linux platform support
- Apache 2.0 License
