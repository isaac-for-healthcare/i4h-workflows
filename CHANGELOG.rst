Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.1.0] - 2024-04-30
--------------------

Added
~~~~~

Core Framework
-------------
- Initial release of Isaac for Healthcare Workflows
- Support for digital prototyping of healthcare robotic systems
- Synthetic data generation (SDG) capabilities for AI model training
- Hardware-in-the-loop (HIL) testing framework
- GPU-accelerated training for reinforcement and imitation learning
- Sim2Real deployment capabilities for Nvidia Holoscan

Robotic Surgery Workflow
-----------------------
- Digital twin capabilities for surgical robotics
- Integration with ORBIT-Surgical framework
- State Machine and Data Collection system
- Policy training and evaluation capabilities
- DDS-based communication system
- Teleoperation support with haptics
- Operating Room (OR) environment setup

Robotic Ultrasound Workflow
--------------------------
- Digital twin capabilities for ultrasound robotics
- Integration with Clarius Cast and Clarius Solum Ultrasound Apps
- Ultrasound raytracing simulator
- Depth data support
- Visualization and annotation tools
- Rate-limited callback system
- DDS-based communication system

Tutorials and Documentation
-------------------------
- Bring Your Own Robot (BYOR) tutorial with example CAD files
- Bring Your Own Patient (BYOP) tutorial
- Installation and environment setup guides
- Comprehensive workflow documentation
- Sim2Real deployment guide
- API documentation and examples

Development Tools
---------------
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code formatting
- Unit test framework with coverage reporting
- License header checking
- Code style enforcement (isort, ruff)
- Asset management system

Technical Details
~~~~~~~~~~~~~~~~

- Compatible with IsaacSim 4.5.0
- Python 3.10 support
- Linux platform support
- Apache 2.0 License