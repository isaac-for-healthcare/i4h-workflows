# Changelog

All notable changes to Isaac for Healthcare Workflows are documented in this file.

## [0.5.0] - New Rheo workflow, I4H CLI, and StreamLift

- **Rheo Workflow**: New end-to-end workflow for smart hospital automation and Physical AI development, featuring digital twin composition, expert demonstration capture, synthetic data generation, GR00T policy training with RL post-training, and pre-deployment validation.
- **I4H CLI**: Unified command-line interface across Robotic Surgery, Robotic Ultrasound, and SO-ARM Starter workflows, streamlining Docker builds, asset downloads, and workflow execution.
- **StreamLift for Telesurgery**: GPU-accelerated 4K image upsampling and downsampling operators for low-latency, high-resolution video streaming in telesurgery pipelines.
- **Repository Restructure**: Tutorials moved to a separate repository; improved layout, consolidated linting, and updated asset paths.

### Rheo Workflow

New comprehensive workflow for autonomous clinical environment development, built on NVIDIA Isaac Lab and Isaac Lab Arena.

- **Digital Twin Composition:** Rapid environment assembly using Isaac Lab-Arena for OR-scale task composition and Isaac Lab for task-centric, manager-based environments with curriculum design and large-scale RL.
- **Expert Demonstration Capture:** Teleoperation via Meta Quest Controls for loco-manipulation tasks (surgical tray pick-and-place, case cart pushing) and precision bimanual manipulation (trocar assembly). Keyboard teleoperation is also supported for loco-manipulation tasks.
- **Synthetic Data Generation:** Simulation-driven data amplification with Isaac Lab Mimic/SkillGen-style pipelines, combined with Cosmos Transfer 2.5 guided generation for cross-scene generalization.
- **Policy Training:** Supervised fine-tuning of GR00T N1.5/N1.6 VLA models on curated datasets, with online RL post-training (PPO via RLinf) for precision manipulation tasks such as multi-step trocar assembly.
- **Pre-Deployment Validation:** Closed-loop policy evaluation runners with WebRTC camera streaming and trigger-based action execution for system-level verification.
- **VLM Agents:** Configurable VLM-powered agents for peri-operative annotation, surgical monitoring, robot control, and user command handling, with automated setup scripts.
- **TensorRT Support:** GR00T N1.6 TensorRT acceleration for Arena-based tasks.

See [Rheo Workflow README](workflows/rheo/README.md).

### Isaac for Healthcare Command Line Interface (I4H CLI)

Unified `./i4h` command-line interface to simplify setup and execution across workflows. Workflows using I4H CLI now favor containerized development rather than setting up Conda environments on the host system.

- **Robotic Surgery:** CLI support for Docker build, asset download, and simulation launch.
- **Robotic Ultrasound:** CLI support for state machine, teleoperation, and evaluation modes with camera runtime configuration.
- **SO-ARM Starter:** Full CLI integration for simulation, teleoperation recording, policy training, and real-world deployment on DGX Spark, Jetson Orin, and Jetson Thor; simplified HDF5 recording path arguments; non-root simulation execution.
- **Performance:** Faster asset downloads by excluding blob data from CLI download steps.

### StreamLift for Telesurgery

- **4K UpSampling/DownSampling:** New GPU-accelerated Holoscan operators (C++ with Python bindings) for real-time 4K image upsampling and downsampling in telesurgery video pipelines.
- **DGX Spark Support:** Added workflow container support for DGX Spark platform.
- **IGX Orin (CUDA 12):** Real-world telesurgery workflow supported on IGX Orin.

See [Telesurgery Workflow README](workflows/telesurgery/README.md).

### Other Workflow Updates

- **Robotic Ultrasound:** Unified container environment for GR00T N1 and Pi0; re-enabled raysim; removed Cosmos Transfer 1 (placeholder for Cosmos Transfer 2.5); improved documentation and Quick Start guide with `i4h` CLI commands.
- **SO-ARM Starter:** Optimized x86_64 Dockerfile; added DGX Spark Isaac Sim container support and optimized DGX Dockerfile; aligned Jetson Thor environment to DGX; fixed Jetson Orin Dockerfile; improved documentation and Quick Start guide.
- **Robotic Surgery:** Updated README to streamline demo experience with `i4h` CLI commands.
- **Repository:** Improved layout and directory structure; added markdown linting; merged linting configs to root; updated IsaacSim 5.1 and IsaacLab compatibility fixes.

## [0.4.0] - Workflow updates

- **SO-ARM Starter Expansions**: Added DGX platform and Jetson Thor/Orin support, plus Holoscan integration for real-time streaming.
- **Workflow Updates**: Updates for IsaacSim 5.x and IsaacLab 2.2/2.3 across ultrasound, telesurgery, and surgery workflows, plus migration to Python 3.11 across all workflows.

### SO-ARM Starter Expansions

- **Jetson Orin and Thor Support:** Deploy to edge with Jetson Orin and Thor for on-device inference.
- **DGX Support:** Simulation and deployment on DGX Spark (IsaacSim 5.1) for accelerated development.
- **Holoscan Integration:** Enable low-latency streaming and processing in the SO-ARM workflow.
- **Documentation Enhancements:** Expanded SO-ARM Starter docs and guidance.

See [SO-ARM Starter Workflow README](workflows/so_arm_starter/README.md).

### Workflow Updates

All workflows now support IsaacSim 5.x and IsaacLab 2.2/2.3 with Python 3.11.

- **Robotic Ultrasound Workflow:** Consolidated on IsaacSim 5.0 and IsaacLab 2.3; updated SE(3) teleoperation for latest IsaacLab API changes; improved documentation for Cosmos-Transfer1; pip-based installation of the ultrasound raytracing package to avoid manual CMake steps.
- **Telesurgery Workflow:** Consolidated on IsaacSim 5.0 and IsaacLab 2.3.
- **Robotic Surgery Workflow:** Consolidated on IsaacSim 5.0 and IsaacLab 2.3.

---

## [0.3.0]

- **SO-ARM Starter Workflow**: Complete end-to-end pipeline for autonomous surgical assistance using SO-ARM101 robotic platform with GR00T N1.5 foundation model integration.
- **HSB and AJA Support for Telesurgery Workflow**: Professional-grade camera support for ultra-low latency video streaming.
- **New Tutorials**: Bring Your Own Operating Room, Cosmos-Transfer1 domain randomization, Medical Data Conversion (CT-to-USD), and Telesurgery Latency Benchmarking.

### SO-ARM Starter Workflow

- **Complete End-to-End Pipeline:** Three-phase workflow covering data collection, GR00T N1.5 model training, and policy deployment for surgical assistance tasks with comprehensive simulation and real-world support.
- **SO-ARM101 Hardware Integration:** Full support for SO-ARM101 leader and follower arms with integrated dual-camera vision system.
- **Multi-Modal Data Collection:** Flexible data collection supporting both simulation-based teleoperation and real-world hardware recording.
- **Sim2Real Mixed Training:** Strategic combination of simulation and real-world data for robust performance.
- **GR00T N1.5 Foundation Model:** Advanced foundation model training and fine-tuning with automated HDF5 to LeRobot format conversion and TensorRT optimization.
- **DDS Communication Framework:** Real-time communication with RTI DDS support.

See [SO-ARM Starter Workflow README](workflows/so_arm_starter/README.md).

### Enhanced Camera Support for Telesurgery Workflow

- **IMX274 Camera with HSB Integration:** High-resolution CMOS sensor supporting 4K and 1080p at 60fps with Holoscan Sensor Bridge and RDMA support.
- **AJA Professional Video Capture:** Broadcast-quality video capture with configurable channel selection and optional RDMA support.
- **YUAN-HSB HDMI Source Support:** HDMI input capture for professional medical imaging devices with 3D-to-2D format conversion and HSB-accelerated processing.

### New Tutorials

- Bring Your Own Operating Room
- Cosmos-Transfer1 Domain Randomization
- Medical Data Conversion (CT-to-USD)
- Telesurgery Latency Benchmarking

---

## [0.2.0]

- **GR00T N1 Policy for the Robotic Ultrasound Workflow**: Integration of NVIDIA's GR00T N1 foundation model with complete training pipeline for multimodal manipulation tasks.
- **Cosmos-Transfer1 as Augmentation Method for Policy Training**: Training-free guided generation bridging simulated and real-world environments.
- **Telesurgery Workflow**: Remote surgical procedures with real-time, high-fidelity interactions.
- **Enhanced Utility Modules**: Apple Vision Pro teleoperation, Haply Inverse3 controller support, and runtime asset downloading.

### GR00T N1 Policy for the Robotic Ultrasound Workflow

- **Complete Training Pipeline:** End-to-end workflow from data collection through trained model inference deployment.
- **LeRobot Format Support:** Automated conversion from HDF5 simulation data to LeRobot format with GR00T N1-specific feature mapping.
- **Liver Scan State Machine with Replay:** Enhanced state machine with replay functionality for HDF5 trajectories.
- **Inference Deployment:** Policy evaluation for trained models in robotic ultrasound simulation.

See [GR00T N1 Training README](workflows/robotic_ultrasound/scripts/training/gr00t_n1/README.md) and [Robotic Ultrasound Workflow README](workflows/robotic_ultrasound/README.md).

### Cosmos-Transfer1

- **Training-free Guided Generation:** Preserves appearance of phantoms and robotic arms while generating diverse backgrounds.
- **Multi-view Video Generation:** Multiple camera perspectives with room-to-wrist view warping.
- **Controllable Realism-Faithfulness Trade-off:** Adjustable guided denoising steps.
- **Spatial Masking Guidance:** Latent-space encoding and spatial masking for generation.

See [Cosmos-transfer1 README](https://github.com/isaac-for-healthcare/i4h-workflows/blob/v0.2.0/workflows/robotic_ultrasound/scripts/simulation/environments/cosmos_transfer1/README.md).

### Telesurgery Workflow

- **Real-World & Simulation Support:** Physical MIRA robots and Isaac Sim-based simulation.
- **Low-Latency Communication:** WebSockets for robot control, DDS for real-time video with NVIDIA Video Codec.
- **Multi-Controller Support:** Xbox controllers and Haply Inverse3 devices.
- **Advanced Video Streaming:** Configurable H.264/HEVC encoding with NVIDIA Video Codec and NVJPEG.

See [Telesurgery Workflow README](workflows/telesurgery/README.md).

### Enhanced Utility Modules

- **Apple Vision Pro Teleoperation:** Spatial computing integration with hand tracking and gesture recognition.
- **Haply Inverse3 Controller Support:** Haptic device integration for telesurgery and imitation learning.
- **Runtime Asset Downloading:** On-demand workflow-specific asset downloads.

---

## [0.1.0]

Initial release of Isaac for Healthcare Workflows.

- **Robotic Ultrasound Workflow**: Simulation environment for robotic ultrasound procedures with teleoperation, state machines, and realistic ultrasound imaging.
- **Robotic Surgery Workflow**: Tools and examples for simulating surgical robot tasks with state machines and reinforcement learning.
- **Tutorials**: Step-by-step guides for customizing simulation environments.

### Robotic Ultrasound Workflow

- **Policy Evaluation & Runner:** Examples for running pre-trained policies in simulation.
- **State Machine Examples:** Structured task execution (e.g. liver scan state machine with data collection).
- **Teleoperation:** Keyboard, SpaceMouse, or gamepad control of the robotic arm and ultrasound probe.
- **Ultrasound Raytracing:** Standalone ultrasound raytracing simulator for realistic images from 3D meshes.
- **DDS Communication:** RTI Connext DDS for inter-process communication.

See [Robotic Ultrasound Workflow README](workflows/robotic_ultrasound/README.md).

### Robotic Surgery Workflow

- **State Machine Implementations:** State-based control examples for surgical procedures.
- **Reinforcement Learning:** Framework for training RL policies for surgical subtasks.

See [Robotic Surgery Workflow README](workflows/robotic_surgery/README.md).

### Tutorials

- Bring Your Own Patient: Import custom CT or MRI scans into USD for simulation.
- Bring Your Own Robot: Import custom robot models (CAD/URDF) and replace components.
- [Sim2Real Transition](workflows/robotic_ultrasound/docs/sim2real/README.md): Adapt simulation-trained policies for physical deployment using DDS.
