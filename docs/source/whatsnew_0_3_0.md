# What's New in Isaac for Healthcare Workflows v0.3.0 🎉🎉

- **SO-ARM Starter Workflow**: Complete end-to-end pipeline for autonomous surgical assistance using SO-ARM101 robotic platform with GR00T N1.5 foundation model integration
- **HSB and AJA Support for Telesurgery Workflow**: Training-free guided generation method bridging the perceptual divide between simulated and real-world environments
- **New Tutorials**: Comprehensive tutorial collection including Bring Your Own Operating Room, Cosmos-Transfer1 domain randomization, Medical Data Conversion (CT-to-USD), and Telesurgery Latency Benchmarking


## SO-ARM Starter Workflow

![SO-ARM Starter Workflow](../source/so_arm_starter_workflow.jpg)

The SO-ARM Starter Workflow addresses the critical need for autonomous surgical assistance by developing intelligent robotic systems that can perform the essential duties of a surgical assistant. This workflow specifically targets the complex, multi-modal task of surgical instrument management using the SO-ARM101 robotic platform. Key features include:

*   **Complete End-to-End Pipeline:** Three-phase workflow covering data collection, GR00T N1.5 model training, and policy deployment for surgical assistance tasks with comprehensive simulation and real-world support.
*   **SO-ARM101 Hardware Integration:** Full support for SO-ARM101 leader and follower arms with integrated dual-camera vision system, providing precise 6-DOF manipulation capabilities for surgical instrument handling.
*   **Multi-Modal Data Collection:** Flexible data collection supporting both simulation-based teleoperation (with keyboard fallback) and real-world hardware recording with synchronized camera feeds and robot trajectories.
*   **Sim2Real Mixed Training:** Strategic combination of simulation and real-world data to achieve robust performance with minimal real-world episodes, reducing costs while maintaining high-quality policy performance.
*   **GR00T N1.5 Foundation Model:** Advanced foundation model training and fine-tuning capabilities with automated HDF5 to LeRobot format conversion and TensorRT optimization for real-time inference.
*   **DDS Communication Framework:** Real-time communication system enabling seamless integration between simulation environments, policy runners, and hardware deployment with RTI DDS support.

Learn more in the [SO-ARM Starter Workflow README](../../workflows/so_arm_starter/README.md).

## Enhanced Camera Support for Telesurgery Workflow

![Enhanced Camera Support](../source/telesurgery_camera_support.png)

The Telesurgery Workflow now features comprehensive professional-grade camera support, enabling ultra-low latency video streaming essential for real-time remote surgical procedures. These advanced camera integrations provide surgeons with high-fidelity visual feedback across various hardware configurations. Key features include:

*   **IMX274 Camera with HSB Integration:** High-resolution CMOS sensor supporting 4K (3840x2160) and 1080p at 60fps with Holoscan Sensor Bridge for ultra-low latency capture with RDMA support.
*   **AJA Professional Video Capture:** Industry-standard AJA hardware providing broadcast-quality video capture with configurable channel selection (NTV2_CHANNEL1), and optional RDMA support for reduced CPU overhead.
*   **YUAN-HSB HDMI Source Support:** HDMI input capture enabling integration with professional medical imaging devices, featuring built-in 3D-to-2D format conversion and HSB-accelerated processing for minimal latency in surgical video streaming.


## New Tutorials

Expand your capabilities with advanced simulation and development techniques:

*   [Bring Your Own Operating Room](../../tutorials/assets/bring_your_own_or/README.md): Comprehensive guidance for acquiring, creating, and integrating custom 3D models of operating room assets into Isaac Sim environments using asset acquisition methods, CAD-to-USD conversion, and photogrammetry techniques.
*   [Cosmos-Transfer1 Domain Randomization](../../tutorials/assets/cosmos_transfer1/README.md): Advanced techniques for generating photorealistic videos from simulation data using NVIDIA's Cosmos Transfer1 model to bridge the sim-to-real gap with multi-modal control systems and distributed inference.
*   [Medical Data Conversion (CT-to-USD)](../../tutorials/assets/CT_to_USD/README.md): Complete workflow for converting medical imaging data from clinical formats (NIfTI/DICOM) to USD format using NVIDIA's MAISI CT foundational model for synthetic data generation and IsaacSim integration.
*   [Telesurgery Latency Benchmarking](../../tutorials/benchmarking/telesurgery_latency_benchmarking.md): Precise methodologies for measuring end-to-end, photon-to-glass latency in telesurgery video pipelines using NVIDIA's LDAT with professional measurement setup and performance optimization.
