# What's New in Isaac for Healthcare Workflows v0.2.0 🎉🎉

- **GR00T N1 policy for the robotic ultrasound workflow**: Integration of NVIDIA's GR00T N1 foundation model into robotic ultrasound workflows with complete training pipeline for multimodal manipulation tasks
- **Cosmos-transfer1 as augmentation method for policy training**: Training-free guided generation method bridging the perceptual divide between simulated and real-world environments
- **Telesurgery workflow**: Cutting-edge solution for remote surgical procedures with real-time, high-fidelity surgical interactions across distances
- **Enhanced utility modules**: Apple Vision Pro teleoperation, Haply Inverse3 controller support, and runtime asset downloading capabilities

## GR00T N1 policy for the robotic ultrasound workflow

![GR00T N1 policy for the robotic ultrasound workflow](../source/robotic_us_workflow.jpg)

The robotic ultrasound workflow now integrates NVIDIA Isaac GR00T N1 models, enabling advanced multimodal manipulation capabilities for ultrasound scanning tasks. Key features include:

*   **Cross-Embodiment Foundation Model:** Multimodal input processing (language and images) for manipulation tasks across diverse environments and robot platforms.
*   **Complete Training Pipeline:** End-to-end workflow from data collection through model deployment, including robot trajectory and camera image capture during ultrasound scans.
*   **LeRobot Format Integration:** Automated conversion from HDF5 simulation data to LeRobot format with GR00T N1-specific feature mapping and modality configuration.
*   **Liver Scan State Machine:** Specialized data collection implementation that generates training episodes emulating clinical liver ultrasound scanning procedures.
*   **Fine-tuning & Deployment:** Comprehensive training configuration with checkpoint management and inference deployment capabilities.

Learn more in the [GR00T N1 Training README](../../workflows/robotic_ultrasound/scripts/training/gr00t_n1/README.md) and [Robotic Ultrasound Workflow README](../../workflows/robotic_ultrasound/README.md).

## Cosmos-transfer1 as augmentation method for policy training

![Cosmos-transfer1 as augmentation method for policy training](../source/cosmos_transfer_result.png)

The Cosmos-Transfer1 integration provides a world-to-world transfer model designed to bridge the perceptual divide between simulated and real-world environments. Key features include:

*   **Training-free Guided Generation:** Novel method that overcomes unsatisfactory results on unseen healthcare simulation assets by preserving the appearance of phantoms and robotic arms while generating diverse backgrounds.
*   **Multi-view Video Generation:** Supports generating videos from multiple camera perspectives, leveraging camera information to warp room views to wrist views for comprehensive scene understanding.
*   **Controllable Realism-Faithfulness Trade-off:** Adjustable number of guided denoising steps allows fine-tuning the balance between realistic appearance and faithful preservation of original simulation assets.
*   **Spatial Masking Guidance:** Advanced technique that encodes simulation videos into latent space and applies spatial masking to guide the generation process for superior results.

Learn more in the [Cosmos-transfer1 README](../../workflows/robotic_ultrasound/scripts/simulation/environments/cosmos_transfer1/README.md).

## Telesurgery workflow

![Telesurgery Workflow](../source/telesurgery_workflow.jpg)

The Telesurgery Workflow is a cutting-edge solution designed for healthcare professionals and researchers working in remote surgical procedures. This comprehensive framework leverages NVIDIA's advanced GPU capabilities to enable real-time, high-fidelity surgical interactions across distances. Key features include:

*   **Real-World & Simulation Support:** Complete workflow supporting both physical MIRA robots from Virtual Incision and Isaac Sim-based simulation environments for development and testing.
*   **Low-Latency Communication:** Dual communication system using WebSockets for robot control commands and DDS for real-time video streaming with NVIDIA Video Codec encoding.
*   **Multi-Controller Support:** Flexible input options including Xbox controllers for immediate use and Haply Inverse3 devices for advanced, intuitive surgical control.
*   **Cross-Distance Operations:** Enables surgeons to perform complex procedures remotely, improving healthcare accessibility and reducing geographical barriers to specialized care.
*   **Advanced Video Streaming:** Configurable H.264/HEVC encoding with NVIDIA Video Codec and NVJPEG support for optimal video quality and minimal latency.

Learn more in the [Telesurgery Workflow README](../../workflows/telesurgery/README.md).

## Enhanced utility modules

These modules offer new ways to interact with simulations, manage assets efficiently, and support advanced control methods. Key enhancements include:

*   **Apple Vision Pro Teleoperation:** Advanced spatial computing integration enabling intuitive teleoperation for imitation learning workflows with natural hand tracking and gesture recognition.
*   **Haply Inverse3 Controller Support:** Professional-grade haptic device integration providing precise force feedback control for both telesurgery and imitation learning applications.
*   **Runtime Asset Downloading:** Streamlined asset management system allowing users to download workflow-specific assets on-demand, eliminating the need to download all assets upfront and reducing initial setup time.
