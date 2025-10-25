# What's New in Isaac for Healthcare Workflows v0.4.0 🎉🎉

- **SO-ARM Starter Expansions**: Added DGX platform and Jetson Thor/Orin support, plus Holoscan integration for real-time streaming.
- **Workflow Updates**: Updates for IsaacSim 5.x and IsaacLab 2.2/2.3 across ultrasound, telesurgery, and surgery workflows, plus migration to Python 3.11 across all workflows.

## SO-ARM Starter Expansions

![SO-ARM Starter Workflow](../source/so_arm_starter_workflow.jpg)

Major expansions improve deployment flexibility and performance.

*   **Jetson Orin and Thor Support:** Deploy to edge with Jetson Orin and Thor for on-device inference.
*   **DGX Support:** Simulation and deployment on DGX Spark (IsaacSim 5.1) for accelerated development.
*   **Holoscan Integration:** Enable low-latency streaming and processing in the SO-ARM workflow.
*   **Documentation Enhancements:** Expanded SO-ARM Starter docs and guidance.

Learn more in the [SO-ARM Starter Workflow README](../../workflows/so_arm_starter/README.md).

## Workflow Updates

All workflows now support IsaacSim 5.x and IsaacLab 2.2/2.3 with Python 3.11.

*   **Robotic Ultrasound Workflow:** Consolidated on IsaacSim 5.0 and IsaacLab 2.3; updated SE(3) teleoperation for latest IsaacLab API changes; improved documentation for Cosmos-Transfer1; pip-based installation of the ultrasound raytracing package to avoid manual CMake steps.
*   **Telesurgery Workflow:** Consolidated on IsaacSim 5.0 and IsaacLab 2.3.
*   **Robotic Surgery Workflow:** Consolidated on IsaacSim 5.0 and IsaacLab 2.3.
