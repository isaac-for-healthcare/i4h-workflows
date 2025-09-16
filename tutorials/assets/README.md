# Asset Tutorials

This directory contains comprehensive tutorials for customizing and extending Isaac for Healthcare workflows with your own assets, data, and environments.

## Bring Your Own Patient

- [Medical Data Conversion (CT-to-USD)](./CT_to_USD/README.md)
  Complete workflow for converting medical imaging data from clinical formats (NIfTI/DICOM) to USD format using NVIDIA's MAISI CT foundational model for synthetic data generation and IsaacSim integration.

- [Bring Your Own Patient](./bring_your_own_patient/README.md)
  Learn how to import and integrate custom CT or MRI scans into USD (Universal Scene Description) files for 3D visualization and simulation in Isaac Sim.

## Bring Your Own Robot

- [Virtual Incision MIRA Teleoperation](./bring_your_own_robot/Virtual_Incision_MIRA/README.md)
  Learn how to teleoperate the [Virtual Incision MIRA](https://virtualincision.com/mira/) robot in Isaac Sim using keyboard controls.

- [Replace Franka Hand with Ultrasound Probe](./bring_your_own_robot/replace_franka_hand_with_ultrasound_probe.md)
  Step-by-step guide to replacing the Franka robot's hand with an ultrasound probe in Isaac Sim, including CAD/URDF conversion, asset import, and joint setup for custom robotic ultrasound simulation.

## Bring Your Own Operating Room

- [Bring Your Own Operating Room](./bring_your_own_or/README.md)
  Comprehensive guidance for acquiring, creating, and integrating custom 3D models of operating room assets into Isaac Sim environments using asset acquisition methods, CAD-to-USD conversion, robot rigging, and photogrammetry techniques.

## Bring Your Own XR Device

- [Bring Your Own Head-Mounted Display with OpenXR](./bring_your_own_xr/README.md)
  Comprehensive guide for using OpenXR-enabled mixed reality devices (like Apple Vision Pro) for immersive robotic teleoperation in Isaac Lab. Learn to set up NVIDIA CloudXR runtime, configure hand tracking controls, and enable simulated robotic teleoperation in Isaac Lab.

## Cosmos-transfer for Domain Randomization

- [Cosmos-transfer for Domain Randomization](./cosmos_transfer1/README.md)
  Step-by-step guide to generate photorealistic videos from custom input videos using the Cosmos Transfer1 model. The Guided generation enables domain randomization by transferring simulation videos to realistic-looking footage while maintaining structural consistency through various control inputs.
