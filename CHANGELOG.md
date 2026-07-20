# Changelog

All notable changes to Isaac for Healthcare Workflows are documented in this file.

## [0.7.0] - Endoluminal Workflow, Agentic Surgical Environments, NVSkills Agent Platform, Rheo Deformable Cloth

- **Endoluminal Workflow**: New GPU fluoroscopy + XPBD catheter navigation workflow with CT digital-twin generation, Slang DRR/DSA rendering, interactive viewport, and `./i4h` CLI modes; user-facing name is **Endoluminal Workflow** (directory remains `workflows/catheter_navigation/`).
- **Agentic Surgical Environments**: Six new dVRK/STAR Arena environments ported from Robotic Surgery, plus scripted state-machine rollouts and a surgical baseline policy path for smoke validation.
- **NVSkills Agent Platform**: Skills relocated to top-level `skills/` with NV-BASE validation, signed artifacts, eval datasets, `AGENTS.md` entry point, and a new **Local Agent** runner.
- **Rheo Surface-Deformable Simulation**: New tablecloth-spreading task with Newton/PhysX backends, XR teleop recording, and dedicated cloth Docker image.

### Endoluminal Workflow

New `workflows/catheter_navigation/` workflow for simulation-driven development of endovascular navigation systems.

- **Fluoroscopy Simulator (`fluorosim`):** Slang-based differentiable DRR renderer with fused Beer-Lambert catheter compositing, batched multi-env rendering, DSA pipeline, and detector realism.
- **Physics Solver:** XPBD Cosserat rod solvers with vessel-mesh containment, track-guided insertion, and hardened bend containment.
- **Vasculature Digital Twin:** CT ingestion (DICOM/NIfTI), segmentation, meshing, and centerline extraction; bring your own CT patient data.
- **Interactive Viewport:** Multi-projection C-arm fluoroscopy with guided catheter navigation; fluoro and mesh-rail viewport paths removed in favor of guided-only runtime.
- **I4H CLI Integration:** Registered workflow with modes `preprocess_ct`, `segment_vessels`, `render_drr`, and `interactive_viewport`; host and Docker execution paths.
- **Pip-Installable Packages:** Runtime pulls `fluoro-simulator`, `vasculature-digital-twin`, and `catheter-vasculature-solver` from pinned Git dependencies.
- **Agent Skills:** Seven `i4h-catheter-navigation*` skills (overview, setup, digital twin, DRR render, viewport, smoke, e2e) with eval prompts and NVSkills signatures.

See [Endoluminal Workflow README](workflows/catheter_navigation/README.md).

### Surgical Agentic Workflow Updates

Expands the Agentic workflow from five pre-trained environments to eleven registered environments.

- **Six New Surgical Environments:** `surgical_reach_psm`, `surgical_reach_dual_psm`, `surgical_reach_star`, `surgical_lift_block`, `surgical_lift_needle`, and `surgical_lift_needle_organs`.
- **New Robot Embodiments:** dVRK PSM, dual PSM, STAR, and ECM configs under `config/robots/`.
- **Scripted State Machines:** New state-machine framework for policy-free rollouts with latched success semantics; supported on scissor, ultrasound, and all six surgical envs via `--state-machine`.
- **Surgical Baseline Policy:** Zero-action GR00T N1.5 inference daemon for validating policy/Arena/Zenoh plumbing on new surgical envs.
- **Policy Health Routing:** Readiness checks for skill and automation flows via `policy_routing.py`.
- **Scene-Edit Bridge:** Per-env bridge port from YAML (`arena.bridge_port`); resolve via `arena/run.sh bridge-url --env <env>`.
- **E2E Pipeline:** Headless camera rendering in record/replay/validate stages; vLLM annotator gains `wait`/`ensure` modes.

See [Agentic Workflow README](workflows/agentic/README.md) and [Arena State Machines README](workflows/agentic/arena/arena/statemachine/README.md).

### Rheo Deformable Cloth

First surface-deformable asset support in Rheo, alongside existing rigid-body tasks.

- **Spread Tablecloth Task:** G1 29DoF + Inspire hands and Fourier H2 + Sharpa Wave hands teleop environments.
- **Physics Backends:** Switchable `--physics_backend newton` (default) or `physx`.
- **XR Teleop Recording:** `record_demos_tablecloth.py` with CloudXR hand tracking.
- **Dedicated Container:** `Dockerfile.rheo_cloth` with separate IsaacLab checkout so cloth deps do not affect other Rheo tasks.
- **Newton Fixes:** Improved G1 Inspire hand finger tracking and hand–cloth collision behavior.

See [Rheo Workflow README](workflows/rheo/README.md).

### NVSkills Agent Platform

Major reorganization of the agent skill system introduced in v0.6.0.

- **Skills Relocated:** Canonical catalog moves from `.claude/skills/i4h-workflow-*` to top-level `skills/`; `.claude/skills` and `.codex/skills` are symlinks.
- **`AGENTS.md`:** New agent entry point with skill routing, supported env table, and `$I4H_WORKFLOWS` bootstrap.
- **NV-BASE Validation:** All skills include `BENCHMARK.md`, `skill-card.md`, `skill.oms.sig`, and `evals/evals.json`.
- **`TESTING.md`:** Validation guide for NV-BASE offline validation, eval schema checks, and acceptance criteria.
- **Local Agent (`local-agent/`):** Run skills with local SGLang or NVIDIA-hosted inference; includes `validate-env.sh`, `validate-bake.sh`, and `vlcheck.py`.
- **Removed Skill:** `i4h-workflow-dataset-transfer` (Cosmos Transfer augmentation) is no longer in the skill catalog.

### Other Updates

- **Documentation:** Updated Agentic, Arena, Rheo, and Telesurgery READMEs; fixed Holoscan Sensor Bridge doc links after docs.nvidia.com restructure.
- **Dependencies:** Agentic GR00T policy stacks switch flash-attn wheels to public URM URLs; catheter workflow container with pinned Python deps in `requirements.txt`.
- **I4H CLI:** Metadata discovery skips agentic virtualenvs to eliminate spurious `metadata.json` scan errors.
- **Fixes:** Catheter viewport input reliability and vessel containment at bends; agentic G1 import and GR00T N1.6 health metadata fixes; skill link and smoke-test alignment fixes.

### Breaking Changes and Renames

- **Skills Path:** Update references from `.claude/skills/i4h-workflow-*` to `skills/i4h-workflow-*`.
- **Removed Skill:** `i4h-workflow-dataset-transfer` no longer shipped; Cosmos Transfer must be invoked manually outside the skill catalog.
- **Workflow Display Name:** **Catheter Navigation** → **Endoluminal Workflow** in user-facing docs; CLI/HoloHub id remains `catheter_navigation`.
- **Catheter Viewport Modes:** Fluoro and mesh-rail paths removed; `interactive_viewport` is guided-only.
- **Scene-Edit Bridge Port:** No longer hardcoded to `8765`; read per-env from YAML via `bridge-url`.
- **Vendor Spelling:** **LightWheel** → **Lightwheel** in Rheo attribution.

---

## [0.6.0] - Agentic Workflow, Claude Code Skills, SO-ARM Starter on GR00T N1.7

- **Agentic Workflow**: New unified IsaacLab-Arena + GR00T/openpi pipeline with five pre-trained environments, covering teleop recording, mimic expansion, VLM annotation, LeRobot conversion, fine-tuning, and rollout validation from a single CLI.
- **Claude Code Agent Skills**: Composable `.claude/skills/i4h-workflow-*` skills that drive the agentic workflow from natural-language prompts, from env creation and scene editing through end-to-end smoke runs.
- **SO-ARM Starter on GR00T N1.7**: SO-ARM workflow upgraded to GR00T N1.7 with refreshed DGX Spark / Jetson Thor / Jetson Orin container images.

### Agentic Workflow

New `workflows/agentic/` workflow unifying IsaacLab-Arena, GR00T (N1.5/N1.6/N1.7), and openpi PI0 behind YAML-driven dispatch.

See [Agentic Workflow README](workflows/agentic/README.md).

### Claude Code Agent Skills

Composable skills under `.claude/skills/i4h-workflow-*` covering setup, env creation, scene editing, dataset capture/curation, finetuning, validation, end-to-end runs, and LeRobot visualization. Includes documented scene-edit bridge recipes (live teleport, USD edits, camera bake-to-YAML) and hybrid env recipes (e.g. G1 + scissor scene) with pre-resolved component choices.

### SO-ARM Starter

- **GR00T N1.7 Upgrade:** SO-ARM Starter workflow updated to GR00T N1.7.
- **Container Refresh:** DGX Spark, Jetson Thor, and Jetson Orin Dockerfiles upgraded.

---

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
