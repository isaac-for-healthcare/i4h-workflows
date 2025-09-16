# Sim2Real with Mixed Training Data

<img src="../../docs/source/so_arm_starter_sim2real.gif" alt="Sim2Real with Mixed Training Data" width="1000" style="max-width: 100%; height: auto;">

## Overview

**Sim2Real**: Simulation-to-Real transfer is a technique where models trained in simulated environments are adapted to work in real-world conditions, bridging the gap between virtual training and physical deployment.

**The Problem**: Training robots in the real world is expensive and limited:
- Each real-world trial has costs: materials, time, wear-and-tear, safety risks
- Real-world data collection is expensive, slow, and limited in quantity and diversity
- Machine learning approaches typically require large-scale real-world data collection to achieve robust performance across diverse environments

**The Solution**: Combine simulation and real-world data strategically:
- Use simulation for high-volume, diverse training scenarios
- Use minimal real-world data for authenticity and grounding
- Train on mixed datasets to achieve robust performance

## Key Benefits

- **Reduced Real-World Data Collection**: Achieve high-quality policies with minimal real-world episodes
- **Access to Rare Scenarios**: Train on diverse, expensive, or hard-to-replicate real-world situations through simulation
- **Accelerated Development**: Rapidly iterate and test policies in simulation before real-world deployment

## Workflow

- [**Step 1: Simulation Data Collection**](./README.md#-phase-1-data-collection)
  Collect large-scale training data in IsaacLab environment.

- [**Step 2: Real Data Collection**](./README.md#real-world-data-collection)
  Collect real-world data using SO-ARM101 hardware.

- [**Step 3: Mixed Dataset Training**](./README.md#-phase-2-model-training)
  Train GR00T N1.5 model on combined sim + real dataset.

- [**Step 4: Real-World Deployment**](./README.md#-phase-3-gr00t-n15-deployment)
  Deploy trained model directly on SO-ARM101 hardware.

For detailed setup instructions, see [Installation](./README.md#installation) and [Hardware Requirements](./README.md#hardware-requirements).

## Quick Start

Follow the [Running Workflows](./README.md#-running-workflows) guide for complete instructions.

**Summary**: Collect sim data → Collect real data → Train on mixed dataset → Deploy

## Why Sim2Real Works

**Cost and Efficiency Benefits**:
- Limited access to real hardware, operators and environments makes data collection challenging
- Sim2Real achieves comparable or better quality with limited real-world episodes

**Robust Performance**: Mixed training exposes the model to both simulated variety and real-world authenticity, creating policies that generalize beyond either domain alone and achieve superior real-world performance.

For troubleshooting, see [Troubleshooting](./README.md#️-troubleshooting).

## Training Overview

| Approach | Sim Data | Real Data | Training | Benefits |
|----------|----------|-----------|----------|----------|
| **Mixed Training** | ~70 episodes | ~10-20 episodes | Combined | More robust |
