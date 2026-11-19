# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment client: a Unitree H2 with SharpaWave hands driven by the served policy.

The client runs on the workstation next to the robot, in a Python 3.10
environment with ROS 2 and the Unitree DDS stack, which is why nothing here
imports ``i4h_common`` or the model. ``protocol`` speaks to the policy server,
``observation`` builds what the served checkpoint asks for, ``execution`` walks
action chunks with real-time chunking, ``robot`` adapts the teleop checkout, and
``deploy`` is the command line that wires them together.
"""
