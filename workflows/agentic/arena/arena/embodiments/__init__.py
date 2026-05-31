# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-robot embodiment modules.

One file per robot type:

* :mod:`arena.embodiments.so_arm`  — SO-ARM 101 (manipulation arm).
* :mod:`arena.embodiments.g1`      — Unitree G1 (humanoid loco-manip + dex hands).
* :mod:`arena.embodiments.franka`  — Franka Panda (ultrasound liver-scan).

Envs that need the asset registry to know about a robot import the matching
module from their :meth:`register_assets` hook so the ``@register_asset``
side effects run before scene construction.
"""
