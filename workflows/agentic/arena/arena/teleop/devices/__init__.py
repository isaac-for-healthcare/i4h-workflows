# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Teleoperation devices.

Two styles coexist here, chosen by the env family that uses them:

* **Manip-style** (SO-ARM scissor env): :mod:`so101_keyboard`, :mod:`so101_leader`.
  Callback-based delta-action drivers consumed by :mod:`arena.teleop`.
  Selected via ``--teleop-device``; the env loop calls ``advance()`` each
  step and translates the result into joint-space deltas.

* **IsaacLab-Arena style** (humanoid envs — locomanip, trocar):
  :mod:`gamepad`. Subclass
  :class:`isaaclab_arena.teleop_devices.teleop_device_base.TeleopDeviceBase`
  with an ``@register_device`` decorator and expose ``get_teleop_device_cfg``
  returning an IsaacLab ``DevicesCfg``. Resolved via
  :class:`isaaclab_arena.assets.AssetRegistry`.

* **Record-script helper**: :mod:`keyboard_23d` provides a 23-DoF G1 WBC
  keyboard adapter for offline data recording flows. It does not register
  itself as a device because it returns raw joint-state actions directly,
  not a ``DevicesCfg``.
"""
