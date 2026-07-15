# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic IsaacLab-Arena package.

A single ``arena`` package exposes a flat set of environments, embodiments,
assets, and tasks. Layout mirrors IsaacLab-Arena conventions:

* :mod:`arena.environments` — env classes inheriting
  :class:`~arena.environments.core.base.AgenticEnvironmentBase`.
* :mod:`arena.embodiments` — robot embodiments under
  ``arena.embodiments.<robot>``, registered via ``@register_asset``.
* :mod:`arena.assets` — scene background/object asset libraries.
* :mod:`arena.tasks` — task definitions.
* :mod:`arena.runtimes` — shared per-policy zenoh runtime helpers.
* :mod:`arena.run` — generic CLI dispatcher.
"""
