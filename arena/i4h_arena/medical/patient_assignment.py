# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which patient each environment holds, resolved from the command line."""

from __future__ import annotations

from collections.abc import Sequence


def normalize_patient_twins(values: str | Sequence[str] | None, num_envs: int) -> str | tuple[str, ...] | None:
    """One twin shared by every environment, or one per environment in order.

    ``--patient-twin`` is repeatable. Given once it keeps its single-twin
    meaning whatever the environment count; given more than once it names
    each environment's patient and must then match ``num_envs``.
    """
    if values is None or isinstance(values, str):
        return values
    twins = tuple(str(value) for value in values)
    if not twins:
        return None
    if len(twins) == 1:
        return twins[0]
    if len(twins) != num_envs:
        raise ValueError(
            f"--patient-twin was given {len(twins)} times for {num_envs} environments; "
            "give it once to share one patient, or once per environment"
        )
    return twins


__all__ = ["normalize_patient_twins"]
