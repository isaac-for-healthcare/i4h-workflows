# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build Arena bundles with the independently installed patient library."""

from patient_digital_twin.__main__ import run_pipeline


def build_patient_twin(*, source, classes, output, input=None, format="bundle", **options):
    """Run generation/segmentation and return the resulting patient manifest."""
    if format != "bundle":
        raise ValueError("The workflow builder requires --format bundle")
    return run_pipeline(source=source, classes=classes, output=output, input=input, format="bundle", **options)
