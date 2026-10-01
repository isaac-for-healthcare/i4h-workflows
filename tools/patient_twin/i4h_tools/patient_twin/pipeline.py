# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build Arena bundles with the independently installed patient library."""

from patient_digital_twin.main import run_pipeline


def build_patient_twin(*, source, classes, output, input=None, format="workflow", **options):
    """Run generation/segmentation and return the resulting patient manifest."""
    if format != "workflow":
        raise ValueError("The workflow builder requires --format workflow")
    return run_pipeline(source=source, classes=classes, output=output, input=input, format="workflow", **options)
