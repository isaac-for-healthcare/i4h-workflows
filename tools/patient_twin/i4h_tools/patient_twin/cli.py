# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a patient with the shared NV-Generate/NV-Segment pipeline."""

from patient_digital_twin.__main__ import parser

from .pipeline import build_patient_twin


def build_parser():
    result = parser()
    result.description = "Build an Arena patient bundle with NV-Generate or NV-Segment."
    result.set_defaults(format="bundle")
    return result


def main(argv=None):
    cli = build_parser()
    args = cli.parse_args(argv)
    try:
        manifest = build_patient_twin(**vars(args))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        cli.error(str(exc))
    print(f"PATIENT_TWIN={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
