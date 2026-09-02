# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point for patient-twin preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import DEFAULT_LABELS, DEFAULT_PRESET, PRESETS, build_patient_twin


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build every artifact required by the Arena fluoroscopy workflow from one TotalSegmentator subject."
    )
    parser.add_argument("subject", type=Path, help="Directory containing ct.nii.gz and segmentations/")
    parser.add_argument("--output", type=Path, help="Write to another directory instead of the subject directory")
    parser.add_argument("--patient-id", help="Default: subject directory name")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated TotalSegmentator vessel labels",
    )
    parser.add_argument(
        "--segment-vessels",
        action="store_true",
        help="Segment the vasculature from the CT with the digital-twin segmenter instead of reading "
        "segmentations/, for a subject that ships no label files. Ignores --labels",
    )
    parser.add_argument("--close-iterations", type=int, default=2)
    parser.add_argument("--surface-step", type=int, default=3)
    parser.add_argument(
        "--hu-to-mu",
        choices=sorted(PRESETS),
        default=DEFAULT_PRESET,
        help="Attenuation curve: 'interventional' separates implant density from bone, 'linear' reproduces "
        "twins built before named curves existed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subject = args.subject.expanduser().resolve()
    output = args.output or subject
    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    build_patient_twin(
        subject,
        output,
        patient_id=args.patient_id,
        labels=labels,
        close_iterations=args.close_iterations,
        surface_step=args.surface_step,
        hu_to_mu_preset=args.hu_to_mu,
        segment_vessels=args.segment_vessels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
