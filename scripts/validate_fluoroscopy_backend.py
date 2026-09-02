# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render two patient-backed DRRs and require C-arm motion to change the image."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from i4h_arena.medical.carm import CArmState
from i4h_arena.medical.catheter import CatheterState
from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_volume import PatientVolume
from i4h_arena.medical.slang_fluoroscopy import SlangFluoroscopyRenderer


def _carm(
    isocenter_world_m: np.ndarray,
    *,
    angle_rad: float,
    source_detector_distance_m: float,
    detector_size_m: float,
) -> CArmState:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    rotation_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
        dtype=np.float64,
    )
    half_distance = 0.5 * source_detector_distance_m
    source = isocenter_world_m + rotation_x @ np.asarray((0.0, 0.0, -half_distance))
    detector = isocenter_world_m + rotation_x @ np.asarray((0.0, 0.0, half_distance))
    return CArmState(
        source_world_m=source[None, :],
        detector_center_world_m=detector[None, :],
        detector_x_axis_world=np.asarray(((1.0, 0.0, 0.0),)),
        detector_size_m=(detector_size_m, detector_size_m),
    )


def validate(args: argparse.Namespace) -> dict[str, object]:
    patient_twin = PatientTwin.load(args.patient_twin)
    patient = PatientVolume.load(patient_twin)
    isocenter = np.asarray(args.isocenter_world_m, dtype=np.float64)
    ap_carm = _carm(
        isocenter,
        angle_rad=0.0,
        source_detector_distance_m=args.source_detector_distance_m,
        detector_size_m=args.detector_size_m,
    )
    oblique_carm = _carm(
        isocenter,
        angle_rad=args.angle_rad,
        source_detector_distance_m=args.source_detector_distance_m,
        detector_size_m=args.detector_size_m,
    )

    started = time.perf_counter()
    renderer = SlangFluoroscopyRenderer(
        patient,
        ap_carm,
        width=args.width,
        height=args.height,
        step_mm=args.step_mm,
        device_type=args.device,
    )
    initialization_s = time.perf_counter() - started
    catheter = CatheterState.empty(1)

    started = time.perf_counter()
    ap = renderer.render(catheter, ap_carm)["rgb"][0]
    ap_render_s = time.perf_counter() - started
    started = time.perf_counter()
    oblique = renderer.render(catheter, oblique_carm)["rgb"][0]
    oblique_render_s = time.perf_counter() - started
    mean_absolute_delta = float(np.abs(ap.astype(np.int16) - oblique.astype(np.int16)).mean())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ap).save(args.output_dir / "ap.png")
    Image.fromarray(oblique).save(args.output_dir / "oblique.png")
    metrics: dict[str, object] = {
        "patient_id": patient_twin.patient_id,
        "volume_shape_zyx": list(patient.shape_zyx),
        "device": args.device,
        "image_shape": [args.height, args.width],
        "angle_rad": args.angle_rad,
        "isocenter_world_m": args.isocenter_world_m,
        "initialization_s": initialization_s,
        "ap_render_s": ap_render_s,
        "oblique_render_s": oblique_render_s,
        "ap_mean": float(ap.mean()),
        "ap_std": float(ap.std()),
        "oblique_mean": float(oblique.mean()),
        "oblique_std": float(oblique.std()),
        "mean_absolute_delta": mean_absolute_delta,
        "minimum_delta": args.min_frame_delta,
        "passed": mean_absolute_delta >= args.min_frame_delta,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-twin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("vulkan", "cuda"), default="vulkan")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--step-mm", type=float, default=1.0)
    parser.add_argument("--angle-rad", type=float, default=0.35)
    parser.add_argument("--min-frame-delta", type=float, default=2.0)
    parser.add_argument("--source-detector-distance-m", type=float, default=0.65)
    parser.add_argument("--detector-size-m", type=float, default=0.36)
    parser.add_argument("--isocenter-world-m", type=float, nargs=3, default=(0.0, 0.0, 0.38))
    args = parser.parse_args()
    if min(args.width, args.height) <= 0:
        parser.error("--width and --height must be positive")
    if min(args.step_mm, args.source_detector_distance_m, args.detector_size_m) <= 0.0:
        parser.error("step, source-detector distance, and detector size must be positive")
    if args.min_frame_delta < 0.0:
        parser.error("--min-frame-delta cannot be negative")
    args.patient_twin = args.patient_twin.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    metrics = validate(args)
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        print(f"fluoroscopy validation failed; inspect {args.output_dir}")
        return 1
    print(f"fluoroscopy validation passed; inspect {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
