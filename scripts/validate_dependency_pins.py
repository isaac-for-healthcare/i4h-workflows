#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate pinned fluoroscopy renderers inside Kit, in one fresh process per device.

Run after setup: arena/.venv/bin/python scripts/validate_dependency_pins.py
Requires both Vulkan and CUDA; a device failure is never converted into a skip.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker(device: str) -> None:
    from i4h_arena.medical.slang_cuda import prepare_slang_for_kit

    prepare_slang_for_kit(device)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import importlib.util
        import json
        import tempfile

        import numpy as np
        import slangpy
        import torch
        from xray_simulator.rendering.diffdrr_slang_renderer import SlangDiffDRRRenderer

        from i4h_arena.medical.carm import CArmState
        from i4h_arena.medical.catheter import CatheterState
        from i4h_arena.medical.patient_volume import PatientVolume
        from i4h_arena.medical.slang_fluoroscopy import _SHARED_SLANG_STATE, SlangFluoroscopyRenderer

        # Reuse small patient fixture geometry, but exercise the actual installed sensor.
        spec = importlib.util.spec_from_file_location("fixture", ROOT / "arena/tests/test_slang_fluoroscopy.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        fixture.test_the_shared_state_is_what_upstream_init_slang_creates()
        print(f"DEPENDENCY {sys.modules[SlangDiffDRRRenderer.__module__].__file__}", flush=True)
        with tempfile.TemporaryDirectory() as directory:
            patient = fixture._patient_volume(Path(directory), with_vessel_mask=True)
            # A graded phantom leaves display headroom for the vessel boost;
            # a uniform tiny cube clips both images to the same display endpoint.
            directory = Path(directory)
            shape = (16, 16, 16)
            mu = np.broadcast_to(np.linspace(0.001, 0.02, 16, dtype=np.float32), shape).copy()
            mask = np.zeros(shape, dtype=np.uint8)
            mask[:, 6:10, 6:10] = 1
            np.save(directory / "mu_volume.npy", mu)
            np.save(directory / "vessel_mask.npy", mask)
            (directory / "metadata.json").write_text(
                json.dumps({"shape_zyx": shape, "spacing_zyx_mm": [3.0, 2.0, 1.0]})
            )
            patient = PatientVolume.load(patient.twin)
            carm = fixture._carm(patient)
            for dsa in (False, True):
                renderer = SlangFluoroscopyRenderer(
                    patient,
                    carm,
                    width=64,
                    height=64,
                    step_mm=1.0,
                    device_type=device,
                    dsa=dsa,
                    catheter_attenuation=False,
                )
                assert renderer._renderer._device.info.type == getattr(slangpy.DeviceType, device)
                for count in (1, 2, 3, 1):
                    offsets = np.zeros((count, 3))
                    offsets[:, 0] = np.arange(count) * 0.004
                    poses = CArmState(
                        source_world_m=np.repeat(carm.source_world_m, count, axis=0) + offsets,
                        detector_center_world_m=np.repeat(carm.detector_center_world_m, count, axis=0) + offsets,
                        detector_x_axis_world=np.repeat(carm.detector_x_axis_world, count, axis=0),
                        detector_size_m=carm.detector_size_m,
                    )
                    frames = renderer.render(CatheterState.empty(count), poses)
                    previous = {key: value.clone() for key, value in frames.items()}
                    repeated = renderer.render(CatheterState.empty(count), poses)
                    for key in previous:
                        assert torch.isfinite(repeated[key]).all()
                        torch.testing.assert_close(previous[key], repeated[key])
                    if count > 1:
                        assert not torch.equal(frames["attenuation"][0], frames["attenuation"][1])
                    if dsa:
                        main, contrast = renderer._renderer, renderer._dsa_renderer
                        assert all(getattr(main, name) is getattr(contrast, name) for name in _SHARED_SLANG_STATE)
                        assert main._batch_cache is not None and contrast._batch_cache is not None
                        assert main._batch_cache is not contrast._batch_cache
                        assert not torch.equal(frames["rgb"], frames["dsa"])
                    print(f"PASS device={device} dsa={dsa} batch={count} consecutive_frames=2", flush=True)
                # Each test case owns its device; avoid overlapping independent
                # Slang devices while Kit's background workers are active.
                device_handle = renderer._renderer._device
                del renderer
                if dsa:
                    del main, contrast
                device_handle.close()
                del device_handle
                print(f"CLOSED device={device} dsa={dsa}", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        print(f"COMPLETE device={device}", flush=True)
        app.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "vulkan"), help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.device:
        worker(args.device)
        return 0
    sys.path.insert(0, str(ROOT / "rl"))
    from i4h_rl.runtime import apply_arm_openmp_env

    environment = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"}
    apply_arm_openmp_env(environment)
    for device in ("cuda", "vulkan"):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--device", device]
        print("RUN " + " ".join(command), flush=True)
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                timeout=args.timeout,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            for output in (error.stdout, error.stderr):
                if output:
                    print(output.decode(errors="replace") if isinstance(output, bytes) else output, flush=True)
            print(f"FAIL {device}: exceeded {args.timeout}s", file=sys.stderr)
            return 1
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode:
            return result.returncode
        if f"COMPLETE device={device}" not in result.stdout:
            print(f"FAIL {device}: exited without completing the assertions", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
