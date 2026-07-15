# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only smoke tests for the fluorosim package and its CLI entry points.

These intentionally avoid the GPU Slang renderer so they can run in CI without a
GPU: they exercise the CT->mu preprocessing path, the synthetic phantom builder,
and that the example argument parsers are wired up correctly.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# The fluorosim package imports itself as a top-level `fluorosim` package, so its
# import root is scripts/simulation. parents[1] is the catheter_navigation dir.
_PKG_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "simulation"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


class TestFluorosimImports(unittest.TestCase):
    def test_core_api_importable(self):
        import fluorosim

        for name in (
            "VolumePreprocessor",
            "FluoroSimulator",
            "SimulatorConfig",
            "PreprocessedVolume",
            "RealismSettings",
        ):
            self.assertTrue(hasattr(fluorosim, name), f"missing {name}")


class TestPreprocessing(unittest.TestCase):
    def test_hu_to_mu_from_numpy(self):
        from fluorosim import PreprocessedVolume, VolumePreprocessor

        hu = np.full((16, 16, 16), -1000.0, dtype=np.float32)
        hu[4:12, 4:12, 4:12] = 1200.0  # a dense block

        volume = VolumePreprocessor.from_numpy(hu, spacing_zyx_mm=(1.0, 1.0, 1.0)).preprocess()

        self.assertIsInstance(volume, PreprocessedVolume)
        self.assertEqual(volume.shape, (16, 16, 16))
        self.assertEqual(volume.mu_volume.dtype, np.float32)
        # Default HU->mu mapping maps into [mu_min, mu_max] = [0.0, 0.02].
        self.assertGreaterEqual(float(volume.mu_volume.min()), 0.0)
        self.assertLessEqual(float(volume.mu_volume.max()), 0.02 + 1e-6)

    def test_preprocess_roundtrip_save_load(self):
        import tempfile

        from fluorosim import PreprocessedVolume, VolumePreprocessor

        hu = np.zeros((8, 8, 8), dtype=np.float32)
        volume = VolumePreprocessor.from_numpy(hu).preprocess()
        with tempfile.TemporaryDirectory() as d:
            volume.save(d)
            reloaded = PreprocessedVolume.load(d)
            self.assertEqual(reloaded.shape, volume.shape)


class TestExampleEntryPoints(unittest.TestCase):
    def test_render_drr_parser_and_phantom(self):
        from fluorosim.examples import render_drr

        args = render_drr.build_parser().parse_args(["--view", "ap", "--realism", "--output", "out.png"])
        self.assertEqual(args.view, "ap")
        self.assertTrue(args.realism)

        realism = render_drr.build_realism(args)
        self.assertTrue(realism.enabled)
        self.assertNotEqual(realism.gamma, 1.0)

        phantom = render_drr.make_synthetic_phantom(size=24)
        self.assertEqual(phantom.shape, (24, 24, 24))

    def test_preprocess_ct_parser_requires_source(self):
        from fluorosim.examples import preprocess_ct

        parser = preprocess_ct.build_parser()
        args = parser.parse_args(["--dicom", "/tmp/x", "--output-dir", "/tmp/y"])
        self.assertEqual(args.dicom, "/tmp/x")
        self.assertEqual(args.output_dir, "/tmp/y")

        # A source (--dicom/--nifti) is mandatory.
        with self.assertRaises(SystemExit):
            parser.parse_args(["--output-dir", "/tmp/y"])

    def test_segment_vessels_parser(self):
        from fluorosim.examples import segment_vessels

        parser = segment_vessels.build_parser()
        args = parser.parse_args(["--ct-dir", "/tmp/cache", "--ts-gt-dir", "/tmp/subj/segmentations"])
        self.assertEqual(args.ct_dir, "/tmp/cache")
        self.assertEqual(args.ts_gt_dir, "/tmp/subj/segmentations")
        # Default arterial tree is aorta -> iliac.
        self.assertIn("aorta", args.ts_labels)  # codespell:ignore assertin

        # --ct-dir is mandatory.
        with self.assertRaises(SystemExit):
            parser.parse_args(["--ts-gt-dir", "/tmp/subj/segmentations"])


class TestInteractiveViewport(unittest.TestCase):
    """The interactive viewport pulls in GPU/UI deps (cv2, torch, warp); only
    exercise its import when those are available so CPU-only CI still passes."""

    def test_viewport_importable_when_deps_present(self):
        import importlib.util

        for dep in ("cv2", "torch", "warp"):
            if importlib.util.find_spec(dep) is None:
                self.skipTest(f"{dep} not installed")

        from fluorosim.examples import interactive_catheter_slang_viewport as viewport

        self.assertTrue(hasattr(viewport, "main"))
        self.assertTrue(hasattr(viewport, "SlangViewportApp"))


if __name__ == "__main__":
    unittest.main()
