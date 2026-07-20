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

"""CPU-only smoke tests for catheter workflow wiring and core imports.

These intentionally avoid the GPU renderer path so they can run in CI without a
GPU. They cover:
  - core `fluorosim` package imports and CPU preprocessing utilities
  - workflow mode command wiring in `metadata.json`
  - local interactive viewport module importability when deps are present
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

# The fluorosim package imports itself as a top-level `fluorosim` package, so its
# import root is scripts/simulation. parents[1] is the catheter_navigation dir.
_PKG_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "simulation"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

_WF_ROOT = Path(__file__).resolve().parents[1]
_METADATA_PATH = _WF_ROOT / "metadata.json"
_VIEWPORT_PATH = (
    _WF_ROOT / "scripts" / "simulation" / "fluorosim" / "examples" / "interactive_catheter_slang_viewport.py"
)


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


class TestWorkflowModeWiring(unittest.TestCase):
    def test_metadata_mode_commands_match_split_package_layout(self):
        self.assertTrue(_METADATA_PATH.is_file(), f"Missing workflow metadata: {_METADATA_PATH}")
        metadata = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
        modes = metadata["workflow"]["modes"]

        self.assertEqual(
            modes["preprocess_ct"]["run"]["command"],
            "python -m vasculature_digital_twin.cli.preprocess_ct",
        )
        self.assertEqual(
            modes["segment_vessels"]["run"]["command"],
            "python -m vasculature_digital_twin.cli.segment_vessels",
        )
        self.assertEqual(
            modes["interactive_viewport"]["run"]["command"],
            "python scripts/simulation/fluorosim/examples/interactive_catheter_slang_viewport.py",
        )
        # DRR rendering is launched via the local workflow script path.
        self.assertEqual(
            modes["render_drr"]["run"]["command"],
            "python scripts/simulation/fluorosim/examples/render_drr.py",
        )


class TestInteractiveViewport(unittest.TestCase):
    """The interactive viewport pulls in GPU/UI deps (cv2, torch, warp); only
    exercise its import when those are available so CPU-only CI still passes."""

    def test_viewport_importable_when_deps_present(self):
        for dep in (
            "cv2",
            "torch",
            "warp",
            "fluorosim",
            "vasculature_digital_twin",
            "catheter_vasculature_solver",
        ):
            if importlib.util.find_spec(dep) is None:
                self.skipTest(f"{dep} not installed")

        if not _VIEWPORT_PATH.is_file():
            self.fail(f"Missing viewport script: {_VIEWPORT_PATH}")

        spec = importlib.util.spec_from_file_location("interactive_catheter_slang_viewport", _VIEWPORT_PATH)
        self.assertIsNotNone(spec, "Failed to create import spec for viewport script")
        self.assertIsNotNone(spec.loader, "Viewport module loader is unavailable")

        viewport = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viewport)

        self.assertTrue(hasattr(viewport, "main"))
        self.assertTrue(hasattr(viewport, "SlangViewportApp"))


if __name__ == "__main__":
    unittest.main()
