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

import unittest

from helpers import requires_isaac_sim, run_with_monitoring_capture

COMMAND = (
    "PYTHONPATH=scripts /isaac-sim/python.sh -u -m simulation.examples.eval_assemble_trocar "
    "--test --headless --num_episodes 1 --max_steps 2 --enable_cameras"
)
TIMEOUT = 900
TARGET_LINES = ["Creating environment", "EVALUATION SUMMARY"]


class TestEvalAssembleTrocar(unittest.TestCase):
    @requires_isaac_sim
    def test_eval_assemble_trocar_starts_env_and_cleans_up(self):
        return_code, found_target, out_lines = run_with_monitoring_capture(COMMAND, TIMEOUT, TARGET_LINES)
        if not found_target:
            tail = "\n".join(out_lines[-200:]) if out_lines else "<no output captured>"
            self.fail(f"Did not find target lines. return_code={return_code}\n--- last output ---\n{tail}")


if __name__ == "__main__":
    unittest.main()
