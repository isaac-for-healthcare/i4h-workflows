# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from helpers import run_with_monitoring
from parameterized import parameterized

SM_CASES = [
    (
        "python -u -m simulation.imitation_learning.pi0_policy.eval --enable_camera --headless",
        300,
        "Resetting the environment.",
    ),
]


class TestPolicyEval(unittest.TestCase):
    @parameterized.expand(SM_CASES)
    def test_policy_eval(self, command, timeout, target_line):
        # Run and monitor command
        _, found_target = run_with_monitoring(command, timeout, target_line)
        self.assertTrue(found_target)


if __name__ == "__main__":
    unittest.main()
