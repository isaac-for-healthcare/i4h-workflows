# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Continuation mismatches must block native qualification, including fresh resume."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "qualify_navigation_native", Path(__file__).parents[1] / "scripts/qualify_navigation_native.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.mark.parametrize(
    "metric,value",
    [
        ("same_process_trajectory_exact", False),
        ("fresh_trajectory_exact", False),
        ("same_process_robot_max_abs_rad", 0.0002943277359008789),
        ("fresh_robot_max_abs_rad", float("nan")),
    ],
)
def test_inexact_continuation_overrides_a_snapshot_pass(metric, value):
    result = driver.finalize_result({"passed": True, "exact_state_roundtrip": True, metric: value})
    assert not result["passed"]
    assert result["continuation_failures"] == [metric]


def test_exact_continuation_preserves_pass_and_existing_failure():
    for passed in (True, False):
        result = dict(passed=passed, fresh_trajectory_exact=True, fresh_robot_max_abs_rad=0.0)
        assert driver.finalize_result(result)["passed"] is passed
