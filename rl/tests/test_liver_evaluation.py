# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from i4h_rl.adapters.liver_reach_retract import terminal_result
from i4h_rl.liver_experiment import summarize


def test_deadline_is_explicit_failure_despite_no_truncation():
    extras = {
        "time_outs": False,
        "terminal_results": [dict(env_id=2, objective="full_cycle", success=False, failure_reason="deadline")],
    }
    assert terminal_result(extras, 2)["success"] is False


def test_full_cycle_rejects_curriculum_success_and_missing_result():
    with pytest.raises(ValueError, match="curriculum"):
        terminal_result({"terminal_results": [dict(env_id=0, objective="reach_only", success=True)]}, 0)
    with pytest.raises(RuntimeError, match="missing"):
        terminal_result({}, 0)


def test_wilson_intervals_and_target_counts():
    rows = [dict(target=252, success=i < 3, failure_reason=None if i < 3 else "retention_loss") for i in range(4)]
    report = summarize(rows)
    assert report["successes"] == 3 and report["episodes"] == 4
    assert 0.29 < report["wilson_95"][0] < 0.31
    assert 0.95 < report["wilson_95"][1] < 0.96
    assert report["per_target"]["252"]["failures"] == {"retention_loss": 1}
