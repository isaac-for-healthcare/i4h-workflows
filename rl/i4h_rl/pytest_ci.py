# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI must execute every regression in the four Trocar review modules."""

import os

REQUIRED_MODULES = {
    "test_trocar_preflight.py",
    "test_rlinf_recording.py",
    "test_rlinf_video.py",
    "test_rlinf_collective_failures.py",
}


def pytest_sessionfinish(session, exitstatus):
    if not os.environ.get("CI"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    passed = {report.nodeid.split("::")[0].rsplit("/", 1)[-1] for report in reporter.stats.get("passed", [])}
    skipped = {report.nodeid.split("::")[0].rsplit("/", 1)[-1] for report in reporter.stats.get("skipped", [])}
    if not passed >= REQUIRED_MODULES or REQUIRED_MODULES & skipped:
        reporter.write_sep("!", "CI requires all four RL regression modules to execute without skips")
        session.exitstatus = 1
