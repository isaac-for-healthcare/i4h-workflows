# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from i4h_common.training import require_trainable, task_default, train_default


def test_defaults_come_from_the_task_manifest():
    task_id = "gr00t_n16/locomanip_push_cart"
    assert task_default(task_id, "data_config", "") == "unitree_g1_sim_wbc"
    assert train_default(task_id, "output_dir", "") == "/tmp/gr00t_g1_cart"
    assert train_default(task_id, "tune_visual", False) is True


def test_inference_only_task_reports_its_manifest():
    with pytest.raises(SystemExit, match=r"assemble_trocar\.yaml"):
        require_trainable("gr00t_n15/assemble_trocar")
