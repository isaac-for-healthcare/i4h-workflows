# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the maintained G1 client config with the simulator's actual Arena API."""

import hashlib
from pathlib import Path


def test_g1_policy_config_selects_upstream_adapter_and_preserves_joint_mapping():
    import numpy as np
    from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg
    from isaaclab_arena_gr00t.policy.embodiment_adapters.gr00t_embodiment_adapter import (
        import_gr00t_embodiment_adapter_class,
    )
    from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml

    root = Path(__file__).resolve().parents[2]
    cfg = create_config_from_yaml(root / "tasks/gr00t_n16/config/g1_locomanipulation.yaml", Gr00tClosedloopPolicyCfg)
    adapter = import_gr00t_embodiment_adapter_class(cfg.gr00t_embodiment_adapter)()
    assert cfg.action_horizon == cfg.action_chunk_length == 16
    assert adapter.action_dim == 50
    # The adapter now owns these mappings; the isolated server keeps its original Arena pin.
    expected = {
        "43dof_joint_space.yaml": "ff7db8a6e4da8cb6ad40329bf0f5175de00d60f804e39c7183dc7d3c2e7d24a7",
        "gr00t_43dof_joint_space.yaml": "28226939039bd29fce7bf93d7f3fccce66bd3aa74a4c49e89d8229cd946357ae",
    }
    for current in (adapter.sim_joints_config_path, adapter.policy_joints_config_path):
        assert hashlib.sha256(Path(current).read_bytes()).hexdigest() == expected[Path(current).name]
    joints = np.arange(43, dtype=np.float64).reshape(1, 43)
    state = adapter.extract_state_np({"policy": {"robot_joint_pos": joints}})
    action = {key: np.repeat(value[:, None, :], 16, axis=1) for key, value in state.items()}
    action["navigate_command"] = np.zeros((1, 16, 3))
    action["base_height_command"] = np.full((1, 16, 1), 0.75)
    translated = adapter.build_action_np(action)
    assert translated.shape == (1, 16, 50)
    assert np.isfinite(translated).all()
    # GR00T supplies the upper body; WBC owns leg commands in the existing mapping.
    expected_joints = joints.copy()
    for name in adapter.policy_joints_config["left_leg"] + adapter.policy_joints_config["right_leg"]:
        expected_joints[:, adapter.sim_joints_config[name]] = 0
    np.testing.assert_array_equal(translated[:, :, :43], np.repeat(expected_joints[:, None, :], 16, axis=1))
