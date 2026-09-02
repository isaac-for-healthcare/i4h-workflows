# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from i4h_arena.assets.contact import filtered_contact_sensor_family


def test_filtered_contact_sensor_family_builds_one_sensor_per_body() -> None:
    sensors = filtered_contact_sensor_family(
        family_name="contact_robot_table",
        sensing_prim_root="{ENV_REGEX_NS}/Robot/",
        body_names=("pelvis", "left_hand"),
        filter_prim_path="{ENV_REGEX_NS}/TableContactProxy",
    )

    assert tuple(sensors) == (
        "contact_robot_table__pelvis",
        "contact_robot_table__left_hand",
    )
    assert sensors["contact_robot_table__pelvis"].prim_path == "{ENV_REGEX_NS}/Robot/pelvis"
    assert sensors["contact_robot_table__left_hand"].filter_prim_paths_expr == ["{ENV_REGEX_NS}/TableContactProxy"]


@pytest.mark.parametrize(
    ("family_name", "body_names"),
    [
        ("", ("pelvis",)),
        ("contact__bad", ("pelvis",)),
        ("contact_robot_table", ()),
        ("contact_robot_table", ("pelvis", "pelvis")),
    ],
)
def test_filtered_contact_sensor_family_rejects_ambiguous_contracts(
    family_name: str,
    body_names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        filtered_contact_sensor_family(
            family_name=family_name,
            sensing_prim_root="{ENV_REGEX_NS}/Robot",
            body_names=body_names,
            filter_prim_path="{ENV_REGEX_NS}/TableContactProxy",
        )
