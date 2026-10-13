# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The catheter descriptor has to satisfy dataset conversion's split contract.

Conversion only emits a GR00T ``modality.json`` when the declared groups tile
the converted tensors exactly; otherwise it silently falls back to a generic
layout and the resulting dataset cannot be used to fine-tune. The failure is
invisible until training, so the rule is asserted here rather than discovered
after a recording session.

``_uses_declared_modality`` in ``tools/dataset`` is the authority; ``tiles``
below restates it so a change on either side shows up as a failed test instead
of a dataset that converts and then will not train.
"""

from __future__ import annotations

import pytest

from i4h_common.config import get_robot_config

CATHETER_GROUP = "catheter"
CARM_GROUP = "carm"


def tiles(splits: tuple[tuple[str, int, int], ...], width: int) -> bool:
    """Whether groups start at 0, stay contiguous, and end at ``width``."""
    cursor = 0
    for _name, start, end in splits:
        if start != cursor or end <= start:
            return False
        cursor = end
    return bool(splits) and cursor == width


@pytest.fixture
def catheter():
    return get_robot_config("catheter")


def test_the_catheter_declares_modality_groups(catheter) -> None:
    """Without these, conversion cannot write a modality.json at all."""
    assert catheter.state_split, "catheter.yaml declares no state_split"
    assert catheter.action_split, "catheter.yaml declares no action_split"


def test_the_state_groups_tile_the_state_vector(catheter) -> None:
    assert tiles(catheter.state_split, len(catheter.state_names))


def test_the_action_groups_tile_the_action_vector(catheter) -> None:
    assert tiles(catheter.action_split, len(catheter.action_names))


def test_the_instrument_and_the_gantry_stay_separate(catheter) -> None:
    """Insertion and rotation move the catheter; the orbit only moves the view.

    Training a navigation policy on the catheter group alone depends on the
    C-arm being addressable as its own group, so a single combined group would
    be a regression even though it would still tile correctly.

    Only the commanded groups appear in the action split. The state split leads
    with the same two, so a checkpoint reading the drive columns finds them at
    the offsets it was fine-tuned against, and then carries the goal groups.
    """
    assert [name for name, _start, _end in catheter.action_split] == [CATHETER_GROUP, CARM_GROUP]

    state_groups = [name for name, _start, _end in catheter.state_split]
    assert state_groups[:2] == [CATHETER_GROUP, CARM_GROUP]
    assert len(set(state_groups)) == len(state_groups), f"a group is declared twice: {state_groups}"


def test_the_catheter_group_is_the_instrument_commands(catheter) -> None:
    """Pin the ordering the group names claim, not just the widths.

    All three instrument channels belong to one group: they act inside the
    patient, so a policy trained on the catheter group alone still has insertion,
    rotation, and the steerable tip. Only the gantry is separable.
    """
    width = catheter.split_width(CATHETER_GROUP)
    assert width == 3
    assert catheter.action_names[:width] == (
        "insertion_velocity_mps",
        "rotation_rate_radps",
        "tip_bend_rate_radps",
    )
    assert catheter.state_names[:width] == ("insertion_m", "rotation_rad", "tip_bend_rad")


def test_the_carm_group_is_the_orbit(catheter) -> None:
    """The gantry stays a group of one, and stays last."""
    assert catheter.split_width(CARM_GROUP) == 1
    assert catheter.action_names[3] == "carm_orbit_rate_radps"
    assert catheter.state_names[3] == "carm_orbit_rad"
