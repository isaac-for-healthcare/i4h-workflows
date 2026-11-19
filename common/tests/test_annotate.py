# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A workflow's grading settings for the VLM annotator (NVBug 6941887)."""

from __future__ import annotations

import pytest

from i4h_common.annotate import JUDGES, Annotate


def test_every_field_is_optional() -> None:
    assert Annotate() == Annotate(criterion="", cameras=(), judge="", frames=0)


def test_the_criterion_is_flattened_and_cameras_become_a_tuple() -> None:
    annotate = Annotate(
        criterion="The probe touches the phantom\n    and sweeps across it.",
        cameras=["room", "wrist"],  # type: ignore[arg-type]
        judge="trajectory",
        frames=8,
    )

    assert annotate.criterion == "The probe touches the phantom and sweeps across it."
    assert annotate.cameras == ("room", "wrist")
    assert annotate.judge in JUDGES


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cameras": "room"}, "not one string"),
        ({"cameras": ("room", "")}, "must not be empty"),
        ({"cameras": ("room", "room")}, "name a camera twice"),
        ({"judge": "final"}, "is not one of"),
        ({"frames": -1}, "frame count"),
        ({"frames": True}, "frame count"),
    ],
)
def test_a_malformed_setting_is_rejected(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Annotate(**kwargs)
