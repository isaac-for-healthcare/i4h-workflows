# WARNING: THIS FILE IS AUTO-GENERATED. DO NOT MODIFY.

# This file was generated from franka_ctrl.idl
# using RTI Code Generator (rtiddsgen) version 4.3.0.
# The rtiddsgen tool is part of the RTI Connext DDS distribution.
# For more information, type 'rtiddsgen -help' at a command shell
# or consult the Code Generator User's Manual.

from dataclasses import field
from typing import Sequence

import rti.idl as idl


@idl.struct
class FrankaCtrlInput:
    target_position: Sequence[float] = field(default_factory=idl.array_factory(float))
    target_orientation: Sequence[float] = field(default_factory=idl.array_factory(float))
    joint_positions: Sequence[float] = field(default_factory=idl.array_factory(float))
    joint_velocities: Sequence[float] = field(default_factory=idl.array_factory(float))
    joint_efforts: Sequence[float] = field(default_factory=idl.array_factory(float))
