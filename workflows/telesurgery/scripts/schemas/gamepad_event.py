# WARNING: THIS FILE IS AUTO-GENERATED. DO NOT MODIFY.

# This file was generated from joystick_event.idl
# using RTI Code Generator (rtiddsgen) version 4.5.0.
# The rtiddsgen tool is part of the RTI Connext DDS distribution.
# For more information, type 'rtiddsgen -help' at a command shell
# or consult the Code Generator User's Manual.

from dataclasses import field
from typing import Sequence

import rti.idl as idl


@idl.struct
class GamepadEvent:
    ts: int = 0
    type: int = 0
    name: str = ""
    os: str = ""
    axes: Sequence[float] = field(default_factory=idl.array_factory(float))
    buttons: Sequence[bool] = field(default_factory=idl.array_factory(bool))
    hats: Sequence[float] = field(default_factory=idl.array_factory(float))
