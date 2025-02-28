# WARNING: THIS FILE IS AUTO-GENERATED. DO NOT MODIFY.

# This file was generated from target_info.idl
# using RTI Code Generator (rtiddsgen) version 4.3.0.
# The rtiddsgen tool is part of the RTI Connext DDS distribution.
# For more information, type 'rtiddsgen -help' at a command shell
# or consult the Code Generator User's Manual.

from dataclasses import field
from typing import Sequence

import rti.idl as idl


@idl.struct
class TargetInfo:
    position: Sequence[float] = field(default_factory=idl.array_factory(float))
    orientation: Sequence[float] = field(default_factory=idl.array_factory(float))
