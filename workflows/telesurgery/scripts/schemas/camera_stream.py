# WARNING: THIS FILE IS AUTO-GENERATED. DO NOT MODIFY.

# This file was generated from camera_stream.idl
# using RTI Code Generator (rtiddsgen) version 4.5.0.
# The rtiddsgen tool is part of the RTI Connext DDS distribution.
# For more information, type 'rtiddsgen -help' at a command shell
# or consult the Code Generator User's Manual.

from dataclasses import field
from typing import Sequence

import rti.idl as idl


@idl.struct
class CameraStream:
    ts: int = 0
    type: int = 0  # color: 2; depth: 1
    format: int = 0
    focal_len: float = 0.0
    stream_id: int = 0
    frame_num: int = 0
    width: int = 0
    height: int = 0
    compress_ratio: float = 0.0
    encode_latency: float = 0.0
    decode_latency: float = 0.0
    data: Sequence[idl.uint8] = field(default_factory=idl.array_factory(idl.uint8))
