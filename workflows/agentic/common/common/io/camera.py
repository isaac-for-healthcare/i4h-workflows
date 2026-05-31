# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

from common.io.base import MessagePublisher, MessageSubscriber
from common.messages import CameraStream, decode_camera

CameraCallback = Callable[[CameraStream], None]


class CameraPublisher(MessagePublisher[CameraStream]):
    pass


class CameraSubscriber(MessageSubscriber[CameraStream]):
    decoder = decode_camera
