# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

from common.io.base import MessagePublisher, MessageSubscriber
from common.messages import RobotCommand, RobotState, decode_command, decode_state

RobotCommandCallback = Callable[[RobotCommand], None]
RobotStateCallback = Callable[[RobotState], None]


class RobotStatePublisher(MessagePublisher[RobotState]):
    pass


class RobotCommandPublisher(MessagePublisher[RobotCommand]):
    pass


class RobotStateSubscriber(MessageSubscriber[RobotState]):
    decoder = decode_state


class RobotCommandSubscriber(MessageSubscriber[RobotCommand]):
    decoder = decode_command
