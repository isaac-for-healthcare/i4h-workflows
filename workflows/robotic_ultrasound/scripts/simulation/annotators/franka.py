# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any

import numpy as np
import omni.usd
from dds.publisher import Publisher
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from dds.subscriber import Subscriber
from isaacsim.franka import KinematicsSolver
from isaacsim.core.robots import Robot
from isaacsim.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.utils.types import ArticulationAction
from simulation.configs.config import FrankaConfig


class FrankaPublisher(Publisher):
    """Publisher for Franka robot state information.

    This class handles publishing robot joint states and other relevant information
    through DDS topics. It supports both IK and direct joint control modes.

    Args:
        franka: Franka robot instance
        ik: Whether to use inverse kinematics
        prim_path: USD path to robot
        topic: DDS topic name
        period: Publishing period in seconds
        domain_id: DDS domain identifier
    """

    def __init__(self, franka: Robot, ik: bool, prim_path: str, topic: str, period: float, domain_id):
        """Initialize the Franka publisher."""
        super().__init__(topic, FrankaInfo, period, domain_id)

        self.franka = franka
        self.ik = ik
        self.prim_path = prim_path
        self.stage = omni.usd.get_context().get_stage()

    def produce(self, dt: float, sim_time: float) -> Any:
        """Produce robot state information for publishing.

        Gathers current joint states and packages them for publishing.

        Args:
            dt: Time delta since last physics step
            sim_time: Current simulation time

        Returns:
            FrankaInfo: Robot state information including joint positions and velocities,
                refer to dds.schemas.franka_info.FrankaInfo.
        """
        joints_state = self.franka.get_joints_state()
        output = FrankaInfo()
        output.joints_state_positions = joints_state.positions.tolist()
        output.joints_state_velocities = joints_state.velocities.tolist()
        return output

    @staticmethod
    def new_instance(config: FrankaConfig, franka: Robot):
        """Create a new FrankaPublisher instance from configuration.

        Args:
            config: Franka configuration object
            franka: Franka robot instance
        """
        if not config.topic_info or not config.topic_info.name:
            return None

        return FrankaPublisher(
            franka=franka,
            ik=config.ik,
            prim_path=config.prim_path,
            topic=config.topic_info.name,
            period=config.topic_info.period,
            domain_id=config.topic_info.domain_id,
        )


class FrankaSubscriber(Subscriber):
    """Subscriber for Franka robot control commands.

    This class handles robot control command subscription and execution. It supports
    both IK and RMPFlow control methods, as well as direct joint control.

    Args:
        franka: Reference to the Franka robot instance
        ik: Flag indicating whether IK mode is enabled
        prim_path: USD path to the Franka robot
        topic: DDS topic name
        period: Subscription period in seconds
        domain_id: DDS domain identifier
    """

    def __init__(self, franka: Robot, ik: bool, prim_path: str, topic: str, period: float, domain_id):
        """Initialize the Franka subscriber."""
        super().__init__(topic, FrankaCtrlInput, period, domain_id)

        self.franka = franka
        self.ik = ik
        self.prim_path = prim_path
        self.logger = logging.getLogger(__name__)
        if ik:
            self.franka_controller = KinematicsSolver(franka)
        else:
            self.franka_controller = RMPFlowController(name="target_follower_controller", robot_articulation=franka)
        self.franka_articulation_controller = franka.get_articulation_controller()

    def consume(self, input: FrankaCtrlInput) -> None:
        """Consume FrankaCtrlInput and apply control actions to the robot.

        Handles both Cartesian space targets (position/orientation) and
        joint space targets (positions/velocities/efforts).

        Args:
            input: Control input message containing target states and joint states,
                refer to dds.schemas.franka_ctrl.FrankaCtrlInput.
        """
        actions = None
        if input.target_position:
            if self.ik:
                actions, success = self.franka_controller.compute_inverse_kinematics(
                    target_position=np.array(input.target_position),
                    target_orientation=np.array(input.target_orientation),
                )
                if not success:
                    self.logger.error(
                        f"Can't compute inverse kinematics. pos: {input.target_position}; "
                        f"ori: {input.target_orientation}"
                    )
            else:
                actions = self.franka_controller.forward(
                    target_end_effector_position=np.array(input.target_position),
                    target_end_effector_orientation=np.array(input.target_orientation),
                )
        elif input.joint_positions:
            joint_velocities = np.array(input.joint_velocities) if input.joint_velocities else None
            joint_efforts = np.array(input.joint_efforts) if input.joint_efforts else None
            actions = ArticulationAction(
                joint_positions=np.array(input.joint_positions),
                joint_velocities=joint_velocities,
                joint_efforts=joint_efforts,
            )

        self.logger.info(f"Franka Actions: {actions}")
        if actions:
            self.franka_articulation_controller.apply_action(actions)

    @staticmethod
    def new_instance(config: FrankaConfig, franka: Robot):
        """Create a new FrankaSubscriber instance from configuration.

        Args:
            config: Franka configuration object
            franka: Franka robot instance
        """
        if not config.topic_ctrl or not config.topic_ctrl.name:
            return None

        return FrankaSubscriber(
            franka=franka,
            ik=config.ik,
            prim_path=config.prim_path,
            topic=config.topic_ctrl.name,
            period=config.topic_ctrl.period,
            domain_id=config.topic_ctrl.domain_id,
        )
