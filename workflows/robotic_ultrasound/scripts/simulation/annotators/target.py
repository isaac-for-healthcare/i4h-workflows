from typing import Any

from omni.isaac.core.prims import XFormPrim
from robotic_ultrasound.scripts.rti_dds.publisher import Publisher
from robotic_ultrasound.scripts.rti_dds.schemas.target_ctrl import TargetCtrlInput
from robotic_ultrasound.scripts.rti_dds.schemas.target_info import TargetInfo
from robotic_ultrasound.scripts.rti_dds.subscriber import Subscriber
from robotic_ultrasound.scripts.simulation.configs.config import TargetConfig


class TargetPublisher(Publisher):
    """Publisher for target object state information.

    This class handles publishing target position and orientation through DDS topics.
    It tracks the target object's transform in the simulation world.

    Args:
        prim_path: USD path to target object
        topic: DDS topic name
        period: Publishing period in seconds
        domain_id: DDS domain identifier
    """

    def __init__(self, prim_path: str, topic: str, period: float, domain_id):
        """Initialize the target publisher."""
        super().__init__(topic, TargetInfo, period, domain_id)
        self.prim_path = prim_path

    def produce(self, dt: float, sim_time: float) -> Any:
        """Produce target state information for publishing.

        Args:
            dt: Time delta since last physics step
            sim_time: Current simulation time

        Returns:
            TargetInfo: Target state information including position and orientation,
                refer to rti_dds.schemas.target_info.TargetInfo.
        """
        target = XFormPrim(prim_path=self.prim_path)
        position, orientation = target.get_world_pose()

        output = TargetInfo()
        output.position = position.tolist()
        output.orientation = orientation.tolist()
        return output

    @staticmethod
    def new_instance(config: TargetConfig):
        """Create a new TargetPublisher instance from configuration.

        Args:
            config: Target configuration object
        """
        if not config.topic_info or not config.topic_info.name:
            return None

        return TargetPublisher(
            prim_path=config.prim_path,
            topic=config.topic_info.name,
            period=config.topic_info.period,
            domain_id=config.topic_info.domain_id,
        )


class TargetSubscriber(Subscriber):
    """Subscriber for target object control.

    This class handles target control input.
    It updates the target object's position and orientation in the simulation.

    Args:
        prim_path: USD path to target object
        topic: DDS topic name
        period: Subscription period in seconds
        domain_id: DDS domain identifier
    """

    def __init__(self, prim_path: str, topic: str, period: float, domain_id):
        """Initialize the target subscriber."""
        super().__init__(topic, TargetCtrlInput, period, domain_id)
        self.prim_path = prim_path

    def consume(self, input: TargetCtrlInput) -> None:
        """Consume target control input and update target object's position and orientation.

        Args:
            input: Control input message containing target position and orientation,
                refer to rti_dds.schemas.target_ctrl.TargetCtrlInput.
        """
        print(f"Target:: Set Target New Position: {input.position}")

        target = XFormPrim(prim_path=self.prim_path)
        target.set_world_pose(position=input.position, orientation=input.orientation)

    @staticmethod
    def new_instance(config: TargetConfig):
        """Create a new TargetSubscriber instance from configuration.

        Args:
            config: Target configuration object
        """
        if not config.topic_ctrl or not config.topic_ctrl.name:
            return None

        return TargetSubscriber(
            prim_path=config.prim_path,
            topic=config.topic_ctrl.name,
            period=config.topic_ctrl.period,
            domain_id=config.topic_ctrl.domain_id,
        )
