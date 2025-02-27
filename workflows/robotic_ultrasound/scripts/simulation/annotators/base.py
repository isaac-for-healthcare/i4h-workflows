from typing import List

import omni.usd
from robotic_ultrasound.scripts.rti_dds.publisher import Publisher
from robotic_ultrasound.scripts.rti_dds.subscriber import Subscriber
from robotic_ultrasound.scripts.simulation.utils.ratelimit import add_physx_step_callback, remove_physx_callback

__all__ = ["Annotator"]

class Annotator:
    """Base class for simulation annotators that handle DDS communication.

    This class provides the basic infrastructure for creating annotators that can
    publish and subscribe to DDS topics in the simulation environment. It manages
    the lifecycle of publishers and subscribers, ensuring proper initialization
    and cleanup.

    Args:
        name: Unique identifier for the annotator
        prim_path: USD path to the sensor primitive in the stage
        publishers: List of DDS publishers to manage
        subscribers: List of DDS subscribers to manage
    """

    def __init__(
        self,
        name: str,
        prim_path: str,
        publishers: List[Publisher] | None = None,
        subscribers: List[Subscriber] | None = None,
    ):
        """Initialize the annotator with the given name, prim path, and DDS endpoints."""
        self.name = name
        self.prim_path = prim_path
        self.stage = omni.usd.get_context().get_stage()
        self.sensor_prim = self.stage.GetPrimAtPath(prim_path)
        self.publishers = [p for p in publishers if p is not None] if publishers else []
        self.subscribers = [s for s in subscribers if s is not None] if subscribers else []

    def start(self, world) -> None:
        """Start all publishers and subscribers with rate-limited callbacks.

        This method registers each publisher and subscriber with the physics
        simulation loop at their specified rates.

        Args:
            world: The simulation world object that provides the physics context
        """
        for publisher in self.publishers:
            add_physx_step_callback(f"{publisher.topic}", publisher.period, publisher.write, world)
        for subscriber in self.subscribers:
            add_physx_step_callback(f"{subscriber.topic}", subscriber.period, subscriber.read, world)
            subscriber.start()

    def stop(self, world):
        """Stop all publishers and subscribers and cleanup their callbacks.

        This method removes all registered callbacks from the physics simulation
        loop and stops the subscribers.

        Args:
            world: The simulation world object that provides the physics context
        """
        for publisher in self.publishers:
            remove_physx_callback(f"{publisher.topic}", world)
        for subscriber in self.subscribers:
            remove_physx_callback(f"{subscriber.topic}", world)
            subscriber.stop()
