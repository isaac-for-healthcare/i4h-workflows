import logging
from typing import List, Optional

import omni.replicator.core as rep
import omni.usd
from omni.kit.viewport.utility import get_active_viewport_window
from omni.replicator.core.scripts.utils import viewport_manager
from omni.syntheticdata import SyntheticData
from rti_dds.publisher import Publisher
from rti_dds.schemas.camera_ctrl import CameraCtrlInput
from rti_dds.schemas.camera_info import CameraInfo
from rti_dds.subscriber import Subscriber
from simulation.configs.config import CameraConfig

__all__ = ["CameraPublisher", "CameraSubscriber"]


class CameraPublisher(Publisher):
    """A publisher class for camera data in the simulation environment.

    This class handles publishing camera data (RGB or depth) from a virtual camera
    in the simulation to DDS topics. It supports different camera annotators and
    can publish both RGB and depth information.

    Args:
        annotator: Type of camera annotator ("rgb" or "distance_to_camera")
        prim_path: USD path to the camera primitive
        height: Height of the camera image in pixels
        width: Width of the camera image in pixels
        topic: DDS topic name for publishing camera data
        period: Period of the camera data in seconds
        domain_id: DDS domain identifier
    """

    def __init__(
        self, annotator: str, prim_path: str, height: int, width: int, topic: str, period: float, domain_id: int
    ) -> None:
        super().__init__(topic, CameraInfo, period, domain_id)

        self.annotator: str = annotator
        self.prim_path: str = prim_path
        self.sdg_iface = SyntheticData.Get()
        self.stage = omni.usd.get_context().get_stage()
        self.resolution: tuple[int, int] = (width, height)

        # Configure viewport
        _viewport_api = get_active_viewport_window().viewport_api
        _viewport_api.resolution = self.resolution
        _viewport_api._hydra_texture.camera_path = self.prim_path

        # Setup render product
        name = f"{self.prim_path.split('/')[-1]}_rp"
        rp = viewport_manager.get_render_product(self.prim_path, self.resolution, False, name)
        self.rp: str = rp.hydra_texture.get_render_product_path()
        self.annot = rep.AnnotatorRegistry.get_annotator(self.annotator)
        self.annot.attach(self.rp)

    def produce(self, dt: float, sim_time: float) -> CameraInfo:
        """Produce camera data for publishing.

        Args:
            dt: Time delta since last physics step
            sim_time: Current simulation time

        Returns:
            CameraInfo object containing camera data and parameters,
                refer to rti_dds.schemas.camera_info.CameraInfo
        """
        prim = self.stage.GetPrimAtPath(self.prim_path)
        output = CameraInfo()
        output.focal_len = prim.GetAttribute("focalLength").Get()
        output.data = self.annot.get_data().tobytes()
        return output

    @staticmethod
    def new_instance(config: CameraConfig, rgb: bool = True) -> Optional["CameraPublisher"]:
        """Create a new CameraPublisher instance based on configuration.

        Args:
            config: Camera configuration object
            rgb: If True, creates RGB camera publisher; if False, creates depth camera publisher
        """
        if rgb:
            if not config.topic_data_rgb or not config.topic_data_rgb.name:
                return None
            return CameraPublisher(
                annotator="rgb",
                prim_path=config.prim_path,
                height=config.height,
                width=config.width,
                topic=config.topic_data_rgb.name,
                period=config.topic_data_rgb.period,
                domain_id=config.topic_data_rgb.domain_id,
            )

        if not config.topic_data_depth or not config.topic_data_depth.name:
            return None
        return CameraPublisher(
            annotator="distance_to_camera",
            prim_path=config.prim_path,
            height=config.height,
            width=config.width,
            topic=config.topic_data_depth.name,
            period=config.topic_data_depth.period,
            domain_id=config.topic_data_depth.domain_id,
        )


class CameraSubscriber(Subscriber):
    """A subscriber class for camera control in the simulation environment.

    This class handles subscribing to camera control commands and updating
    camera parameters in the simulation accordingly.

    Args:
        prim_path: USD path to the camera primitive
        topic: DDS topic name for camera control
        period: Subscription period in seconds
        domain_id: DDS domain identifier
    """

    def __init__(self, prim_path: str, topic: str, period: float, domain_id: int) -> None:
        super().__init__(topic, CameraCtrlInput, period, domain_id)

        self.prim_path: str = prim_path
        self.current_focal_length: float = 20.0
        self.current_xyz: List[float] = [0.0, 0.0, 0.0]
        self.logger = logging.getLogger(__name__)

    def consume(self, input: CameraCtrlInput) -> None:
        """Process received camera control commands.

        Args:
            input: Camera control input containing new camera parameters,
                refer to rti_dds.schemas.camera_ctrl.CameraCtrlInput
        """
        if input.focal_len:
            self.logger.info(f"Camera:: Set New Focal Length: {input.focal_len}")
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.prim_path)
            prim.GetAttribute("focalLength").Set(input.focal_len)

    @staticmethod
    def new_instance(config: CameraConfig) -> Optional["CameraSubscriber"]:
        """Create a new CameraSubscriber instance based on configuration.

        Args:
            config: Camera configuration object
        """
        if not config.topic_ctrl or not config.topic_ctrl.name:
            return None

        return CameraSubscriber(
            prim_path=config.prim_path,
            topic=config.topic_ctrl.name,
            period=config.topic_ctrl.period,
            domain_id=config.topic_ctrl.domain_id,
        )
