from simulation.configs.config import (
    Config,
    FrankaConfig,
    RoomCameraConfig,
    TargetConfig,
    UltraSoundConfig,
    WristCameraConfig,
)
from workflows.robotic_ultrasound.scripts.utils.assets import robotic_ultrasound_assets as rus_assets

config = Config(
    main_usd_path=rus_assets.basic,
    room_camera=RoomCameraConfig(prim_path="/RoomCamera", enabled=True),
    wrist_camera=WristCameraConfig(prim_path="/Franka/panda_hand/geometry/realsense/realsense_camera", enabled=True),
    franka=FrankaConfig(prim_path="/Franka", ik=False, auto_pos=False, enabled=True),
    target=TargetConfig(prim_path="/Target", auto_pos=False, enabled=True),
    ultrasound=UltraSoundConfig(prim_path="/Target", enabled=True),
)
