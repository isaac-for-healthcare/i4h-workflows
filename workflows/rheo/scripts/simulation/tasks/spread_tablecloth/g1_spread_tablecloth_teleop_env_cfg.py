import isaaclab.utils.math as math_utils
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode, XrCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.array import convert_to_torch
from isaaclab_arena_g1.g1_env.mdp import g1_events as g1_events_mdp
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action import G1DecoupledWBCPinkAction
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    LEFT_WRIST_POS_END_IDX,
    LEFT_WRIST_POS_START_IDX,
    LEFT_WRIST_QUAT_END_IDX,
    LEFT_WRIST_QUAT_START_IDX,
    RIGHT_WRIST_POS_END_IDX,
    RIGHT_WRIST_POS_START_IDX,
    RIGHT_WRIST_QUAT_END_IDX,
    RIGHT_WRIST_QUAT_START_IDX,
)
from teleop_devices.motion_controllers import MotionControllersTeleopDevice

from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action_cfg import (  # isort: skip
    G1DecoupledWBCPinkActionCfg,
)

from .h2_spread_tablecloth_env_cfg import G1SpreadTableclothEnvCfg


class G1SpreadTableclothFixedLegsWBCPinkAction(G1DecoupledWBCPinkAction):
    """Spread-tablecloth teleop action with lower body fixed in a standing pose."""

    _FIXED_LEG_WAIST_JOINTS = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    _FIXED_BASE_HEIGHT_CMD = 0.765

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        joint_names = self._asset.data.joint_names
        self._fixed_leg_waist_joint_ids = [joint_names.index(name) for name in self._FIXED_LEG_WAIST_JOINTS]

    def process_actions(self, actions):
        actions_fixed = actions.clone()

        # Convert XR wrist positions from world frame into the robot root frame
        # expected by the upper-body PINK IK controller.
        root_pos_w = convert_to_torch(self._asset.data.root_link_pos_w, device=self.device)
        root_quat_w = convert_to_torch(self._asset.data.root_link_quat_w, device=self.device)

        left_wrist_pos_w = actions_fixed[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX]
        right_wrist_pos_w = actions_fixed[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX]
        left_wrist_quat_w = actions_fixed[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX]
        right_wrist_quat_w = actions_fixed[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX]

        # The G1 WBC action buffer still stores wrist quaternions as wxyz, but
        # IsaacLab quaternion math utilities now operate on xyzw.
        left_wrist_quat_w_xyzw = math_utils.convert_quat(left_wrist_quat_w, to="xyzw")
        right_wrist_quat_w_xyzw = math_utils.convert_quat(right_wrist_quat_w, to="xyzw")

        actions_fixed[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX] = math_utils.quat_apply_inverse(
            root_quat_w,
            left_wrist_pos_w - root_pos_w,
        )
        actions_fixed[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX] = math_utils.quat_apply_inverse(
            root_quat_w,
            right_wrist_pos_w - root_pos_w,
        )
        left_wrist_quat_b_xyzw = math_utils.quat_mul(
            math_utils.quat_inv(root_quat_w),
            left_wrist_quat_w_xyzw,
        )
        right_wrist_quat_b_xyzw = math_utils.quat_mul(
            math_utils.quat_inv(root_quat_w),
            right_wrist_quat_w_xyzw,
        )
        actions_fixed[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX] = math_utils.convert_quat(
            left_wrist_quat_b_xyzw, to="wxyz"
        )
        actions_fixed[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX] = math_utils.convert_quat(
            right_wrist_quat_b_xyzw, to="wxyz"
        )

        nav_start = -self.navigate_cmd_dim - self.base_height_cmd_dim - self.torso_orientation_rpy_cmd_dim
        nav_end = -self.base_height_cmd_dim - self.torso_orientation_rpy_cmd_dim
        base_start = -self.base_height_cmd_dim - self.torso_orientation_rpy_cmd_dim
        base_end = -self.torso_orientation_rpy_cmd_dim
        torso_start = -self.torso_orientation_rpy_cmd_dim

        actions_fixed[:, nav_start:nav_end] = 0.0
        actions_fixed[:, base_start:base_end] = self._FIXED_BASE_HEIGHT_CMD
        actions_fixed[:, torso_start:] = 0.0

        super().process_actions(actions_fixed)
        self._processed_actions[:, self._fixed_leg_waist_joint_ids] = 0.0


@configclass
class G1SpreadTableclothFixedLegsWBCPinkActionCfg(G1DecoupledWBCPinkActionCfg):
    """Config for spread-tablecloth teleop action with fixed lower body."""

    class_type: type[ActionTerm] = G1SpreadTableclothFixedLegsWBCPinkAction


@configclass
class TeleopActionsCfg:
    """23-D WBC+PINK action for Meta Quest teleoperation."""

    g1_action: ActionTermCfg = G1SpreadTableclothFixedLegsWBCPinkActionCfg(asset_name="robot", joint_names=[".*"])


@configclass
class G1SpreadTableclothTeleopEnvCfg(G1SpreadTableclothEnvCfg):
    """Meta Quest teleoperation variant of the spread-tablecloth environment."""

    actions: TeleopActionsCfg = TeleopActionsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -1.0),
        anchor_rot=(0.70711, 0.0, 0.0, -0.70711),
    )

    def __post_init__(self):
        super().__post_init__()

        self.sim.render_interval = 2
        self.episode_length_s = 300.0

        self.events.reset_wbc_policy = EventTermCfg(
            func=g1_events_mdp.reset_decoupled_wbc_pink_policy,
            mode="reset",
        )

        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        mc = MotionControllersTeleopDevice(sim_device=self.sim.device)
        self.teleop_devices = mc.get_teleop_device_cfg(
            xr_cfg=self.xr,
            use_trocar_retargeter=False,
            use_tablecloth_retargeter=True,
        )
