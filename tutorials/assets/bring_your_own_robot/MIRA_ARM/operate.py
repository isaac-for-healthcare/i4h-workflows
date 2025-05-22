import carb
import omni.kit.app
import omni
from pxr import UsdPhysics
from omni.isaac.core.prims import XFormPrim

# --- Robot prims and joint paths ---
robot_base = XFormPrim("/World/A5_GUI_MODEL")

LJ_PATHS = [
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_L654321/LJ1/LJ1_joint",
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_L654321/ASM_L65432/LJ2/LJ2_joint",
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_L654321/ASM_L65432/ASM_L6543/LJ3/LJ3_joint",
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_L654321/ASM_L65432/ASM_L6543/ASM_L654/LJ4/LJ4_joint",
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_L654321/ASM_L65432/ASM_L6543/ASM_L654/ASM_L65/LJ5/LJ5_joint",
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_L654321/ASM_L65432/ASM_L6543/ASM_L654/ASM_L65/ASM_L61/LJ6/LJ6_1_joint"
]
RJ_PATHS = [
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_R654321/RJ1/RJ1_joint",
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_R654321/ASM_R65432/RJ2/RJ2_joint",
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_R654321/ASM_R65432/ASM_R6543/RJ3/RJ3_joint",
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_R654321/ASM_R65432/ASM_R6543/ASM_R654/RJ4/RJ4_joint",
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_R654321/ASM_R65432/ASM_R6543/ASM_R654/ASM_R65/RJ5/RJ5_joint",
    "/World/A5_GUI_MODEL/A5_GUI_MODEL_001/ASM_R654321/ASM_R65432/ASM_R6543/ASM_R654/ASM_R65/ASM_R6/RJ6/RJ6_joint"
]

stage = omni.usd.get_context().get_stage()
LJ_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in LJ_PATHS]
RJ_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in RJ_PATHS]

# --- Initial joint positions ---
left_pose = [0.0] * 6
right_pose = [0.0] * 6

# --- Key mapping (IsaacLab style) ---
KEY_MAP = {
    # Left arm
    "W": ("left", 0, 0.1),   # X+
    "S": ("left", 0, -0.1),  # X-
    "A": ("left", 1, 0.1),   # Y+
    "D": ("left", 1, -0.1),  # Y-
    "Q": ("left", 2, 0.1),   # Z+
    "E": ("left", 2, -0.1),  # Z-
    "Z": ("left", 3, 0.1),   # elbow+
    "X": ("left", 3, -0.1),  # elbow-
    "C": ("left", 4, 0.1),   # roll+
    "V": ("left", 4, -0.1),  # roll-
    "B": ("left", 5, 0.1),   # gripper+
    "N": ("left", 5, -0.1),  # gripper-
    # Right arm
    "I": ("right", 0, 0.1),  # X+
    "K": ("right", 0, -0.1), # X-
    "J": ("right", 1, 0.1),  # Y+
    "L": ("right", 1, -0.1), # Y-
    "U": ("right", 2, 0.1),  # Z+
    "O": ("right", 2, -0.1), # Z-
    "UP": ("right", 3, 0.1),     # elbow+
    "DOWN": ("right", 3, -0.1),  # elbow-
    "RIGHT": ("right", 4, 0.1),  # roll+
    "LEFT": ("right", 4, -0.1),  # roll-
    "Y": ("right", 5, 0.1),      # gripper+
    "M": ("right", 5, -0.1),     # gripper-
}

def on_update(dt):
    # Apply joint positions to robot
    for i, api in enumerate(LJ_apis):
        api.GetTargetPositionAttr().Set(left_pose[i])
    for i, api in enumerate(RJ_apis):
        api.GetTargetPositionAttr().Set(right_pose[i])

def on_keyboard_event(event, *args):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        key = event.input.name
        if key in KEY_MAP:
            arm, idx, delta = KEY_MAP[key]
            if arm == "left":
                left_pose[idx] += delta
            else:
                right_pose[idx] += delta
        print(f"Key pressed: {key} | left_pose: {left_pose} | right_pose: {right_pose}")
    return True

input_interface = carb.input.acquire_input_interface()
keyboard = omni.appwindow.get_default_app_window().get_keyboard()
keyboard_sub = input_interface.subscribe_to_keyboard_events(
    keyboard,
    lambda event, *args: on_keyboard_event(event, *args)
)

update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update)

print("Keyboard control active!\nLeft arm: W/S(X), A/D(Y), Q/E(Z), Z/X(elbow), C/V(roll), B/N(gripper)\nRight arm: I/K(X), J/L(Y), U/O(Z), UP/DOWN(elbow), RIGHT/LEFT(roll), Y/M (gripper)\nAvoid viewport focus to ensure key events are captured.\nCurrent joint values will be printed in the console for debugging.\n")

# To clean up (unsubscribe) when done, you can call:
# keyboard_sub.unsubscribe()
# update_sub.unsubscribe()
