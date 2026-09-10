# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the arm-borne catheter embodiment.

The embodiment modules pull in ``isaaclab.sim`` and cannot be imported without
Isaac Sim, so these read the source the way ``test_kuka_embodiment`` does. The
contract worth pinning is which physics backend the arm selects: an articulation
in the model means the rod-only solver is no longer sufficient, and losing
``rigid_bodies_enabled`` would leave the arm with no rigid integrator while the
scene still looked correct.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import pytest

import yaml

ARENA = Path(__file__).parents[1] / "i4h_arena"
SOURCE = ARENA / "embodiments" / "franka_catheter.py"
ARM_SCENE = ARENA / "scenes" / "endoluminal_navigation_arm.py"
EMBODIMENT_MANIFEST = ARENA / "embodiments" / "manifest" / "franka_catheter.yaml"
SCENE_MANIFEST = ARENA / "scenes" / "manifest" / "endoluminal_navigation_arm.yaml"

#: The recorded state contract. Catheter and C-arm lead so a recording made with
#: the arm keeps the armless scene's column meanings, and the arm is appended.
EXPECTED_STATE_NAMES = [
    "insertion_m",
    "rotation_rad",
    "tip_bend_rad",
    "carm_orbit_rad",
    *[f"arm.panda_joint{index}" for index in range(1, 8)],
]


PLAIN_CATHETER = ARENA / "embodiments" / "catheter.py"
BASE_SCENE = ARENA / "scenes" / "endoluminal_navigation.py"

#: Slider the fluoroscopy view offers for insertion speed.
INSERTION_CONTROL = "catheter_insertion_speed_mps"


def source() -> str:
    return SOURCE.read_text()


def module_constant(name: str) -> Any:
    """Read one module-level constant from the embodiment, without importing it."""
    return ast.literal_eval(module_constant_node(name))


def module_constant_expr(name: str) -> str:
    """The same constant as written, for values that are not literals."""
    return ast.unparse(module_constant_node(name))


def module_constant_node(name: str) -> ast.expr:
    for statement in ast.parse(source()).body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(getattr(target, "id", None) == name for target in statement.targets):
            return statement.value
    raise AssertionError(f"{name} not found in {SOURCE.name}")


def class_default(path: Path, class_name: str, field: str) -> float:
    """Read one annotated class-level default without importing the module.

    These modules pull in ``isaaclab``, so importing them needs a simulator the
    light test environment does not have.
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and getattr(statement.target, "id", None) == field:
                return ast.literal_eval(statement.value)
    raise AssertionError(f"{class_name}.{field} not found in {path.name}")


def insertion_slider() -> dict[str, float]:
    """The fluoroscopy velocity slider's keyword arguments."""
    for node in ast.walk(ast.parse(BASE_SCENE.read_text())):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != "SensorSliderSpec":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        control = keywords.get("control")
        if control is not None and ast.literal_eval(control) == INSERTION_CONTROL:
            return {name: ast.literal_eval(value) for name, value in keywords.items() if name != "control"}
    raise AssertionError(f"no SensorSliderSpec driving {INSERTION_CONTROL} in {BASE_SCENE.name}")


def test_the_velocity_slider_cannot_ask_for_more_than_the_arm_allows() -> None:
    """Or its top end would move the handle without moving the catheter.

    The action term clamps whatever the slider sends, so a slider whose maximum
    outruns that ceiling is just a lie told in millimetres per second.
    """
    slider = insertion_slider()
    ceiling = class_default(SOURCE, "ArmDrivenCatheterActionCfg", "max_insertion_velocity_mps")

    assert slider["maximum"] * slider["scale"] == pytest.approx(ceiling)


def test_both_catheter_drives_share_one_insertion_ceiling() -> None:
    """One slider feeds both scenes, so a mismatch silently clips one of them."""
    arm = class_default(SOURCE, "ArmDrivenCatheterActionCfg", "max_insertion_velocity_mps")
    plain = class_default(PLAIN_CATHETER, "CatheterVelocityActionCfg", "max_insertion_velocity_mps")

    assert arm == pytest.approx(plain)


def test_the_insertion_ceiling_stays_inside_the_contact_budget() -> None:
    """Above this the wire steps further than the vessel contact can catch.

    Physics runs at 240 Hz against a 0.5 mm particle radius. Spending more than
    half a radius per step lets the tip cross a vessel wall between solves,
    which shows up as a catheter that escapes the anatomy rather than as an
    error.
    """
    physics_hz = 240.0
    particle_radius_m = 0.0005

    ceiling = class_default(SOURCE, "ArmDrivenCatheterActionCfg", "max_insertion_velocity_mps")

    assert ceiling / physics_hz <= 0.5 * particle_radius_m


def test_the_embodiment_extends_the_plain_catheter() -> None:
    """Reimplementing it would fork the centerline alignment and the goal."""
    module = ast.parse(source())
    classes = {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}

    assert "FrankaCatheterEmbodiment" in classes
    bases = {base.id for base in classes["FrankaCatheterEmbodiment"].bases if isinstance(base, ast.Name)}
    assert "CatheterEmbodiment" in bases


def test_the_arm_switches_the_scene_onto_the_coupled_solver() -> None:
    """The rod solver integrates particles only, so an arm needs MJWarp beside it."""
    assert "rigid_bodies_enabled = True" in source()


def test_the_flange_is_named_as_the_catheter_drive_body() -> None:
    """Which turns the contact two-way, so the arm feels the wire it carries.

    Named rather than indexed, and named with the same constant the action term
    servos, so the body the reaction lands on cannot drift away from the body
    the drive is bolted to.
    """
    assert "drive_body_name = FRANKA_FLANGE_BODY" in source()


def test_the_scene_cfg_carries_the_arm() -> None:
    assert "robot" in source()
    assert "make_franka_panda_catheter_cfg" in source()


def test_the_articulation_uses_backend_agnostic_schemas() -> None:
    """This scene runs Newton, so the arm must not be described with PhysX
    schemas. ``embodiments.franka`` builds its configs from
    ``isaaclab_physx.sim.schemas``, which is why the config is rebuilt here
    instead of imported -- the import would look like harmless reuse."""
    text = source()

    assert "sim_utils.RigidBodyPropertiesCfg" in text
    assert "sim_utils.ArticulationRootPropertiesCfg" in text
    assert "isaaclab_physx" not in text
    assert "PhysxRigidBodyPropertiesCfg" not in text


def test_the_arm_is_not_left_to_sag_onto_the_patient() -> None:
    """A position-servo'd arm that sags moves the hand without being asked to,
    and the hand carries the wire's entry point, so the sag drags the wire off
    the access site. It measured 16 cm.

    The fix has to be at the scene, not the spawn config: ``disable_gravity``
    is a PhysX-namespace attribute and this scene runs Newton, which has no
    per-body gravity. Authoring it looks like a fix and does nothing.
    """
    assert "disable_gravity=" not in source()
    assert "env_cfg.sim.gravity = (0.0, 0.0, 0.0)" in ARM_SCENE.read_text()


def test_the_drive_is_mounted_on_the_hand_that_grips_it() -> None:
    """The drive unit is clamped in the fingers, so the hand is the body that
    carries it and the body the wire's reaction has to land on. The bundle's
    Panda this replaced had no hand, only a bare ``TCP`` tool plate."""
    assert 'FRANKA_FLANGE_BODY = "panda_hand"' in source()
    assert "flange_body_name: str = FRANKA_FLANGE_BODY" in source()


def test_the_hand_holds_the_drive_unit_rather_than_hanging_open() -> None:
    """An arm posed beside a wire it is not touching is the whole reason this
    asset was swapped in, so the fingers have to be driven closed on the drive
    unit's barrel -- and driven stiffly, or the wire's reaction eases them
    open and the hand reads as dropping the drive."""
    text = source()

    assert 'joint_names_expr=["panda_finger_joint.*"]' in text
    assert '"panda_finger_joint1": _DRIVE_BARREL_HALF_WIDTH_M' in text
    assert '"panda_finger_joint2": _DRIVE_BARREL_HALF_WIDTH_M' in text


def test_the_arm_goes_to_the_introducer_instead_of_holding_where_it_spawned() -> None:
    """The home pose is a starting point, and the access site moves with the
    patient twin. Latching whatever pose the arm happened to reach is what left
    the hand holding nothing 30 cm from the wire, so the pose to hold has to be
    resolved from the rod."""
    text = source()

    assert "_aim_at_the_introducer" in text
    assert "root_pos - approach * (_SHEATH_LENGTH_M + _HAND_TO_GRIP_M)" in text


def test_the_hand_stays_outside_the_patient() -> None:
    """The wire's proximal particle is 4.5 cm *inside* the body on this twin,
    so gripping it directly closed the fingers under the skin. The sheath is
    what keeps the drive out of the tissue, and it has to be long enough to
    clear the surface once the tilt is taken into account."""
    lift = module_constant("_SHEATH_LENGTH_M") * math.sin(math.radians(45.0))

    assert lift > 0.045


def test_the_wire_is_not_dragged_along_during_the_approach() -> None:
    """The introducer carries the wire's entry rigidly with the flange, and the
    flange crosses the room on its way to the access site. Reading transport
    before it arrives would tow the wire behind it."""
    body = source().partition("def apply_actions")[2].partition("def reset")[0]

    assert "if self._parked else torch.zeros_like(flange_pos)" in body


def test_the_drive_is_tilted_out_of_the_vessel_it_feeds() -> None:
    """Pointing the hand straight down the wire is not reachable from a cart
    beside the table -- no pose within the joint limits reaches it, at any base
    yaw. The tilt is also the real femoral puncture angle."""
    tilt = module_constant_expr("_INTRODUCER_TILT_RAD")

    assert tilt == "math.radians(45.0)"


def test_the_home_pose_is_clear_of_every_joint_limit() -> None:
    """The pose this replaced sat ``panda_joint2`` at 1.740 against a 1.7628
    limit, so the shoulder was against its stop before the first step and the
    IK had nowhere left to go."""
    limits = {
        "panda_joint1": (-2.8973, 2.8973),
        "panda_joint2": (-1.7628, 1.7628),
        "panda_joint3": (-2.8973, 2.8973),
        "panda_joint4": (-3.0718, -0.0698),
        "panda_joint5": (-2.8973, 2.8973),
        "panda_joint6": (-0.0175, 3.7525),
        "panda_joint7": (-2.8973, 2.8973),
    }
    # The finger entries are named constants rather than literals, so the dict
    # is read key by key instead of evaluated whole.
    node = module_constant_node("FRANKA_HOME_JOINT_POS")
    home = {}
    for key, value in zip(node.keys, node.values):
        try:
            home[ast.literal_eval(key)] = ast.literal_eval(value)
        except ValueError:
            continue

    assert set(home) == set(limits)
    for joint, (low, high) in limits.items():
        assert min(home[joint] - low, high - home[joint]) > 0.1, joint


def test_the_grip_is_not_closed_all_the_way() -> None:
    """Fingers touching each other read as an empty hand. They close on a
    barrel, and the fingers' own travel is 0 to 40 mm, so the half-width has
    to sit strictly inside that."""
    assert 0.0 < module_constant("_DRIVE_BARREL_HALF_WIDTH_M") < 0.04


def test_the_asset_comes_from_a_reachable_path() -> None:
    """``embodiments.franka`` builds FRANKA_PANDA_CFG from an Isaac Lab nucleus
    path that 404s against the 6.0 asset layout, which is why that config is
    unused here. Reintroducing it would fail at scene spawn, not at import."""
    text = source()

    assert "FRANKA_PANDA_USD = FRANKA_PANDA_HAND_USD" in text
    assert "from i4h_arena.assets.constants import FRANKA_PANDA_HAND_USD" in text
    assert "ISAACLAB_NUCLEUS_DIR" not in text


def test_the_franka_is_pinned_to_the_layout_that_still_has_it() -> None:
    """The hand is why this asset was chosen, and it is the 6.0 layout dropping
    the asset that sent this workflow to a gripperless Panda before. Floating
    the version would quietly take the hand away again."""
    constants = (ARENA / "assets" / "constants.py").read_text()

    assert "/Assets/Isaac/5.0/Isaac/" in constants
    assert "FRANKA_PANDA_HAND_USD" in constants
    assert "panda_instanceable.usd" in constants


def test_the_arm_is_spawned_without_the_newton_repair_overlay() -> None:
    """That overlay exists to ground a floating root and turn reversed wrist
    joints around, both faults of the Panda this replaced. This asset ships a
    world-grounded ``rootJoint``, and running the anchor pass over one that is
    already grounded would author a second, redundant world joint."""
    text = source()

    assert "panda_usd_repair" not in text
    # The overlay was substituted for ``usd_path`` by a custom spawner, so its
    # absence is what says the raw asset is now spawned as shipped.
    assert "_spawn_panda_for_newton" not in text


def test_the_centerline_facts_survive_the_config_swap() -> None:
    """The parent resolves the access site and the C-arm isocenter from the
    patient centerline and writes them onto the configs this embodiment then
    replaces. Both have to be carried across: losing the isocenter moves the
    C-arm off the patient, which still renders, just wrongly."""
    embodiment = source().partition("class FrankaCatheterEmbodiment")[2]
    body = embodiment.partition("def __init__")[2].partition("    @property")[0]

    assert "isocenter = self.action_config.carm_orbit.isocenter_world_m" in body
    assert "self.action_config.carm_orbit.isocenter_world_m = isocenter" in body
    assert "access_site = self.scene_config.catheter_root.init_state.pos" in body
    assert "self.scene_config.catheter_root.init_state.pos = access_site" in body


def test_the_drive_feeds_along_the_wire_rather_than_a_configured_axis() -> None:
    """Feeding along a fixed axis is what made the rod fold back on itself.

    The vessel curves away from the introducer's direction, so a root pushed
    along that axis leaves the lumen -- 44 mm outside it by 300 mm of depth on
    the s0011 twin. The guard is that no axis is configured at all, so there is
    nothing to go stale when the wire rounds a bend.
    """
    assert "insertion_axis_world" not in source()
    assert "tangent" in source().partition("def apply_actions")[2].partition("def reset")[0]


def test_the_arm_is_its_own_scene_rather_than_a_flag() -> None:
    """The arm changes facts the scene manifest owns: the embodiment gains seven
    joints and the model gains an articulation. A CLI flag would hide both from
    lint, which only ever reads the manifest."""
    scene = ARM_SCENE.read_text()

    assert "FrankaCatheterEmbodiment" in scene
    assert "with_arm" not in scene


def test_the_scene_manifest_declares_the_composite_embodiment() -> None:
    manifest = yaml.safe_load(SCENE_MANIFEST.read_text())

    assert manifest["embodiment"] == "franka_catheter"
    assert manifest["impl"] == "i4h_arena.scenes.endoluminal_navigation_arm:EndoluminalNavigationArmScene"
    # The arm is servo'd from the same two catheter actions, not commanded, so
    # the action space must not grow.
    assert manifest["dof"] == 4
    assert manifest["action_space"] == "catheter_carm_velocity"


def test_the_embodiment_manifest_matches_the_recorded_state_order() -> None:
    """The LeRobot converter only applies these names when the count matches the
    recorded width, and otherwise falls back to positional names with a warning.
    A silent disagreement here produces mislabelled datasets, so the order is
    pinned rather than derived."""
    manifest = yaml.safe_load(EMBODIMENT_MANIFEST.read_text())

    assert manifest["name"] == "franka_catheter"
    assert manifest["state_names"] == EXPECTED_STATE_NAMES
    assert manifest["joint_names"] == EXPECTED_STATE_NAMES


def test_the_manifest_arm_names_match_the_usd_joint_names() -> None:
    """The provider builds these from ``find_joints``, so a tidier alias in the
    manifest would silently relabel columns that hold Franka joints."""
    manifest = yaml.safe_load(EMBODIMENT_MANIFEST.read_text())
    arm_names = [name for name in manifest["state_names"] if name.startswith("arm.")]

    assert arm_names == [f"arm.panda_joint{index}" for index in range(1, 8)]
    assert 'FRANKA_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))' in source()


def test_the_action_names_stay_identical_to_the_armless_catheter() -> None:
    """The arm adds no actions, so datasets recorded either way share an action
    layout and a checkpoint trained on one can be rolled out on the other."""
    composite = yaml.safe_load(EMBODIMENT_MANIFEST.read_text())
    plain = yaml.safe_load((ARENA / "embodiments" / "manifest" / "catheter.yaml").read_text())

    assert composite["action_names"] == plain["action_names"]


def test_the_manifest_does_not_restate_scene_owned_facts() -> None:
    """`dof`, `action_space` and `gripper` belong to the scene: the same arm can
    be mounted under different controllers."""
    manifest = yaml.safe_load(EMBODIMENT_MANIFEST.read_text())

    for scene_owned in ("dof", "action_space", "gripper"):
        assert scene_owned not in manifest


def test_the_provider_concatenates_in_the_declared_order() -> None:
    """Catheter, then C-arm, then arm -- matching the manifest above."""
    provider = source().partition("class ArmCatheterCArmJointStateProvider")[2]
    concat = provider.partition("np.concatenate((")[2]

    assert concat.startswith("catheter.pos, carm.pos, arm.pos")


def test_the_wire_is_placed_from_the_realized_flange_not_the_request() -> None:
    """What putting the arm in the loop buys, now that feed is a roller command.

    Insertion no longer depends on arm travel, so the arm's hold on the scene is
    positional: the wire's entry point is computed from the flange pose read
    back out of simulation. Using the commanded target instead would leave the
    wire where the arm was *asked* to be, and an arm at a joint limit or held by
    contact would stop mattering at all.
    """
    body = source().partition("def apply_actions")[2].partition("def reset")[0]

    # The realized pose is what gets transported into the wire ...
    assert "self._introducer.transport(flange_pos)" in body
    # ... and the wire's own state is what it is transported from.
    assert "self._proximal_frame()" in body
    assert "self._asset.place_proximal(root_target, quat_target" in body


def test_insertion_does_not_depend_on_arm_travel() -> None:
    """The roller drive's defining property.

    Deriving feed from flange travel capped the reachable insertion at the arm's
    own straight-line reach, which is far short of the rod's 0.4 m. The guard is
    that the commanded action reaches the rollers directly.
    """
    body = source().partition("def apply_actions")[2].partition("def reset")[0]

    assert "self._introducer.advance(" in body
    assert "self._processed_actions[:, 0], self._processed_actions[:, 1], dt" in body
    # The abandoned gearbox differenced the flange pose to get a feed rate.
    assert "_drive.compute" not in body
