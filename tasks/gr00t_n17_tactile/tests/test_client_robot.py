# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The robot adapters: teleop checkout lookup, DDS initialisation, the H2 adapter on injected fakes, the fake robot."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17_tactile.client import robot  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.client.robot import FakeRobot  # noqa: E402

# ── locating the teleop checkout ─────────────────────────────────────────────


def test_repository_root_is_the_workflow_checkout() -> None:
    root = robot.repository_root()
    assert (root / "third_party" / "setup.sh").is_file() and (root / "DESIGN.md").is_file()


def test_teleop_root_honours_the_env_override_and_names_setup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(robot.XR_TELEOPERATE_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="setup.sh"):
        robot.teleop_root()
    _make_teleop_checkout(tmp_path)
    assert robot.teleop_root() == tmp_path.resolve()
    monkeypatch.setattr(sys, "path", list(sys.path))
    robot.ensure_teleop_importable()
    robot.ensure_teleop_importable()
    assert sys.path.count(str(tmp_path.resolve())) == 1


def _make_teleop_checkout(root: Path) -> None:
    (root / "teleop" / "robot_control").mkdir(parents=True)
    (root / "teleop" / "robot_control" / "robot_arm.py").write_text("")


# ── the fake robot ───────────────────────────────────────────────────────────


def test_fake_robot_frames_and_first_order_tracking() -> None:
    bino = FakeRobot(frame_hw=(4, 6), lag=1.0)
    frames = bino.frames()
    assert frames["head"].shape == (4, 12, 3) and frames["head"][1, 1, 0] == 30 and frames["head"][1, 7, 0] == 40
    assert frames["left_wrist"][1, 1, 0] == 10 and frames["right_wrist"][1, 1, 0] == 20
    assert frames["head"][0, 0, 0] == 1 and bino.frames()["head"][0, 0, 0] == 2
    assert FakeRobot(frame_hw=(4, 6), binocular=False).frames()["head"].shape == (4, 6, 3)
    snap = bino.snapshot(hands=("left",))
    assert set(snap.hand_q) == {"left"} and snap.arm_q.shape == (14,) and snap.t_capture > 0
    target = np.arange(14, dtype=np.float64)
    bino.send_arm(target)
    assert bino.arm_q().tolist() == target.tolist() and len(bino.sent_arm) == 1
    bino.send_hand("left", np.ones(22))
    assert bino.hand_q("left").tolist() == [1.0] * 22 and bino.hand_q("right").tolist() == [0.0] * 22
    bino.go_home()
    assert bino.home_calls == 1 and bino.arm_q().tolist() == [0.0] * 14 and bino.hand_q("left").tolist() == [0.0] * 22
    bino.go_to_init(np.full(7, 0.1), np.full(7, 0.2), {"right": [0.3] * 22})
    assert bino.init_calls == 1 and bino.arm_q()[7] == 0.2 and bino.hand_q("right")[0] == np.float32(0.3)
    bino.close()
    assert bino.closed
    none = FakeRobot(arm=False, hands=())
    assert none.has_arm is False and none.hands == () and none.snapshot(("left",)).arm_q is None


# ── hand state and operator helpers ──────────────────────────────────────────


class _Err:
    def __init__(self, code: int) -> None:
        self.code = code


class _Hand:
    def __init__(self, err, degrees) -> None:
        self.err, self.degrees = err, degrees

    def get_joint_position_degree(self):
        if isinstance(self.err, Exception):
            raise self.err
        return self.err, self.degrees


def test_read_hand_state_accepts_both_sdk_error_styles_and_zero_fills_failures() -> None:
    degrees = [90.0] * 22
    assert robot.read_hand_state(None).tolist() == [0.0] * 22
    assert np.allclose(robot.read_hand_state(_Hand(0, degrees)), np.pi / 2)
    assert np.allclose(robot.read_hand_state(_Hand(_Err(0), degrees)), np.pi / 2)
    assert robot.read_hand_state(_Hand(_Err(3), degrees)).tolist() == [0.0] * 22
    assert robot.read_hand_state(_Hand(1, degrees)).tolist() == [0.0] * 22
    assert robot.read_hand_state(_Hand(RuntimeError("x"), degrees)).tolist() == [0.0] * 22
    assert robot.read_hand_state(_Hand(0, degrees)).dtype == np.float32


def test_init_and_home_with_the_countdown_stubbed(monkeypatch) -> None:
    monkeypatch.setattr(robot, "countdown", lambda *a, **k: None)
    fake = FakeRobot(lag=1.0)
    robot.init_and_hold(fake, None, None, 3)
    assert fake.init_calls == 0
    robot.init_and_hold(fake, {"left_arm": {"qpos": [0.1] * 7}, "left_ee": {"qpos": [0.2] * 22}}, np.full(7, 0.5), 0)
    assert fake.init_calls == 1 and fake.arm_q()[:7].tolist() == [0.1] * 7 and fake.arm_q()[7:].tolist() == [0.5] * 7
    assert fake.hand_q("left").tolist() == [np.float32(0.2)] * 22
    with pytest.raises(ValueError):
        robot.init_and_hold(fake, {"left_arm": {"qpos": [0.1] * 6}}, None, 0)
    robot.home_and_hold(fake, 0)
    assert fake.home_calls == 1 and fake.arm_q().tolist() == [0.0] * 14


# ── fakes for the teleop checkout and the Unitree SDK ────────────────────────


def _install_dds_fakes(monkeypatch, calls: dict, peers_from_discovery=None, discovery_error=False) -> None:
    channel = types.ModuleType("unitree_sdk2py.core.channel")

    def ChannelFactoryInitialize(domain=0, networkInterface=None, networkConfig=None):  # noqa: N802,N803
        calls["init"] = (domain, networkInterface, networkConfig)
        calls.setdefault("order", []).append("dds")

    channel.ChannelFactoryInitialize = ChannelFactoryInitialize
    net = types.ModuleType("teleop.utils.unitree_dds_network")
    net.WORKSTATION_ROBOT_DDS_ADDRESS = "10.1.1.2"
    net.THOR_DDS_ADDRESS = "10.1.1.3"
    net.H2_BODY_DDS_ADDRESSES = ("10.1.1.4",)
    net.build_address_config = lambda address, peers, **kw: f"<xml local={address} peers={','.join(peers)} {kw}>"
    disc = types.ModuleType("teleop.utils.discover_h2_dds_peer")
    disc.DEFAULT_THOR_SSH_TARGET = "u@thor"

    class H2PeerDiscoveryError(Exception):
        pass

    def discover(*, ssh_target, source_ip, timeout_s):
        calls["discover"] = (ssh_target, source_ip, timeout_s)
        if discovery_error:
            raise H2PeerDiscoveryError("no spdp")
        return list(peers_from_discovery or [])

    disc.H2PeerDiscoveryError = H2PeerDiscoveryError
    disc.discover_h2_body_peers_via_ssh = discover
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel)
    monkeypatch.setitem(sys.modules, "teleop.utils.unitree_dds_network", net)
    monkeypatch.setitem(sys.modules, "teleop.utils.discover_h2_dds_peer", disc)


def _install_teleop_fakes(monkeypatch, calls: dict) -> None:
    """``H2_ArmController``, ``SharpaHandDDSClient`` and ``ROSImageClient`` that record how they are built."""

    class _Frame:
        def __init__(self, bgr) -> None:
            self.bgr = bgr

    class H2_ArmController:  # noqa: N801
        def __init__(
            self,
            motion_mode=False,
            simulation_mode=False,
            kp_low=None,
            kp_wrist=None,
            kd_low=None,
            kd_wrist=None,
            kp_head=None,
            kd_head=None,
            head_pitch_home=0.0,
        ) -> None:
            calls["arm"] = dict(
                motion_mode=motion_mode,
                head_pitch_home=head_pitch_home,
                kp_low=kp_low,
                kp_wrist=kp_wrist,
                kd_low=kd_low,
                kd_wrist=kd_wrist,
                kp_head=kp_head,
                kd_head=kd_head,
            )
            calls["order"].append("arm")
            self.arm_velocity_limit = 1.0
            self.commands: list = []

        def get_current_dual_arm_q(self):
            return np.arange(14, dtype=np.float64) / 10

        def ctrl_dual_arm(self, q, tauff):
            self.commands.append(("q", list(q), list(tauff)))

        def ctrl_dual_arm_go_home(self):
            self.commands.append("home")

    class SharpaHandDDSClient:
        def __init__(self, side: str) -> None:
            self.side = side
            self.commands: list = []
            calls.setdefault("hands", []).append(side)
            calls["order"].append(f"hand:{side}")

        def set_joint_position(self, q, blocking):
            self.commands.append((list(q), blocking))

        def get_joint_position_degree(self):
            return 0, [45.0] * 22

        def go_neutral(self):
            self.commands.append("neutral")

    class ROSImageClient:
        def __init__(self, warmup_timeout, head_mode, cyclonedds_uri=None) -> None:
            calls["cameras"] = dict(warmup_timeout=warmup_timeout, head_mode=head_mode, cyclonedds_uri=cyclonedds_uri)
            calls["order"].append("cameras")
            self.closed = False

        def get_head_frame(self):
            return _Frame(np.zeros((2, 4, 3), np.uint8))

        def get_left_wrist_frame(self):
            return _Frame(np.zeros((2, 4, 3), np.uint8))

        def get_right_wrist_frame(self):
            return _Frame(np.zeros((2, 4, 3), np.uint8))

        def close(self):
            self.closed = True

    arm = types.ModuleType("teleop.robot_control.robot_arm")
    arm.H2_ArmController = H2_ArmController
    hand = types.ModuleType("teleop.robot_control.robot_hand_sharpa_dds")
    hand.SharpaHandDDSClient = SharpaHandDDSClient
    cams = types.ModuleType("teleop.utils.ros_image_client")
    cams.ROSImageClient = ROSImageClient
    monkeypatch.setitem(sys.modules, "teleop.robot_control.robot_arm", arm)
    monkeypatch.setitem(sys.modules, "teleop.robot_control.robot_hand_sharpa_dds", hand)
    monkeypatch.setitem(sys.modules, "teleop.utils.ros_image_client", cams)


# ── DDS initialisation ───────────────────────────────────────────────────────


def test_init_unitree_dds_pins_addresses_and_discovers_the_body_peers(monkeypatch) -> None:
    calls: dict = {}
    _install_dds_fakes(monkeypatch, calls, peers_from_discovery=["10.1.1.4:7410", "10.1.1.4:7412"])
    robot.init_unitree_dds(robot.DdsOptions(domain=1, discovery_ssh="op@box", discovery_timeout_s=7.0))
    domain, iface, xml = calls["init"]
    assert domain == 1 and iface is None
    for needle in ("local=10.1.1.2", "10.1.1.3", "10.1.1.4:7410", "10.1.1.4:7412", "'allow_multicast': 'spdp'"):
        assert needle in xml
    assert calls["discover"] == ("op@box", "10.1.1.4", 7.0)


def test_init_unitree_dds_explicit_peers_or_no_arm_skip_discovery(monkeypatch) -> None:
    calls: dict = {}
    _install_dds_fakes(monkeypatch, calls, discovery_error=True)
    robot.init_unitree_dds(
        robot.DdsOptions(network_address="10.9.9.9", network_peer="10.9.9.8", body_peers=("10.9.9.7:1",))
    )
    assert "discover" not in calls and "local=10.9.9.9" in calls["init"][2] and "10.9.9.7:1" in calls["init"][2]
    calls = {}
    _install_dds_fakes(monkeypatch, calls, discovery_error=True)
    robot.init_unitree_dds(robot.DdsOptions(discover_body_peers=False))
    assert "discover" not in calls and "peers=10.1.1.3 " in calls["init"][2]


def test_init_unitree_dds_interface_shortcut_and_discovery_failure(monkeypatch) -> None:
    calls: dict = {}
    _install_dds_fakes(monkeypatch, calls)
    robot.init_unitree_dds(robot.DdsOptions(domain=2, network_interface="eth9"))
    assert calls["init"] == (2, "eth9", None)
    _install_dds_fakes(monkeypatch, calls, discovery_error=True)
    with pytest.raises(SystemExit, match="--body-dds-peer"):
        robot.init_unitree_dds(robot.DdsOptions())


# ── the H2 adapter ───────────────────────────────────────────────────────────


def test_h2robot_connect_builds_the_teleop_objects_in_the_old_clients_order(monkeypatch, tmp_path: Path) -> None:
    """DDS factory, hands (neutral command), arm controller with the old kwargs, then cameras."""
    calls: dict = {"order": []}
    _install_dds_fakes(monkeypatch, calls, peers_from_discovery=["10.1.1.4:7410"])
    _install_teleop_fakes(monkeypatch, calls)
    _make_teleop_checkout(tmp_path)
    monkeypatch.setenv(robot.XR_TELEOPERATE_ENV, str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    slept: list[float] = []
    monkeypatch.setattr(robot.time, "sleep", slept.append)
    options = robot.RobotOptions(
        arm_options=robot.ArmOptions(motion=True, head_pitch_home=0.45, velocity_limit=12.0, kp_low=120.0, kd_head=2.5),
        camera_options=robot.CameraOptions(
            head_mode="mono", warmup_s=0.2, network_address="10.2.2.1", network_peer="10.2.2.2"
        ),
    )
    h2 = robot.H2Robot(options).connect()
    assert calls["order"] == ["dds", "hand:left", "hand:right", "arm", "cameras"]
    assert calls["arm"] == dict(
        motion_mode=True,
        head_pitch_home=0.45,
        kp_low=120.0,
        kp_wrist=None,
        kd_low=None,
        kd_wrist=None,
        kp_head=None,
        kd_head=2.5,
    )
    assert h2._arm.arm_velocity_limit == 12.0
    assert calls["cameras"]["warmup_timeout"] == 0.2 and calls["cameras"]["head_mode"] == "mono"
    assert (
        "local=10.2.2.1" in calls["cameras"]["cyclonedds_uri"]
        and "max_message_size_bytes" in calls["cameras"]["cyclonedds_uri"]
    )
    assert slept and slept[0] == 2.0, "the hands get two seconds to reach neutral"
    assert h2.has_arm and h2.hands == ("left", "right")
    for side in ("left", "right"):
        assert h2._hands[side].commands[0] == ([0.0] * 22, False)
    assert np.allclose(h2.arm_q(), np.arange(14) / 10) and np.allclose(h2.hand_q("left"), np.pi / 4)
    assert h2.frames()["head"].shape == (2, 4, 3)
    h2.send_arm(np.ones(14))
    assert h2._arm.commands[-1] == ("q", [1.0] * 14, [0.0] * 14)
    h2.go_home()
    assert h2._arm.commands[-1] == "home" and h2._hands["left"].commands[-1] == ([0.0] * 22, True)
    hands, cams = dict(h2._hands), h2._cameras
    h2.close()
    h2.close()
    assert all(client.commands[-1] == "neutral" for client in hands.values()) and cams.closed and h2.hands == ()


def test_h2robot_no_arm_skips_dds_discovery_and_the_controller(monkeypatch, tmp_path: Path) -> None:
    calls: dict = {"order": []}
    _install_dds_fakes(monkeypatch, calls, discovery_error=True)
    _install_teleop_fakes(monkeypatch, calls)
    _make_teleop_checkout(tmp_path)
    monkeypatch.setenv(robot.XR_TELEOPERATE_ENV, str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(robot.time, "sleep", lambda _s: None)
    options = robot.RobotOptions(
        arm=False, hands=("left",), cameras=False, dds=robot.DdsOptions(discover_body_peers=False)
    )
    h2 = robot.H2Robot(options).connect()
    assert calls["order"] == ["dds", "hand:left"] and "discover" not in calls
    assert (
        h2.has_arm is False
        and h2.hands == ("left",)
        and h2.frames() == {"head": None, "left_wrist": None, "right_wrist": None}
    )
    assert h2.arm_q().tolist() == [0.0] * 14
    h2.send_arm(np.zeros(14))  # inert without an arm
    snap = h2.snapshot(("left", "right"))
    assert set(snap.hand_q) == {"left"} and snap.arm_q is None


def test_h2robot_without_connect_is_inert() -> None:
    h2 = robot.H2Robot(robot.RobotOptions())
    assert h2.has_arm is False and h2.hands == ()
    assert h2.arm_q().tolist() == [0.0] * 14 and h2.hand_q("left").tolist() == [0.0] * 22
    assert h2.frames() == {"head": None, "left_wrist": None, "right_wrist": None}
    h2.send_arm(np.zeros(14))
    h2.send_hand("left", np.zeros(22))
    h2.go_home()
    h2.close()
