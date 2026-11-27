# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compare controller outputs against the retained pinned Lab arithmetic."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "third_party/IsaacLab-3939445/source/isaaclab/isaaclab"


def load_math_and_ik():
    spec = importlib.util.spec_from_file_location("servo_reference_math", LAB / "utils/math.py")
    math = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    tree = ast.parse((LAB / "controllers/differential_ik.py").read_text())
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    namespace = {"apply_delta_pose": math.apply_delta_pose, "compute_pose_error": math.compute_pose_error}
    exec(compile(tree, "pinned_differential_ik.py", "exec"), namespace)
    return math, namespace["DifferentialIKController"]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("count", [1, 8, 16])
@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "near_zero",
        "near_pi",
        "singular",
        "zero",
        "mixed",
        "scaled",
        "strided",
        "weighted",
        "avoidance",
        "zero_root",
    ],
)
def test_frames_and_ik_match_pinned_reference(device, count, case):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    wp.init()
    math, controller = load_math_and_ik()
    torch.manual_seed(42)

    def rand(*shape):
        return torch.randn(*shape, device=device)

    root_p, flange_p, target_p = [rand(count, 3) for _ in range(3)]
    root_q, flange_q, target_q = [torch.nn.functional.normalize(rand(count, 4), dim=-1) for _ in range(3)]
    if case in ("near_zero", "near_pi"):
        angle = 1e-7 if case == "near_zero" else 3.1415926
        flange_q = math.quat_from_angle_axis(
            torch.full((count,), angle, device=device), torch.tensor([1.0, 0.0, 0.0], device=device).expand(count, -1)
        )
    if case == "zero":
        flange_q.zero_()
    if case == "zero_root":
        root_q.zero_()
    if case == "mixed":
        flange_q[0].zero_()
    if case == "scaled":
        root_q.mul_(1.1)
    if case == "strided":
        root_p, flange_p, root_q, flange_q = [
            value.repeat_interleave(2, dim=1)[:, ::2] for value in (root_p, flange_p, root_q, flange_q)
        ]
    # Nontrivial body/column selection and strided flange/root views.
    joints = [0, 2, 3, 5, 6, 7, 9]
    jac = rand(count, 3, 6, 10)
    if case == "singular":
        jac.zero_()
    inputs = (root_p, root_q, target_p, target_q, flange_p, flange_q, jac)
    copies = [value.clone() for value in inputs]
    # TorchScript profiles initial calls and changes fusion after warmup.
    # Compare steady-state implementations after warming the same shapes.
    for _ in range(8):
        math.subtract_frame_transforms(root_p, root_q, target_p, target_q)
        math.subtract_frame_transforms(root_p, root_q, flange_p, flange_q)
        math.matrix_from_quat(math.quat_inv(root_q))
    command_p, command_q = math.subtract_frame_transforms(root_p, root_q, target_p, target_q)
    pos, quat = math.subtract_frame_transforms(root_p, root_q, flange_p, flange_q)
    reference_jac = jac[:, 1, :, joints].clone()
    rotation = math.matrix_from_quat(math.quat_inv(root_q))
    reference_jac[:, :3] = torch.bmm(rotation, reference_jac[:, :3])
    reference_jac[:, 3:] = torch.bmm(rotation, reference_jac[:, 3:])
    cfg = SimpleNamespace(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
        orientation_weight=0.3 if case == "weighted" else None,
        joint_limit_avoidance_gain=0.1 if case == "avoidance" else 0.0,
        joint_limit_avoidance_margin=0.1,
    )
    # Retain the former implementation as the reference, including its host
    # whole-batch guard and the exact pinned transform/Jacobian arithmetic.
    reference = controller(cfg, count, device)
    reference.set_command(torch.cat((command_p, command_q), dim=-1))
    joint_pos = rand(count, 7)
    previous = rand(count, 7)
    action = make_action(controller(cfg, count, device), math, inputs, joints, joint_pos, previous)
    if case == "avoidance":
        for ik in (reference, action._ik):
            ik.set_joint_pos_limits(torch.full((7,), -2.0, device=device), torch.full((7,), 2.0, device=device))
    for _ in range(8):
        action._servo_arm(flange_p, flange_q)
        if float(quat.norm()) != 0.0:
            reference.compute(pos, quat, reference_jac, joint_pos)
    expected = previous if float(quat.norm()) == 0.0 else reference.compute(pos, quat, reference_jac, joint_pos)
    action._servo_arm(flange_p, flange_q)
    torch.testing.assert_close(action.written, expected, rtol=1e-5, atol=1e-6, equal_nan=True)
    for value, before in zip(inputs, copies, strict=True):
        torch.testing.assert_close(value, before, rtol=0, atol=0)


def make_action(ik, math, inputs, joints, joint_pos, previous):
    from collections.abc import Sequence
    from typing import Any

    source = ROOT / "arena/i4h_arena/embodiments/franka_catheter.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "ArmDrivenCatheterAction"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_servo_arm", "_jacobian_root_frame")
    ]
    namespace = dict(torch=torch, wp=wp, math_utils=math, Any=Any, Sequence=Sequence)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    from types import MethodType

    root_p, root_q, target_p, target_q, _, _, jac = inputs

    def proxy(value):
        return SimpleNamespace(torch=value)

    target = torch.zeros((root_p.shape[0], 10), device=root_p.device)
    target[:, joints] = previous
    positions = target.clone()
    positions[:, joints] = joint_pos
    data = SimpleNamespace(
        root_pos_w=proxy(root_p),
        root_quat_w=proxy(root_q),
        body_link_jacobian_w=proxy(jac),
        joint_pos=proxy(positions),
        joint_pos_target=proxy(target),
    )
    action = SimpleNamespace(
        _robot=SimpleNamespace(data=data),
        num_envs=root_p.shape[0],
        _arm_targets=torch.empty_like(target),
        _arm_joint_mask=wp.from_torch(
            torch.isin(torch.arange(10, device=root_p.device), torch.tensor(joints, device=root_p.device))
        ),
        _ik=ik,
        _approach=SimpleNamespace(command_pos=target_p, command_quat=target_q),
        _joint_ids=joints,
        _joint_index=torch.tensor(joints, device=root_p.device),
        _jacobi_body_idx=1,
        _jacobi_joint_ids=joints,
        _ik_command=torch.empty((root_p.shape[0], 7), device=root_p.device),
        _identity_quat=torch.tensor([0.0, 0.0, 0.0, 1.0], device=root_p.device),
        _guard_jacobian=torch.eye(6, len(joints), device=root_p.device),
    )

    def write(*, value, joint_mask, env_mask):
        selected = wp.to_torch(env_mask)[:, None] & wp.to_torch(joint_mask)[None, :]
        target.copy_(torch.where(selected, value, target))
        torch.testing.assert_close(
            target[:, ~wp.to_torch(joint_mask)], torch.zeros_like(target[:, ~wp.to_torch(joint_mask)])
        )
        action.written = target[:, joints].clone()

    action._robot.actuators = SimpleNamespace(target_command=SimpleNamespace(set_position_mask=write))
    for name in ("_servo_arm", "_jacobian_root_frame"):
        setattr(action, name, MethodType(namespace[name], action))
    return action


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_reconstruction_scratch_does_not_read_unbound_articulation_views(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    wp.init()
    source = ROOT / "arena/i4h_arena/embodiments/franka_catheter.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "ArmDrivenCatheterAction"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_rebuild_servo_scratch")
    namespace = dict(torch=torch, wp=wp)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

    class UnboundRobot:
        @property
        def num_joints(self):
            raise AssertionError("articulation root_view is not bound during PHYSICS_READY")

    env = SimpleNamespace(_arrival_readout_cache=(0, object()))
    action = SimpleNamespace(
        _robot=UnboundRobot(), num_envs=3, device=device, _arm_joint_count=9, _joint_ids=list(range(7)), _env=env
    )
    rebuild = namespace["_rebuild_servo_scratch"]
    rebuild(action, None)
    previous = action._arm_targets
    assert not hasattr(env, "_arrival_readout_cache")
    assert action._arm_targets.shape == (3, 9)
    torch.testing.assert_close(
        wp.to_torch(action._arm_joint_mask), torch.tensor([True] * 7 + [False] * 2, device=device)
    )
    rebuild(action, None)
    assert action._arm_targets is not previous
