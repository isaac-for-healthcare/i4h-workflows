# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rule-based pick and place of the surgical scissors for the SO-ARM 101.

The SO-ARM takes joint-position targets, so the controller plays joint-space
keyframes rather than solving IK. Per env: note where the randomized scissors lie,
then play the stages below, each easing from the joints it starts at through its
keyframes. Two closed-loop corrections read that scissors pose: a shoulder-pan
correction proportional to its Y in the four pan-sensitive stages, and a reach
correction proportional to its X in the descent and the grasp, so the jaw closes
on the same part of the blade wherever the scissors start. After the release the
arm holds while the scissors settle, checks they landed in the tray, then returns
home with the jaw open; the scene's success term (scissors settled in the tray,
tool tip back home) ends the episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from i4h_arena.tensor_utils import to_torch

_OPEN = 0.35
_CLOSED = -0.16


@dataclass(frozen=True)
class Correction:
    """Shift every joint target by the scissors' displacement along one axis.

    ``gains`` is the joint-space direction, in radians per metre, that moves the tool
    along ``axis`` at the stage's pose, read off the arm's Jacobian there. The
    displacement from ``reference`` is clipped to ``limits`` (metres) first, so a pose
    outside the range the gains were derived for cannot drive the arm somewhere
    unvalidated.
    """

    axis: int
    reference: float
    gains: tuple[float, ...]
    limits: tuple[float, float]

    def offset(self, position: torch.Tensor) -> torch.Tensor:
        displacement = (position[:, self.axis] - self.reference).clamp(*self.limits)
        return displacement.unsqueeze(-1) * torch.tensor(self.gains, device=position.device)


# Both corrections were measured, not assumed (NVBug 6903612). The wrist camera
# rides on the gripper, so the blade's column in its image as the jaw closes is the
# lateral grasp error; across 76 recorded episodes it fits pose and correction
# linearly to 2.4 px. The earlier pan rule (10 rad/m on Y, clipped to [0, 0.12])
# turned the jaw twice as far as Y moves the blade: at the high-Y end of the
# randomization the blade sat against the fixed jaw and the grasp slipped.
#
# Shoulder-pan correction for the randomized scissors Y (-0.025..-0.010 m): 4.51
# rad/m holds the blade mid-jaw. It gives the old rule's pan at the nominal pose, so
# the keyframes' own reference is unchanged.
_PAN = Correction(axis=1, reference=-0.0267, gains=(4.51, 0.0, 0.0, 0.0, 0.0, 0.0), limits=(-0.005, 0.025))

# Reach correction for the randomized scissors X (0.11..0.13 m). The scissors lie
# along the arm's reach, so X decides where along them the jaw closes: uncorrected, a
# pose 7-10 mm out closed near the finger rings, which slipped during the lift, and
# one 10 mm in left the scissors caught in the jaw at release. The shoulder, elbow
# and wrist gains move the gripper along world X, solved from its PhysX Jacobian at
# open_on_scissors and close_on_scissors; the pan term cancels the sideways part of
# that motion at the blade.
_REACH = Correction(axis=0, reference=0.12, gains=(-1.35, -6.05, 9.70, -3.64, -1.86, 0.0), limits=(-0.015, 0.015))

#: Stages of (keyframes, corrections). Each keyframe is a joint delta from the home
#: pose, held for its duration [s]; index 5 is the jaw (+open / -close).
STAGES: tuple[tuple[tuple[tuple[tuple[float, ...], float], ...], tuple[Correction, ...]], ...] = (
    # leave home
    (
        (
            ((0.0, 0.0, 0.0, 0.0, 0.0, _OPEN), 0.25),
            ((0.03, 0.34, -0.19, -0.33, 0.25, -0.08), 0.35),
            ((0.09, 0.37, -0.20, -0.48, 0.42, -0.12), 0.33),
            ((0.16, 0.25, -0.14, -0.71, 0.51, -0.12), 0.33),
            ((0.25, 0.06, 0.03, -0.97, 0.50, -0.15), 0.33),
        ),
        (),
    ),
    # align over the scissors
    (
        (
            ((-0.14, -0.03, 0.11, -1.08, 0.51, -0.15), 0.33),
            ((-0.32, -0.05, 0.13, -1.08, 0.52, _CLOSED), 0.33),
        ),
        (_PAN,),
    ),
    # descend and open on them
    (
        (
            ((-0.37, 0.59, -0.44, -1.08, 0.16, _CLOSED), 0.33),
            ((-0.37, 1.23, -0.92, -1.07, -0.14, -0.07), 0.33),
            ((-0.41, 1.75, -1.14, -1.07, -0.16, 0.23), 0.33),
            ((-0.48, 1.92, -1.05, -1.00, -0.15, 0.25), 0.33),
        ),
        (_PAN, _REACH),
    ),
    # close the grip
    (
        (
            ((-0.46, 1.93, -0.94, -0.95, -0.14, -0.15), 0.33),
            ((-0.43, 1.90, -0.92, -1.05, -0.16, _CLOSED), 0.33),
            ((-0.27, 1.68, -1.08, -0.51, -0.10, _CLOSED), 0.33),
        ),
        (_PAN, _REACH),
    ),
    # lift
    ((((-0.01, 1.36, -1.10, -0.42, -0.13, _CLOSED), 0.33),), (_PAN,)),
    # carry to the tray
    (
        (
            ((0.50, 1.16, -1.18, -0.41, 0.12, -0.15), 0.33),
            ((0.69, 1.33, -1.16, -0.72, 0.44, _CLOSED), 0.33),
            ((0.58, 1.70, -1.24, -1.15, 0.62, _CLOSED), 0.33),
        ),
        (),
    ),
    # Release: open the jaw fully and hold the arm still while the scissors drop.
    # Opening part-way (0.16, then 0.29) while already lifting left them caught in the
    # jaw on some grasps, dragged up and dropped tip-first, so they stood in the tray
    # and toppled for seconds afterwards.
    (
        (
            ((0.52, 1.89, -1.14, -1.21, 0.65, _OPEN), 0.33),
            ((0.52, 1.89, -1.14, -1.21, 0.65, _OPEN), 0.33),
        ),
        (),
    ),
    # retreat
    (
        (
            ((0.50, 1.33, -1.21, -1.24, 0.61, _OPEN), 0.33),
            ((0.47, 0.61, -0.52, -1.22, 0.32, _OPEN), 0.33),
            ((0.38, -0.07, 0.00, -1.22, 0.30, _OPEN), 0.33),
            ((0.22, -0.15, 0.14, -0.69, 0.56, _OPEN), 0.65),
        ),
        (),
    ),
)

#: The tray volume the scissors must end up in, in the env frame [m].
TRAY_BOX = ((0.02, 0.15, 0.24), (0.22, 0.35, 0.34))

LOCATE = 0
SETTLE = 1 + len(STAGES)
VERIFY = SETTLE + 1
HOME = VERIFY + 1


@dataclass
class ScissorPickPlacePolicyCfg(PolicyCfg):
    settle_timeout_s: float = 0.4
    """Longest hold for the released scissors to settle (speed below 1 cm/s) [s]. The
    retreat already gives them 1.65 s, and this keeps the worst case inside the
    scene's 600-step budget."""

    verify_timeout_s: float = 0.5
    """Longest hold for the scissors to show up inside the tray before homing anyway [s]."""

    home_s: float = 1.0
    """Time to ease back to the home pose [s]."""


@register_policy
class ScissorPickPlacePolicy(PolicyBase[ScissorPickPlacePolicyCfg]):
    """Pick up the scene's ``scissors`` and place them in the tray with joint keyframes."""

    name = "i4h_scissor_pick_place"

    def __init__(self, config: ScissorPickPlacePolicyCfg):
        super().__init__(config)
        self._phase: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._phase is None:
            return
        ids = slice(None) if env_ids is None else env_ids
        self._phase[ids] = LOCATE
        self._tick[ids] = 0
        self._frame[ids] = 0
        self._command[ids] = self._home[ids]

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        env = env.unwrapped
        cfg = self.config
        robot, scissors = env.scene["robot"].data, env.scene["scissors"].data
        n, device, dt = env.num_envs, env.device, env.step_dt
        joints = to_torch(robot.joint_pos)
        if self._phase is None:
            self._home = to_torch(robot.default_joint_pos).clone()
            self._phase = torch.full((n,), LOCATE, dtype=torch.long, device=device)
            self._tick = torch.zeros(n, dtype=torch.long, device=device)
            self._frame = torch.zeros(n, dtype=torch.long, device=device)
            # Until a stage commands otherwise, hold the home pose.
            self._command = self._home.clone()
            self._start = torch.zeros_like(joints)
            self._scissors = torch.zeros(n, 3, device=device)
        self._tick += 1
        done = torch.zeros(n, dtype=torch.bool, device=device)

        locate = self._phase == LOCATE
        if locate.any():
            position = to_torch(scissors.root_pos_w) - env.scene.env_origins
            self._scissors[locate] = position[locate]
            done |= locate

        for index, (frames, corrections) in enumerate(STAGES):
            stage = self._phase == 1 + index
            if not stage.any():
                continue
            # A stage starts from the joints the arm reached; each later keyframe from the last one's target.
            entering = stage & (self._tick == 1) & (self._frame == 0)
            self._start[entering] = joints[entering]
            frame = self._frame.clamp(max=len(frames) - 1)
            targets = torch.tensor([target for target, _ in frames], device=device)[frame]
            steps = torch.tensor([max(1, round(duration / dt)) for _, duration in frames], device=device)[frame]
            goal = self._home + targets
            for correction in corrections:
                goal = goal + correction.offset(self._scissors)
            alpha = (self._tick.float() / steps).clamp(max=1.0)
            eased = (alpha * alpha * (3.0 - 2.0 * alpha)).unsqueeze(-1)
            command = self._start + (goal - self._start) * eased
            self._command[stage] = command[stage]
            frame_done = stage & (self._tick >= steps)
            self._start[frame_done] = goal[frame_done]
            self._frame[frame_done] += 1
            self._tick[frame_done] = 0
            done |= frame_done & (self._frame >= len(frames))

        settle = self._phase == SETTLE
        if settle.any():
            speed = to_torch(scissors.root_lin_vel_w).norm(dim=-1)
            done |= settle & ((speed < 0.01) | (self._tick * dt >= cfg.settle_timeout_s))

        verify = self._phase == VERIFY
        if verify.any():
            position = to_torch(scissors.root_pos_w) - env.scene.env_origins
            low, high = (torch.tensor(corner, device=device) for corner in TRAY_BOX)
            in_tray = ((position >= low) & (position <= high)).all(dim=-1)
            done |= verify & (in_tray | (self._tick * dt >= cfg.verify_timeout_s))

        home = self._phase == HOME
        entering = home & (self._tick == 1)
        self._start[entering] = joints[entering]
        if home.any():
            alpha = (self._tick.float() / max(1, round(cfg.home_s / dt))).clamp(max=1.0)
            eased = (alpha * alpha * (3.0 - 2.0 * alpha)).unsqueeze(-1)
            command = self._start + (self._home - self._start) * eased
            command[:, -1] = _OPEN
            self._command[home] = command[home]

        self._phase[done] += 1
        self._tick[done] = 0
        self._frame[done] = 0
        return self._command.clone()
