# SPDX-License-Identifier: Apache-2.0
"""Arena-owned liver objective, sampled at 60 Hz; getters are pure reads."""

from __future__ import annotations

import torch
import warp as wp

PHASES = ("approach", "capture", "qualify", "retract", "hold", "release", "relax", "passed", "failed")
OBJECTIVES = ("full_cycle", "reach_only", "grasped_retract")
FAILURES = (
    None,
    "non_finite",
    "workspace_exit",
    "deadline",
    "wrong_patch_capture",
    "retention_loss",
    "empty_capture",
    "release_timeout",
    "reacquisition_during_relaxation",
    "premature_open",
)
# Integer state columns: phase, step, phase_start, hold_streak, acquisition,
# release_command, surface_clear, volume_clear, failure.
# Float columns: destination xyz, acquired xyz, max_lift, max_radial_residual,
# max_attachment_drift, max_inversions. All mutable state has a world row.
ENCODER = {
    "revision": "liver_57_v1",
    "features": 57,
    "geometry_scale": 100.0,
    "motion_scale": 3000.0,
    "residual_scale": 1000.0,
    "count_scale": 1 / 64,
    "timer_scale": 1 / 1200,
    "hold_scale": 1 / 60,
    "actor_obs_normalization": False,
    "critic_obs_normalization": False,
}
ENCODER["layout"] = [
    {"name": "target_minus_root", "start": 0, "width": 3, "scale": 100.0},
    {"name": "surface_normal", "start": 3, "width": 3, "scale": 1.0},
    {"name": "jaw_center_minus_root", "start": 6, "width": 3, "scale": 100.0},
    {"name": "nearest_sphere_minus_root", "start": 9, "width": 3, "scale": 100.0},
    {"name": "target_control_displacement", "start": 12, "width": 3, "scale": 3000.0},
    {"name": "root_control_displacement", "start": 15, "width": 3, "scale": 3000.0},
    {"name": "destination_minus_target", "start": 18, "width": 3, "scale": 100.0},
    {"name": "command_quaternion_xyzw_and_grip", "start": 21, "width": 5, "scale": 1.0},
    {"name": "jaw_angle_and_held", "start": 26, "width": 2, "scale": 1.0},
    {"name": "acquisition_patch_valid_retained_invalid_wrong_counts", "start": 28, "width": 6, "scale": 1 / 64},
    {"name": "maximum_and_mean_radial_residual", "start": 34, "width": 2, "scale": 1000.0},
    {"name": "surface_volume_counts_and_active_slots", "start": 36, "width": 4, "scale": 1 / 64},
    {"name": "previous_applied_normalized_action", "start": 40, "width": 4, "scale": 1.0},
    {"name": "phase_one_hot", "start": 44, "width": 9, "values": list(PHASES)},
    {
        "name": "attempt_phase_hold_acquisition_timers",
        "start": 53,
        "width": 4,
        "scales": [1 / 1200, 1 / 1200, 1 / 60, 1 / 1200],
    },
]
TASK_CONTRACT = dict(
    targets=[252, 257, 255, 1688, 1098],
    physics_hz=60,
    control_hz=30,
    decimation=2,
    substeps=8,
    iterations=8,
    max_cartesian_increment_m=0.001,
    start_jitter_m=0.005,
    initial_retention_s=0.5,
    retraction_y_m=0.02,
    hold_s=1.0,
    hold_tolerance_m=0.002,
    release_deadline_s=0.5,
    open_relaxation_s=1.0,
    attempt_deadline_s=20.0,
    deadline_is_failure=True,
)
REWARD_REVISION = "valid_patch_capped_lift_no_retention_v3"


@wp.kernel
def reduce_phase(
    s: wp.array2d(dtype=int),
    f: wp.array2d(dtype=float),
    geometry: wp.array2d(dtype=wp.vec3),
    counts: wp.array2d(dtype=int),
    latched: wp.array(dtype=int),
    mechanisms: wp.array2d(dtype=int),
    gap: wp.array(dtype=float),
    actions: wp.array3d(dtype=float),
    root: wp.array2d(dtype=wp.vec3),
    q: wp.array2d(dtype=wp.vec3),
    workspace: wp.array3d(dtype=float),
    objective: int,
):
    e = wp.tid()
    phase = s[e, 0]
    if phase < 7:
        step = s[e, 1] + 1
        s[e, 1] = step
        failure = int(0)  # noqa: UP018, RUF046 - Warp loop-variable type declaration
        for j in range(q.shape[1]):
            p = q[e, j]
            for k in range(3):
                if not wp.isfinite(p[k]):
                    failure = 1
        for k in range(10):
            if not wp.isfinite(actions[e, 0, k]):
                failure = 1
        for k in range(3):
            if root[e, 0][k] < workspace[e, 0, k] or root[e, 0][k] > workspace[e, 1, k]:
                failure = 2
            if actions[e, 0, k] < workspace[e, 0, k] or actions[e, 0, k] > workspace[e, 1, k]:
                failure = 2
        if step >= 1200:
            failure = 3
        grip = actions[e, 0, 7]
        if failure == 0:
            if objective == 1 and gap[e] >= -0.0005 and gap[e] <= 0.0:
                phase = 7
            elif latched[e] != 0 and s[e, 4] < 0:
                s[e, 4] = step
                for k in range(3):
                    f[e, k + 3] = geometry[e, 0][k]
                    f[e, k] = geometry[e, 0][k]
                f[e, 1] += 0.02
                phase = 2
                s[e, 2] = step
            if s[e, 4] >= 0:
                f[e, 6] = wp.max(f[e, 6], geometry[e, 0][1] - f[e, 4])
                if phase < 6 and grip >= 0.5:
                    # stats are updated by the diagnostic kernel below.
                    pass
                if phase < 5 or (phase == 5 and s[e, 5] < 0 and grip >= 0.5):
                    if counts[e, 5] != 0 or counts[e, 1] == 0:
                        failure = 4
                    elif counts[e, 3] < (counts[e, 1] + 1) // 2:
                        failure = 5
                    elif grip < 0.5:
                        failure = 9
            if phase == 0 and grip >= 0.5:
                phase = 1
                s[e, 2] = step
            elif phase == 1 and step - s[e, 2] >= 60:
                failure = 6
            elif phase == 2 and step - s[e, 2] >= 30:
                phase = 3
                s[e, 2] = step
            elif phase == 3 or phase == 4:
                error = wp.vec3(geometry[e, 0][0] - f[e, 0], geometry[e, 0][1] - f[e, 1], geometry[e, 0][2] - f[e, 2])
                s[e, 3] = wp.where(wp.length(error) <= 0.002, s[e, 3] + 1, 0)
                phase = wp.where(s[e, 3] > 0, 4, 3)
                if s[e, 3] >= 60:
                    phase = wp.where(objective == 2, 7, 5)
                    s[e, 2] = step
            elif phase == 5:
                if grip < 0.5 and s[e, 5] < 0:
                    s[e, 5] = step - 1
                if s[e, 5] >= 0:
                    if mechanisms[e, 0] == 0 and mechanisms[e, 2] == 0 and s[e, 6] < 0:
                        s[e, 6] = step
                    if mechanisms[e, 1] == 0 and mechanisms[e, 3] == 0 and s[e, 7] < 0:
                        s[e, 7] = step
                    if s[e, 6] >= 0 and s[e, 7] >= 0:
                        phase = 6
                        s[e, 2] = step
                    elif step - s[e, 5] >= 30:
                        failure = 7
            elif phase == 6:
                if grip >= 0.5 or mechanisms[e, 0] + mechanisms[e, 1] + mechanisms[e, 2] + mechanisms[e, 3] > 0:
                    failure = 8
                elif step - s[e, 2] >= 60:
                    phase = 7
        if failure != 0:
            phase = 8
            s[e, 8] = failure
        s[e, 0] = phase


@wp.kernel
def diagnostics(
    s: wp.array2d(dtype=int),
    f: wp.array2d(dtype=float),
    stats: wp.array2d(dtype=float),
    actions: wp.array3d(dtype=float),
    q: wp.array2d(dtype=wp.vec3),
    attached: wp.array(dtype=int),
    rest: wp.array(dtype=wp.vec3),
    tets: wp.array2d(dtype=int),
    rest_volume: wp.array(dtype=float),
    counts: wp.array2d(dtype=int),
):
    e = wp.tid()
    if s[e, 4] >= 0 and s[e, 0] != 6 and actions[e, 0, 7] >= 0.5:
        f[e, 7] = wp.max(f[e, 7], stats[e, 0])
    for j in range(attached.shape[0]):
        f[e, 8] = wp.max(f[e, 8], wp.length(q[e, attached[j]] - rest[attached[j]]))
    if s[e, 4] >= 0 and (s[e, 0] == 2 or s[e, 0] == 3 or s[e, 0] == 4 or s[e, 8] == 5):
        ratio = float(counts[e, 3]) / float(wp.max(counts[e, 1], 1))
        f[e, 10] = wp.min(f[e, 10], ratio)
        f[e, 11] = wp.max(f[e, 11], float(counts[e, 4]))
    inverted = int(0)  # noqa: UP018, RUF046 - Warp loop-variable type declaration
    for j in range(tets.shape[0]):
        a = q[e, tets[j, 0]]
        b = q[e, tets[j, 1]]
        c = q[e, tets[j, 2]]
        d = q[e, tets[j, 3]]
        if wp.dot(b - a, wp.cross(c - a, d - a)) / rest_volume[j] < 0.0:
            inverted += 1
    f[e, 9] = wp.max(f[e, 9], float(inverted))


class LiverObjective:
    def __init__(self, worlds, device, objective="full_cycle"):
        if objective not in OBJECTIVES:
            raise ValueError(f"unsupported liver objective: {objective}")
        self.objective = objective
        self.s = torch.zeros((worlds, 9), dtype=torch.int32, device=device)
        self.f = torch.zeros((worlds, 12), dtype=torch.float32, device=device)
        self.workspace = torch.zeros((worlds, 2, 3), device=device)
        self.s_wp, self.f_wp, self.workspace_wp = map(wp.from_torch, (self.s, self.f, self.workspace))
        self.reset(slice(None))

    def reset(self, ids):
        self.s[ids] = 0
        self.s[ids, 4:8] = -1
        self.f[ids] = 0
        self.f[ids, 10] = 1

    def sample(self, probe, q):
        wp.launch(
            reduce_phase,
            dim=self.s.shape[0],
            inputs=[
                self.s_wp,
                self.f_wp,
                probe.geometry,
                probe.counts,
                probe.latched,
                probe.mechanism_counts,
                probe.gap,
                probe.ic.actions,
                probe.ic.root,
                q,
                self.workspace_wp,
                OBJECTIVES.index(self.objective),
            ],
            device=probe.binding.scene.device,
        )

    def results(self, ids, targets, geometry):
        s, f, g = self.s.cpu().numpy(), self.f.cpu().numpy(), geometry.cpu().numpy()
        rows = []
        for e in ids:
            acquired = s[e, 4] >= 0
            clear = max(int(s[e, 6]), int(s[e, 7]))
            latency = None if s[e, 5] < 0 or min(s[e, 6:8]) < 0 else (clear - int(s[e, 5])) / 60
            rows.append(
                dict(
                    env_id=int(e),
                    objective=self.objective,
                    success=bool(s[e, 0] == 7),
                    failure_reason=FAILURES[s[e, 8]],
                    phase=PHASES[s[e, 0]],
                    target=int(targets[e]),
                    physics_steps=int(s[e, 1]),
                    actual_lift_m=float(g[e, 0, 1] - f[e, 4]) if acquired else None,
                    max_lift_m=float(f[e, 6]) if acquired else None,
                    max_radial_residual_m=float(f[e, 7]) if acquired else None,
                    release_latency_s=latency,
                    surface_release_latency_s=None if s[e, 6] < 0 else (int(s[e, 6]) - int(s[e, 5])) / 60,
                    volume_release_latency_s=None if s[e, 7] < 0 else (int(s[e, 7]) - int(s[e, 5])) / 60,
                    max_attachment_drift_m=float(f[e, 8]),
                    max_inverted_tets=int(f[e, 9]),
                    final_destination_error_m=float(((g[e, 0] - f[e, :3]) ** 2).sum() ** 0.5) if acquired else None,
                    original_acquisition_step=int(s[e, 4]),
                    retained_hold_samples=int(s[e, 3]),
                    minimum_retained_fraction=float(f[e, 10]) if acquired else None,
                    maximum_invalid_original_bindings=int(f[e, 11]) if acquired else None,
                )
            )
        return rows


def encode_observation(
    geometry,
    root,
    physical_action,
    jaw,
    counts,
    stats,
    mechanisms,
    previous_action,
    previous_target,
    previous_root,
    s,
    f,
):
    destination = torch.where((s[:, 4] >= 0)[:, None], f[:, :3], geometry[:, 0] + geometry.new_tensor([0.0, 0.02, 0.0]))
    timers = torch.stack(
        (
            s[:, 1] / 1200,
            (s[:, 1] - s[:, 2]) / 1200,
            s[:, 3] / 60,
            torch.where(s[:, 4] < 0, 0.0, (s[:, 1] - s[:, 4]) / 1200),
        ),
        dim=-1,
    )
    obs = torch.cat(
        (
            (geometry[:, 0] - root) * 100,
            geometry[:, 1],
            (geometry[:, 2] - root) * 100,
            (geometry[:, 3] - root) * 100,
            (geometry[:, 0] - previous_target) * 3000,
            (root - previous_root) * 3000,
            (destination - geometry[:, 0]) * 100,
            physical_action[:, 3:8],
            jaw,
            counts / 64,
            stats * 1000,
            mechanisms / 64,
            previous_action,
            torch.nn.functional.one_hot(s[:, 0].long(), 9),
            timers,
        ),
        dim=-1,
    )
    if obs.shape[-1] != 57:
        raise RuntimeError(f"liver encoder width changed: {obs.shape}")
    return obs


def normalized_action(actions):
    clipped = actions.clamp(-1.0, 1.0)
    xyz = clipped[:, :3] / torch.linalg.vector_norm(clipped[:, :3], dim=-1, keepdim=True).clamp(min=1.0)
    return torch.cat((xyz, clipped[:, 3:4]), dim=-1)


@wp.kernel
def settling_sample(q: wp.array2d(dtype=wp.vec3), previous: wp.array2d(dtype=wp.vec3), state: wp.array2d(dtype=int)):
    e = wp.tid()
    speed = float(0.0)  # noqa: UP018 - Warp loop-variable type declaration
    for j in range(q.shape[1]):
        speed = wp.max(speed, wp.length(q[e, j] - previous[e, j]) * 60.0)
        previous[e, j] = q[e, j]
    state[e, 0] += 1
    state[e, 1] = wp.where(speed < 0.005, state[e, 1] + 1, 0)
