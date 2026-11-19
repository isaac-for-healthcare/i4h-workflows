# SPDX-License-Identifier: Apache-2.0
import warp as wp
from i4h_surgical.systems.instruments.instrument_control import shaft_rotation, wrist_rotation


@wp.kernel
def stage(
    mask: wp.array(dtype=wp.bool),
    actions: wp.array3d(dtype=float),
    trocar: wp.array(dtype=wp.vec3),
    has_trocar: wp.array(dtype=int),
    root: wp.array2d(dtype=wp.vec3),
    shaft: wp.array2d(dtype=wp.quat),
    wrist: wp.array2d(dtype=wp.quat),
    jaw: wp.array2d(dtype=wp.quat),
):
    e = wp.tid()
    if mask[e]:
        p = wp.vec3(actions[e, 0, 0], actions[e, 0, 1], actions[e, 0, 2])
        q = wp.normalize(wp.quat(actions[e, 0, 3], actions[e, 0, 4], actions[e, 0, 5], actions[e, 0, 6]))
        s = shaft_rotation(p, q, trocar[0], has_trocar[0])
        root[e, 0] = p
        shaft[e, 0] = s
        wrist[e, 0] = wrist_rotation(s, q)
        jaw[e, 0] = q
