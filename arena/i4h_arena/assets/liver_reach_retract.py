# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visible diagnostic mesh and virtual-grasper spheres for environment zero.

Display copies occur outside physics capture and are disabled for headless RL.
They are USD drawing geometry, with no physics schemas or solver ownership.
"""

import numpy as np
from pxr import Gf, UsdGeom, UsdLux, Vt


class LiverDisplay:
    def __init__(self, env):
        self.env = env
        stage = env.sim.stage
        self.mesh = UsdGeom.Mesh.Define(stage, "/World/LiverDisplay/Tissue")
        tris = env.liver.probe.triangles
        self.mesh.CreateFaceVertexCountsAttr(np.full(len(tris), 3))
        self.mesh.CreateFaceVertexIndicesAttr(tris.reshape(-1))
        self.mesh.CreateSubdivisionSchemeAttr("none")
        self.mesh.CreateDoubleSidedAttr(True)
        self.mesh.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.055, 0.04)])
        self.tools = UsdGeom.Points.Define(stage, "/World/LiverDisplay/VirtualGrasper")
        self.tools.CreateDisplayColorAttr([Gf.Vec3f(0.65, 0.75, 0.83)])
        self.markers = UsdGeom.Points.Define(stage, "/World/LiverDisplay/TargetDestination")
        self.markers.CreateDisplayColorAttr([Gf.Vec3f(0.1, 1.0, 0.2), Gf.Vec3f(1.0, 0.8, 0.05)])
        self.markers.CreateWidthsAttr([0.004, 0.004])
        light = UsdLux.DomeLight.Define(stage, "/World/LiverDisplay/Light")
        light.CreateIntensityAttr(1000.0)
        self.update()

    def update(self):
        runtime = self.env.liver
        runtime.component.sync_views()
        p = runtime.probe
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(runtime.q.numpy()[0]))
        active = p.ic.sphere_active.numpy()[0] != 0
        self.tools.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p.ic.sphere_q.numpy()[0, active]))
        self.tools.CreateWidthsAttr((p.ic.sphere_radius.numpy()[active] * 2).astype(np.float32))
        if int(runtime.views["target"][0]) >= 0:
            target = runtime.views["geometry"][0, 0].cpu().numpy()
            destination = (
                runtime.objective.f[0, :3].cpu().numpy()
                if runtime.objective.s[0, 4] >= 0
                else target + [0.0, 0.02, 0.0]
            )
            self.markers.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([target, destination], np.float32)))
