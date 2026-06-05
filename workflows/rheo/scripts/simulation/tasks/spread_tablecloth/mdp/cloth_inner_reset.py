# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reset the inner rigid body (Cloth_In) embedded inside the deformable cloth USD."""

from __future__ import annotations

import contextlib
import re

import numpy as np
import omni.physx
import omni.usd
import torch
import warp as wp
from isaaclab_physx.physics.physx_manager import PhysxManager
from pxr import Gf, Usd, UsdGeom
from utils.logging import make_logger

_DEFAULT_CLOTH_IN_REL_PATH = "Cloth_In001/Cloth_In001"
# Caches stashed on the env instance:
_CACHE_INIT = "_cloth_inner_init"  # dict[env_id -> {pos_gf, rot_gf, pose7}]
_CACHE_VIEW = "_cloth_inner_rigid_view"  # PhysX rigid body view
_CACHE_GLOB = "_cloth_inner_view_pattern"  # glob this view was built for
_CACHE_NT_VIEW = "_cloth_inner_newton_view"  # Newton ArticulationView
_CACHE_NT_GLOB = "_cloth_inner_newton_view_pattern"  # glob the Newton view was built for

_log, _error = make_logger("CLOTHRST")


def _is_newton_backend(env) -> bool:
    """True when the active physics backend is a Newton manager.

    ``physics_manager`` is a LazyType proxy, so we compare ``__name__`` rather
    than using ``isinstance`` (matches IsaacLab's own backend checks).
    """
    try:
        return "newton" in env.sim.physics_manager.__name__.lower()
    except Exception:
        return False


def _resolve_env_prim_path(prim_path_expr: str, env_id: int) -> str:
    path = prim_path_expr
    if "{ENV_REGEX_NS}" in path:
        path = path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")
    path = re.sub(r"env_\.\*", f"env_{env_id}", path)
    path = re.sub(r"env_\d+", f"env_{env_id}", path)
    return path


def _expr_to_glob(prim_path_expr: str) -> str:
    """Convert IsaacLab regex prim path into a wildcard glob for PhysX views."""
    g = prim_path_expr.replace("{ENV_REGEX_NS}", "/World/envs/env_*")
    g = g.replace("env_.*", "env_*")
    return g


def _get_isaaclab_sim_view():
    """Reuse the warp-backed SimulationView IsaacLab already created.

    Available after the first physics warmup. Returns None when not ready yet.
    """
    try:
        return PhysxManager._view
    except Exception:
        return None


def _try_physx_tensor_reset(env, env_ids, prim_path_expr, inner_rel_path):
    """Reset Cloth_In via the PhysX tensor (warp) API. True on success."""
    sim_view = _get_isaaclab_sim_view()
    if sim_view is None:
        # Sim view isn't ready before play; USD fallback writes still propagate.
        _log("PhysX SimView not ready yet, USD fallback")
        return False

    glob = f"{_expr_to_glob(prim_path_expr)}/{inner_rel_path}"

    rigid_view = getattr(env, _CACHE_VIEW, None)
    if rigid_view is None or getattr(env, _CACHE_GLOB, None) != glob:
        try:
            rigid_view = sim_view.create_rigid_body_view(glob)
        except Exception as e:
            _error(f"create_rigid_body_view({glob!r}) failed: {e!r} -> USD fallback")
            return False
        if rigid_view is None or rigid_view.count == 0:
            _log(f"create_rigid_body_view({glob!r}) returned empty -> USD fallback")
            return False
        _log(f"created rigid_view for {glob!r}, count={rigid_view.count}")
        setattr(env, _CACHE_VIEW, rigid_view)
        setattr(env, _CACHE_GLOB, glob)

    n_total = rigid_view.count
    init_dict = getattr(env, _CACHE_INIT)

    m = len(env_ids)
    pose_np = np.empty((m, 7), dtype=np.float32)
    for i, eid in enumerate(env_ids):
        cache = init_dict.get(eid)
        if cache is None:
            _log(f"no cached init pose for env {eid} -> USD fallback")
            return False
        p = cache["pose7"]
        pose_np[i] = p.cpu().numpy() if hasattr(p, "cpu") else p

    idx_np = np.array([e for e in env_ids if 0 <= int(e) < n_total], dtype=np.int32)
    if idx_np.shape[0] != m:
        _log(f"env_ids out of view range ({m} -> {idx_np.shape[0]}) -> USD fallback")
        return False

    device = str(env.device)
    try:
        pose_wp = wp.array(pose_np, dtype=wp.float32, device=device)
        idx_wp = wp.array(idx_np, dtype=wp.int32, device=device)
        vel_wp = wp.zeros((m, 6), dtype=wp.float32, device=device)
        rigid_view.set_transforms(pose_wp, indices=idx_wp)
        rigid_view.set_velocities(vel_wp, indices=idx_wp)
        _log(f"PhysX tensor reset OK: m={m} pose0={pose_np[0].tolist()}")
        return True
    except Exception as e:
        _error(f"tensor reset failed: {e!r} -> invalidating view, USD fallback")
        try:
            setattr(env, _CACHE_VIEW, None)
            setattr(env, _CACHE_GLOB, None)
        except Exception:
            pass
        return False


def _try_newton_tensor_reset(env, env_ids, prim_path_expr, inner_rel_path):
    """Reset Cloth_In via the Newton ``ArticulationView`` API. True on success.

    Under the Newton (MJWarp) backend the embedded rigid body is imported as a
    single-body floating-base articulation whose root pose lives in
    ``state.joint_q[0:7]``. We overwrite those rows for the requested envs and
    mark the FK dirty so ``body_q`` (collision/render) is refreshed.

    ``values`` passed to ``set_root_transforms`` must be full-view-sized; the
    per-world ``mask`` selects which envs are actually written. We assume the
    view's world order matches env index (IsaacLab replicates envs in order).
    """
    try:
        import numpy as np
        import warp as wp
        from isaaclab_newton.physics import NewtonManager
        from newton.selection import ArticulationView
    except Exception as e:  # Newton not available -> let caller fall back.
        _log(f"Newton import failed: {e!r} -> USD fallback")
        return False

    model = NewtonManager.get_model()
    state = NewtonManager.get_state_0()
    if model is None or state is None:
        _log("Newton model/state not ready yet -> USD fallback")
        return False

    glob = f"{_expr_to_glob(prim_path_expr)}/{inner_rel_path}"

    view = getattr(env, _CACHE_NT_VIEW, None)
    if view is None or getattr(env, _CACHE_NT_GLOB, None) != glob:
        try:
            view = ArticulationView(model, glob, verbose=False)
        except Exception as e:
            _error(f"ArticulationView({glob!r}) failed: {e!r} -> USD fallback")
            return False
        if getattr(view, "count", 0) == 0:
            _log(f"ArticulationView({glob!r}) matched 0 articulations -> USD fallback")
            return False
        if not getattr(view, "is_floating_base", False):
            # A fixed-base inner body has no movable root joint; nothing to reset.
            _log(f"inner body {glob!r} is fixed-base, skipping Newton root reset")
            return False
        _log(f"created Newton view for {glob!r}, count={view.count}")
        setattr(env, _CACHE_NT_VIEW, view)
        setattr(env, _CACHE_NT_GLOB, glob)

    init_dict = getattr(env, _CACHE_INIT)

    try:
        # Root transforms as torch, preserving the view's native shape so the
        # write passes set_root_transforms' `values.shape == attrib.shape` check.
        # Last dim is 7 = [px,py,pz, qx,qy,qz,qw]; quaternion is (x,y,z,w).
        rt = wp.to_torch(view.get_root_transforms(state)).clone()
        lead_shape = rt.shape[:-1]
        flat = rt.reshape(-1, 7)  # one row per articulation, in world order
        n_rows = flat.shape[0]
        mask = torch.zeros(lead_shape[0], dtype=torch.bool, device=flat.device)

        n_written = 0
        for eid in env_ids:
            eid = int(eid)
            cache = init_dict.get(eid)
            # per_world == 1 for the cloth inner body, so flat row index == env index.
            if cache is None or not (0 <= eid < n_rows):
                continue
            flat[eid] = cache["pose7"].to(flat.device, dtype=flat.dtype)
            mask[eid] = True
            n_written += 1

        if n_written == 0:
            _log("no Newton-resettable env_ids -> USD fallback")
            return False

        mask_wp = wp.array(mask.cpu().numpy().astype(np.bool_), dtype=wp.bool, device=str(env.device))
        pose_wp = wp.from_torch(flat.reshape(*lead_shape, 7).contiguous(), dtype=wp.transform)
        vel_wp = wp.zeros(tuple(lead_shape), dtype=wp.spatial_vector, device=str(env.device))

        view.set_root_transforms(state, pose_wp, mask=mask_wp)
        view.set_root_velocities(state, vel_wp, mask=mask_wp)

        # Refresh FK so body_q (collision + rendering) reflects the new root pose.
        env_ids_wp = wp.array(
            np.asarray([int(e) for e in env_ids], dtype=np.int32), dtype=wp.int32, device=str(env.device)
        )
        try:
            NewtonManager.invalidate_fk(env_ids=env_ids_wp, articulation_ids=getattr(view, "articulation_ids", None))
        except TypeError:
            NewtonManager.invalidate_fk(env_ids=env_ids_wp)
        NewtonManager.forward()
        _log(f"Newton reset OK: wrote {n_written} env(s)")
        return True
    except Exception as e:
        _error(f"Newton reset failed: {e!r} -> invalidating view, USD fallback")
        with contextlib.suppress(Exception):
            setattr(env, _CACHE_NT_VIEW, None)
            setattr(env, _CACHE_NT_GLOB, None)
        return False


def _usd_pose_reset(stage, prim_path, init_pos, init_rot):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(init_pos)
    xformable.AddOrientOp().Set(
        Gf.Quatf(
            float(init_rot.GetReal()),
            Gf.Vec3f(*[float(x) for x in init_rot.GetImaginary()]),
        )
    )
    return True


def _autodetect_inner_rel_path(stage, prim_path_expr) -> str | None:
    """Find the first descendant carrying PhysicsRigidBodyAPI under env_0.

    Avoids hard-coding Cloth_In001 vs Cloth_In002 across fold variants.
    """
    base = prim_path_expr
    if "{ENV_REGEX_NS}" in base:
        base = base.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    base = re.sub(r"env_\.\*", "env_0", base)
    template = stage.GetPrimAtPath(base)
    if not template or not template.IsValid():
        _log(f"autodetect: template prim {base!r} is invalid")
        return None

    base_len = len(base.rstrip("/")) + 1
    visited = []
    for prim in Usd.PrimRange.AllPrims(template):
        if prim == template:
            continue
        path_str = prim.GetPath().pathString
        applied = list(prim.GetAppliedSchemas())
        visited.append((path_str, applied))
        if "PhysicsRigidBodyAPI" in applied:
            rel = path_str[base_len:]
            _log(f"autodetect: found rigid body at {path_str!r}, rel={rel!r}")
            return rel

    _log(f"autodetect FAILED under {base!r}. " f"Visited {len(visited)} prims, none had PhysicsRigidBodyAPI:")
    for p, schemas in visited[:30]:
        _log(f"   - {p}  schemas={schemas}")
    if len(visited) > 30:
        _log(f"   ... and {len(visited) - 30} more")
    return None


def reset_cloth_inner(
    env,
    env_ids,
    cloth_asset_name: str = "tablecloth",
    inner_rel_path: str | None = None,
):
    """Reset Cloth_In rigid body back to its initial pose."""
    _log(
        f"reset_cloth_inner called: cloth_asset_name={cloth_asset_name!r} "
        f"inner_rel_path={inner_rel_path!r} env_ids={env_ids!r}"
    )
    stage = omni.usd.get_context().get_stage()

    cloth_cfg = getattr(env.scene.cfg, cloth_asset_name, None)
    if cloth_cfg is None:
        return
    prim_path_expr = cloth_cfg.prim_path

    # Resolve which sub-prim to reset (cached on env to avoid re-walking USD).
    if inner_rel_path is None:
        inner_rel_path = getattr(env, "_cloth_inner_rel_path", None)
        if inner_rel_path is None:
            detected = _autodetect_inner_rel_path(stage, prim_path_expr)
            inner_rel_path = detected or _DEFAULT_CLOTH_IN_REL_PATH
            setattr(env, "_cloth_inner_rel_path", inner_rel_path)
            _log(
                f"using inner_rel_path={inner_rel_path!r} "
                f"(autodetected={'yes' if detected else 'no, fallback to default'})"
            )

    if env_ids is None:
        env_ids = list(range(env.num_envs))
    elif isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.detach().cpu().tolist()
    else:
        env_ids = list(env_ids)

    if not hasattr(env, _CACHE_INIT):
        setattr(env, _CACHE_INIT, {})
    init_dict = getattr(env, _CACHE_INIT)

    # Cache init pose for each requested env.
    valid_env_ids = []
    inner_paths = []
    for env_id in env_ids:
        base = _resolve_env_prim_path(prim_path_expr, env_id)
        path = f"{base}/{inner_rel_path}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            _log(f"prim not found: {path}")
            continue

        if env_id not in init_dict:
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
            t = xf.ExtractTranslation()
            q = xf.ExtractRotationQuat()
            init_dict[env_id] = {
                "pos_gf": Gf.Vec3d(t),
                "rot_gf": Gf.Quatd(q),
                # PhysX set_transforms layout: [x,y,z, qx,qy,qz,qw]
                "pose7": torch.tensor(
                    [
                        float(t[0]),
                        float(t[1]),
                        float(t[2]),
                        float(q.GetImaginary()[0]),
                        float(q.GetImaginary()[1]),
                        float(q.GetImaginary()[2]),
                        float(q.GetReal()),
                    ],
                    dtype=torch.float32,
                ),
            }
            _log(
                f"cached env {env_id} init pos={tuple(float(x) for x in t)} "
                f"quat(xyzw)=({float(q.GetImaginary()[0]):.4f},"
                f"{float(q.GetImaginary()[1]):.4f},"
                f"{float(q.GetImaginary()[2]):.4f},"
                f"{float(q.GetReal()):.4f})"
            )

        valid_env_ids.append(env_id)
        inner_paths.append(path)

    if not valid_env_ids:
        _log("no valid env_ids found (all prims invalid?), nothing to do")
        return

    # Path A: backend tensor view. Newton uses an ArticulationView on the MJWarp
    # model; PhysX uses the warp-backed rigid-body SimulationView.
    if _is_newton_backend(env):
        if _try_newton_tensor_reset(env, valid_env_ids, prim_path_expr, inner_rel_path):
            return
    else:
        if _try_physx_tensor_reset(env, valid_env_ids, prim_path_expr, inner_rel_path):
            return

    # Path B: USD xform writes + flush (only at startup before play, or when
    # suppressReadback is False).
    _log(f"USD fallback path for env_ids={valid_env_ids}")
    for env_id, path in zip(valid_env_ids, inner_paths):
        cache = init_dict.get(env_id)
        if cache is None:
            continue
        _usd_pose_reset(stage, path, cache["pos_gf"], cache["rot_gf"])

    # The PhysX fast-cache flush only applies to the PhysX backend. Under Newton
    # the authoritative state is the model state + FK, and the USD xform writes
    # above are picked up on the next model reset/build.
    if not _is_newton_backend(env):
        try:
            omni.physx.get_physx_interface().update_transformations(
                updateToFastCache=True,
                updateToUsd=False,
                updateVelocitiesToUsd=False,
                outputVelocitiesLocalSpace=False,
            )
        except Exception as e:
            _error(f"physx update_transformations failed: {e!r}")
