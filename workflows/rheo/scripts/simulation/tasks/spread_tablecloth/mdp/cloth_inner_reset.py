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
"""Reset the inner rigid body (Cloth_In) embedded inside the deformable cloth USD.

Mirrors `unitree_sim_isaaclab/tasks/common_event/cloth_inner_reset.py` so the
spread_tablecloth task can keep `Cloth_In002` from drifting upward across
resets when reusing the `Cloth_fold10.usd` asset.
"""
from __future__ import annotations

import re
import sys
import torch
from pxr import Gf, UsdGeom


_DEFAULT_CLOTH_IN_REL_PATH = "Cloth_In001/Cloth_In001"
# Attribute names of caches we stash on the env instance (one helper, many resets):
_CACHE_INIT = "_cloth_inner_init"          # dict[env_id -> {pos_gf, rot_gf, pose7}]
_CACHE_VIEW = "_cloth_inner_rigid_view"    # PhysX rigid body view (created once)
_CACHE_GLOB = "_cloth_inner_view_pattern"  # glob string the view was built for


def _resolve_env_prim_path(prim_path_expr: str, env_id: int) -> str:
    path = prim_path_expr
    if "{ENV_REGEX_NS}" in path:
        path = path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")
    path = re.sub(r"env_\.\*", f"env_{env_id}", path)
    path = re.sub(r"env_\d+", f"env_{env_id}", path)
    return path


def _expr_to_glob(prim_path_expr: str) -> str:
    """Convert IsaacLab regex prim path into a wildcard glob accepted by PhysX views."""
    g = prim_path_expr.replace("{ENV_REGEX_NS}", "/World/envs/env_*")
    g = g.replace("env_.*", "env_*")
    return g


def _get_isaaclab_sim_view():
    """Reuse the warp-backed SimulationView IsaacLab already created.

    Available after the first physics warmup (``PhysicsManager._on_play``).
    Returns ``None`` if the view does not exist yet -- callers should fall back to USD.
    """
    try:
        from isaaclab_physx.physics.physx_manager import PhysxManager
        return PhysxManager._view
    except Exception:
        return None


def _try_physx_tensor_reset(env, env_ids, prim_path_expr, inner_rel_path):
    """Reset the Cloth_In rigid bodies via the PhysX tensor (warp) API.

    Returns True on success, False so the caller can fall back to USD.
    """
    try:
        import warp as wp
        import numpy as np
    except Exception as e:
        _log(f"warp/numpy unavailable: {e!r} -> USD fallback")
        return False

    sim_view = _get_isaaclab_sim_view()
    if sim_view is None:
        # Sim view isn't ready yet (very first reset before play). USD fallback is fine here
        # because the simulation hasn't started -- USD writes still propagate to PhysX.
        _log("PhysX SimView not ready yet, USD fallback")
        return False

    glob = f"{_expr_to_glob(prim_path_expr)}/{inner_rel_path}"

    # Cache the rigid body view so we don't pay the create cost on every reset
    rigid_view = getattr(env, _CACHE_VIEW, None)
    if rigid_view is None or getattr(env, _CACHE_GLOB, None) != glob:
        try:
            rigid_view = sim_view.create_rigid_body_view(glob)
        except Exception as e:
            _log(f"create_rigid_body_view({glob!r}) failed: {e!r} -> USD fallback")
            return False
        if rigid_view is None or rigid_view.count == 0:
            _log(f"create_rigid_body_view({glob!r}) returned empty view -> USD fallback")
            return False
        _log(f"created rigid_view for {glob!r}, count={rigid_view.count}")
        setattr(env, _CACHE_VIEW, rigid_view)
        setattr(env, _CACHE_GLOB, glob)

    n_total = rigid_view.count
    init_dict = getattr(env, _CACHE_INIT)

    # Build pose buffer for the indices we want to reset (m x 7)
    m = len(env_ids)
    pose_np = np.empty((m, 7), dtype=np.float32)
    for i, eid in enumerate(env_ids):
        cache = init_dict.get(eid)
        if cache is None:
            _log(f"no cached init pose for env {eid} -> USD fallback")
            return False
        p = cache["pose7"]
        pose_np[i] = p.cpu().numpy() if hasattr(p, "cpu") else p

    # Filter out env_ids beyond the view (shouldn't happen, but be safe)
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
        _log(f"tensor reset failed: {e!r} -> invalidating view, USD fallback")
        # Drop the cached view so it gets recreated on the next attempt
        try:
            setattr(env, _CACHE_VIEW, None)
            setattr(env, _CACHE_GLOB, None)
        except Exception:
            pass
        return False


def _usd_pose_reset(stage, prim_path, init_pos, init_rot):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(init_pos)
    xformable.AddOrientOp().Set(Gf.Quatf(
        float(init_rot.GetReal()),
        Gf.Vec3f(*[float(x) for x in init_rot.GetImaginary()]),
    ))
    return True


def _autodetect_inner_rel_path(stage, prim_path_expr) -> str | None:
    """Walk the env_0 cloth template and return the rel path of the first
    descendant that carries ``PhysicsRigidBodyAPI``. This avoids hard-coding
    ``Cloth_In001`` vs ``Cloth_In002`` etc. across different fold variants.
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
    from pxr import Usd  # local import: pxr already pulled in at module load
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

    _log(f"autodetect FAILED under {base!r}. "
         f"Visited {len(visited)} prims, none had PhysicsRigidBodyAPI:")
    for p, schemas in visited[:30]:
        _log(f"   - {p}  schemas={schemas}")
    if len(visited) > 30:
        _log(f"   ... and {len(visited) - 30} more")
    return None


def _log(msg: str) -> None:
    """Diagnostic print that survives the omni.rtx error spam.

    Writes to *stderr* (not buffered the same way as stdout when Kit logging
    is hammering the terminal) and prefixes with a unique tag so the caller
    can ``2> >(grep CLOTHRST)`` to isolate our trace.
    """
    print(f"[CLOTHRST] {msg}", file=sys.stderr, flush=True)


def reset_cloth_inner(
    env,
    env_ids,
    cloth_asset_name: str = "tablecloth",
    inner_rel_path: str | None = None,
):
    """Reset Cloth_In rigid body back to its initial pose.

    Args:
        cloth_asset_name: name of the cloth attribute on ``env.scene.cfg``.
            Defaults to ``"tablecloth"`` to match the spread_tablecloth scene.
        inner_rel_path: relative path from the cloth spawn root to the inner
            rigid body (e.g. ``"Cloth_In002/Cloth_In002"``). When ``None``
            (default) the function walks the spawned cloth and picks the first
            child prim with ``PhysicsRigidBodyAPI`` applied; if auto-detection
            fails, falls back to ``Cloth_In001/Cloth_In001`` for backward
            compatibility with older fold04/05/06 USDs.

    NOTE: ``env_ids`` is intentionally a required positional (no default).
    IsaacLab's ``EventManager._resolve_common_term_cfg(min_argc=2)``
    expects the first two parameters (``env`` and ``env_ids``) to be
    *required* -- giving ``env_ids`` a default pushes it into the validator's
    "optional" bucket and produces a misleading
    ``expects mandatory parameters: [] ... but received: ['cloth_asset_name']``
    error.
    """
    _log(f"reset_cloth_inner called: cloth_asset_name={cloth_asset_name!r} "
         f"inner_rel_path={inner_rel_path!r} env_ids={env_ids!r}")
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    cloth_cfg = getattr(env.scene.cfg, cloth_asset_name, None)
    if cloth_cfg is None:
        return
    prim_path_expr = cloth_cfg.prim_path

    # Resolve which sub-prim to reset. Cache the resolved value on env to avoid
    # re-walking the USD on every reset.
    if inner_rel_path is None:
        inner_rel_path = getattr(env, "_cloth_inner_rel_path", None)
        if inner_rel_path is None:
            detected = _autodetect_inner_rel_path(stage, prim_path_expr)
            inner_rel_path = detected or _DEFAULT_CLOTH_IN_REL_PATH
            setattr(env, "_cloth_inner_rel_path", inner_rel_path)
            _log(f"using inner_rel_path={inner_rel_path!r} "
                 f"(autodetected={'yes' if detected else 'no, fallback to default'})")

    if env_ids is None:
        env_ids = list(range(env.num_envs))
    elif isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.detach().cpu().tolist()
    else:
        env_ids = list(env_ids)

    if not hasattr(env, _CACHE_INIT):
        setattr(env, _CACHE_INIT, {})
    init_dict = getattr(env, _CACHE_INIT)

    # First pass: make sure every requested env has a cached init pose.
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
                # PhysX tensor layout for set_transforms: [x,y,z, qx,qy,qz,qw]
                # (xyzw, NOT IsaacLab's external wxyz convention). IsaacLab
                # itself does torch.roll(quat, -1) before calling set_transforms,
                # so we just bake the xyzw order in here once.
                "pose7": torch.tensor([
                    float(t[0]), float(t[1]), float(t[2]),
                    float(q.GetImaginary()[0]),
                    float(q.GetImaginary()[1]),
                    float(q.GetImaginary()[2]),
                    float(q.GetReal()),
                ], dtype=torch.float32),
            }
            _log(f"cached env {env_id} init pos={tuple(float(x) for x in t)} "
                 f"quat(xyzw)=({float(q.GetImaginary()[0]):.4f},"
                 f"{float(q.GetImaginary()[1]):.4f},"
                 f"{float(q.GetImaginary()[2]):.4f},"
                 f"{float(q.GetReal()):.4f})")

        valid_env_ids.append(env_id)
        inner_paths.append(path)

    if not valid_env_ids:
        _log("no valid env_ids found (all prims invalid?), nothing to do")
        return

    # Path A: PhysX tensor view (works on GPU + suppressReadback=True).
    if _try_physx_tensor_reset(env, valid_env_ids, prim_path_expr, inner_rel_path):
        return

    # Path B: USD xform writes + flush.
    # Only effective at startup before play, or when suppressReadback is False.
    _log(f"USD fallback path for env_ids={valid_env_ids}")
    for env_id, path in zip(valid_env_ids, inner_paths):
        cache = init_dict.get(env_id)
        if cache is None:
            continue
        _usd_pose_reset(stage, path, cache["pos_gf"], cache["rot_gf"])

    try:
        import omni.physx
        omni.physx.get_physx_interface().update_transformations(
            updateToFastCache=True, updateToUsd=False,
            updateVelocitiesToUsd=False, outputVelocitiesLocalSpace=False,
        )
    except Exception as e:
        _log(f"physx update_transformations failed: {e!r}")
