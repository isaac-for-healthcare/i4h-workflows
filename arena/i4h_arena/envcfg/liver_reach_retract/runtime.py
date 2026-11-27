# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hosted liver buffers and continuation; this module never steps the simulator."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import warp as wp
from i4h_surgical.grasp_probe import GraspTargetProbe
from i4h_surgical.grasp_snapshot import HarnessSnapshot, continuation_arrays, identity
from i4h_surgical.systems.instruments.instrument_control import place_instruments

from i4h_arena.medical.liver import LiverObjective, diagnostics, encode_observation, normalized_action, settling_sample

from .kinematics import stage

CALIBRATION = Path(__file__).with_name("calibration.json")


class LiverRuntime:
    def __init__(self, env, component):
        self.env, self.component = env, component
        # Newton assigns random debug particle colors. Canonicalize this display-only
        # Model array so strict identity is reproducible across fresh processes.
        colors = getattr(component.physics.model, "particle_colors", None)
        if colors is not None:
            colors.fill_(0x8C140F)
        self.probe = p = GraspTargetProbe(component.physics)
        self.objective = task = LiverObjective(component.scene.envs, env.device, env.cfg.liver_objective)
        self.examples = json.loads(CALIBRATION.read_text())["examples"]
        if [x["particle_ids"][0] for x in self.examples] != [252, 257, 255, 1688, 1098]:
            raise ValueError("liver calibration target contract changed")
        self.q = component.scene.field("q", p.body)
        self.settling_previous = wp.clone(self.q)
        self.settling_state = wp.zeros((component.scene.envs, 2), dtype=int, device=component.scene.device)
        self.root = wp.to_torch(p.ic.root)[:, 0]
        self.physical = wp.to_torch(p.ic.actions)[:, 0]
        self.views = p.torch_views()
        E = component.scene.envs
        self.previous_action = torch.zeros((E, 4), device=env.device)
        self.previous_target = torch.zeros((E, 3), device=env.device)
        self.previous_root = torch.zeros((E, 3), device=env.device)
        self.held = self.physical.clone()
        self.controller_action = self.physical.clone()
        self.controller_arrival = torch.zeros((E, 1), dtype=torch.int32, device=env.device)
        self.reward = torch.zeros(E, device=env.device)
        self.before_phase = task.s[:, 0].clone()
        self.before_lift = task.f[:, 6].clone()
        self.before_distance = torch.zeros(E, device=env.device)
        self.target_index = torch.zeros(E, dtype=torch.long, device=env.device)
        self.rng = [np.random.default_rng(seed) for seed in np.random.SeedSequence(env.cfg.seed).spawn(E)]
        self.next_specs = {}
        self.snapshot = None
        self.grasped_bundles = {}
        self._identity = None
        self.teachers = [None] * E
        pins = p.body.pin.index[0]
        self.attached = wp.array(pins[pins >= 0], dtype=int, device=component.scene.device)
        attached_rest = p.rest.copy()
        attached_rest[pins[pins >= 0]] = p.body.pin.rest[0]
        self.rest = wp.array(attached_rest, dtype=wp.vec3, device=component.scene.device)
        self.tets = wp.array(p.body.tet.tets[0], dtype=int, device=component.scene.device)
        self.rest_volume = wp.array(p.body.tet.rest_volume[0], dtype=float, device=component.scene.device)
        # No episode is active until its offline snapshot has been restored.
        task.s[:, 0] = 7
        component.order_inputs()
        component.on_post_physics(self.sample)

    @property
    def snapshot_path(self):
        return Path(self.env.cfg.liver_snapshot_dir) / f"settled_{self.env.num_envs}.npz"

    def configuration_identity(self):
        if self._identity is None:
            cfg = self.env.cfg
            self._identity = identity(
                self.component.physics,
                dict(
                    case=self.component.solver_cfg.case,
                    worlds=self.env.num_envs,
                    host="arena_liver_v1",
                    physics_hz=60,
                    decimation=cfg.decimation,
                ),
                os.environ["I4H_DATA"],
            )
        return self._identity

    def sample(self):
        wp.launch(
            settling_sample,
            dim=self.env.num_envs,
            inputs=[self.q, self.settling_previous, self.settling_state],
            device=self.component.scene.device,
        )
        self.probe.sample()
        self.objective.sample(self.probe, self.q)
        wp.launch(
            diagnostics,
            dim=self.env.num_envs,
            inputs=[
                self.objective.s_wp,
                self.objective.f_wp,
                self.probe.stats,
                self.probe.ic.actions,
                self.q,
                self.attached,
                self.rest,
                self.tets,
                self.rest_volume,
                self.probe.counts,
            ],
            device=self.component.scene.device,
        )

    def observation(self):
        self.component.sync_views()
        v = self.views
        return encode_observation(
            v["geometry"],
            self.root,
            self.physical,
            v["jaw_state"],
            v["counts"],
            v["stats"],
            v["mechanism_counts"],
            self.previous_action,
            self.previous_target,
            self.previous_root,
            self.objective.s,
            self.objective.f,
        )

    def process(self, actions):
        self.component.sync_views()
        # Accumulate exactly once per control tick. apply_actions only writes held.
        applied = normalized_action(actions)
        self.before_phase.copy_(self.objective.s[:, 0])
        self.before_lift.copy_(self.objective.f[:, 6])
        self.before_distance.copy_(
            torch.linalg.vector_norm(self.views["geometry"][:, 0] - self.views["geometry"][:, 2], dim=-1)
        )
        self.previous_target.copy_(self.views["geometry"][:, 0])
        self.previous_root.copy_(self.root)
        self.previous_action.copy_(applied)
        self.held[:, :3] += applied[:, :3] * 0.001
        self.held[:, 7] = (applied[:, 3] + 1) / 2
        self.held[:, 8:] = 0

    def apply(self):
        self.component.write_actions(self.held)

    def control_reward(self):
        # Pure function of saved pre-action values and completed physics state.
        self.component.sync_views()
        s, f, v = self.objective.s, self.objective.f, self.views
        distance = torch.linalg.vector_norm(v["geometry"][:, 0] - v["geometry"][:, 2], dim=-1)
        reward = ((self.before_distance - distance) / 0.001).clamp(-1, 1) * 0.1
        reward += ((f[:, 6].clamp(0, 0.02) - self.before_lift.clamp(0, 0.02)) / 0.001).clamp(-1, 1) * 0.2
        valid = (v["counts"][:, 1] > 0) & (v["counts"][:, 5] == 0)
        reward += ((self.before_phase < 2) & (s[:, 4] >= 0) & valid) * 2
        reward += ((self.before_phase != 5) & (s[:, 0] == 5)) * 3
        reward += (s[:, 0] == 7) * 10 - (s[:, 0] == 8) * 3
        # Lab multiplies by step_dt; preserve the per-control-tick reward.
        return reward / self.env.step_dt

    def terminal(self, ids):
        self.component.sync_views()
        done = ids[self.objective.s[ids, 0] >= 7].tolist()
        # Construction/prepare has no episode; acquisition target -1 is unscored.
        done = [e for e in done if int(self.views["target"][e]) >= 0]
        rows = self.objective.results(done, self.views["target"].tolist(), self.views["geometry"])
        for row in rows:
            e = row["env_id"]
            row["acquisition_size"] = int(self.views["counts"][e, 0])
            row["original_patch_size"] = int(self.views["counts"][e, 1])
            row["final_retained_particles"] = int(self.views["counts"][e, 3])
            if row["original_patch_size"] == 0:
                row["minimum_retained_fraction"] = None
        return rows

    def restore(self, ids, specs=None):
        if self.objective.objective == "grasped_retract":
            return self.restore_grasped(ids, specs)
        if self.snapshot is None:
            self.snapshot = HarnessSnapshot(self.component.physics, self.configuration_identity(), {})
            try:
                self.snapshot.load(self.snapshot_path)
            except ValueError:
                self.snapshot_path.with_name(f"rejected_identity_{self.env.num_envs}.json").write_text(
                    json.dumps(self.configuration_identity(), indent=2) + "\n"
                )
                raise
        # Lab has already reset host assets/events. Flush kinematics and pending resets first.
        self.env.sim.forward()
        self.component.order_inputs()
        self.snapshot.restore({}, self.probe, ids.tolist())
        self.component.sync_views()
        self.objective.reset(ids)
        physical = self.physical.clone()
        from i4h_arena.medical.liver_controller import LiverReferenceController

        for i, e in enumerate(ids.tolist()):
            spec = (
                specs[i]
                if specs is not None
                else {"target_index": int(self.rng[e].integers(5)), "jitter_m": self.rng[e].uniform(-0.005, 0.005, 3)}
            )
            index = spec["target_index"]
            ex = self.examples[index]
            self.target_index[e] = index
            target = ex["particle_ids"][0]
            self.probe.select(target, e)
            position = self.q.numpy()[e, target]
            goal = position + np.asarray(ex["capture_root"]) - ex["capture_particle_positions"][0]
            command = np.zeros(10, np.float32)
            command[:3] = goal + [0.0, 0.0, 0.03] + np.asarray(spec["jitter_m"])
            command[3:7] = ex["capture_quaternion_xyzw"]
            points = np.asarray([command[:3], goal, goal + [0.0, 0.02, 0.0]])
            workspace = np.array([points.min(0) - 0.02, points.max(0) + 0.02])
            self.objective.workspace[e] = torch.as_tensor(workspace, device=self.env.device)
            self.teachers[e] = LiverReferenceController(command, workspace, ex)
            physical[e] = torch.as_tensor(command, device=self.env.device)
            self.probe.arm(e)
        self.physical[ids] = physical[ids]
        self.held[ids] = physical[ids]
        self.controller_action[ids] = physical[ids]
        self.controller_arrival[ids] = 0
        self.previous_action[ids] = 0
        # Initialize only the tool's kinematics/history after its commanded start changes;
        # the settled tissue and every continuation-managed history field stay restored.
        self.component.order_inputs()
        mask_t = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        mask_t[ids] = True
        mask = wp.from_torch(mask_t)
        self.component.order_inputs()
        p = self.probe
        wp.launch(
            stage,
            dim=self.env.num_envs,
            inputs=[mask, p.ic.actions, p.ic.trocar, p.ic.has_trocar],
            outputs=[p.ic.root, p.ic.shaft_q, p.ic.wrist_q, p.ic.jaw_q],
            device=self.component.scene.device,
        )
        place_instruments(p.ic, self.component.scene, mask, 1)
        p.sample()
        self.component.sync_views()
        self.previous_target[ids] = self.views["geometry"][ids, 0]
        self.previous_root[ids] = self.root[ids]
        self.before_phase[ids] = 0
        self.before_lift[ids] = 0
        self.before_distance[ids] = torch.linalg.vector_norm(
            self.views["geometry"][ids, 0] - self.views["geometry"][ids, 2], dim=-1
        )
        self.component.order_inputs()

    def teacher_action(self):
        self.component.sync_views()
        self.probe.check()
        result = np.zeros((self.env.num_envs, 4), np.float32)
        # The reference only reads geometry. Download each required batch once;
        # other probe fields include per-particle capture history and diagnostics.
        geometry = self.views["geometry"].cpu().numpy()
        physical = self.physical.cpu().numpy()
        commands = self.controller_action.cpu().numpy().copy()
        arrivals = self.controller_arrival.cpu().numpy().copy()
        states = self.objective.s.cpu().numpy()
        destinations = self.objective.f[:, :3].cpu().numpy()
        from i4h_arena.medical.liver import PHASES

        for e, teacher in enumerate(self.teachers):
            teacher.action[:] = commands[e]
            teacher.arrival_ticks = int(arrivals[e, 0])
            # The Arena reducer supplies phase and timing; the controller only commands.
            destination = None if states[e, 4] < 0 else destinations[e].copy()
            command = teacher.command({"geometry": geometry[e]}, PHASES[states[e, 0]], destination)
            result[e, :3] = (command[:3] - physical[e, :3]) / 0.001
            result[e, 3] = command[7] * 2 - 1
            commands[e] = command
            arrivals[e, 0] = teacher.arrival_ticks
        self.controller_action.copy_(torch.as_tensor(commands, device=self.env.device))
        self.controller_arrival.copy_(torch.as_tensor(arrivals, device=self.env.device))
        return result

    def capture_state(self):
        """Complete model-compatible numerical, probe, objective and action continuation."""
        self.component.sync_views()
        numerical = {
            k: a.numpy().copy()
            for k, a in continuation_arrays(
                self.component.physics, self.component.physics.state, self.component.physics.control
            ).items()
        }
        probe = self.probe.read()
        tensors = {name: value.detach().cpu().numpy().copy() for name, value in self.state_tensors().items()}
        import hashlib

        manifest = hashlib.sha256(json.dumps(self.configuration_identity(), sort_keys=True).encode()).hexdigest()
        return {
            "numerical": numerical,
            "probe": probe,
            "task": tensors,
            "identity": np.frombuffer(manifest.encode(), dtype=np.uint8)[None, :],
            "rng": self.rng_state(),
        }

    def state_tensors(self):
        return dict(
            s=self.objective.s,
            f=self.objective.f,
            workspace=self.objective.workspace,
            held=self.held,
            previous_action=self.previous_action,
            previous_target=self.previous_target,
            previous_root=self.previous_root,
            before_phase=self.before_phase,
            before_lift=self.before_lift,
            before_distance=self.before_distance,
            target_index=self.target_index,
            controller_action=self.controller_action,
            controller_arrival=self.controller_arrival,
            episode_length=self.env.episode_length_buf,
            manager_action=self.env.action_manager.action,
            manager_previous_action=self.env.action_manager.prev_action,
        )

    def rng_state(self):
        encoded = [json.dumps(rng.bit_generator.state, sort_keys=True).encode() for rng in self.rng]
        state = np.zeros((len(encoded), 512), np.uint8)
        length = np.zeros((len(encoded), 1), np.int32)
        for e, value in enumerate(encoded):
            if len(value) > 512:
                raise ValueError("RNG state exceeds schema capacity")
            state[e, : len(value)] = np.frombuffer(value, np.uint8)
            length[e, 0] = len(value)
        return dict(state=state, length=length)

    def restore_complete_state(self, bundle, ids):
        """Validate all payloads before mutation; no reset/history operation follows."""
        import hashlib

        expected = hashlib.sha256(json.dumps(self.configuration_identity(), sort_keys=True).encode()).hexdigest()
        if np.asarray(bundle["identity"]).tobytes().decode() != expected:
            raise ValueError("recorded complete-state model/source/assets/world count mismatch")
        state = self.component.physics.model.state()
        control = self.component.physics.model.control(clone_variables=True)
        arrays = continuation_arrays(self.component.physics, state, control)
        families = {
            "numerical": arrays,
            "probe": {k: getattr(self.probe, k) for k in self.probe.fields},
            "task": self.state_tensors(),
        }
        for family, destinations in families.items():
            if set(destinations) != set(bundle[family]):
                raise ValueError(f"recorded {family} layout mismatch")
            for key, destination in destinations.items():
                expected_array = (
                    destination.detach().cpu().numpy() if isinstance(destination, torch.Tensor) else destination.numpy()
                )
                source = np.asarray(bundle[family][key])
                if source.shape != expected_array.shape or source.dtype != expected_array.dtype:
                    raise ValueError(f"recorded {family}/{key} shape/dtype mismatch")
        rng = bundle["rng"]
        if rng["state"].shape != (self.env.num_envs, 512) or rng["length"].shape != (self.env.num_envs, 1):
            raise ValueError("recorded RNG layout mismatch")
        decoded = [json.loads(np.asarray(rng["state"][e, : int(rng["length"][e, 0])]).tobytes()) for e in ids.tolist()]
        self.env.sim.forward()
        for key, array in arrays.items():
            array.assign(bundle["numerical"][key])
        self.component.restore(state, control, ids)
        self.component.sync_views()
        for name in self.probe.fields:
            view = self.views[name]
            view[ids] = torch.as_tensor(bundle["probe"][name], device=self.env.device)[ids]
        for name, view in self.state_tensors().items():
            view[ids] = torch.as_tensor(bundle["task"][name], device=self.env.device)[ids]
        from i4h_arena.medical.liver_controller import LiverReferenceController

        for e in ids.tolist():
            index = int(self.target_index[e])
            self.teachers[e] = LiverReferenceController(
                self.held[e].cpu().numpy().copy(),
                self.objective.workspace[e].cpu().numpy().copy(),
                self.examples[index],
            )
        for e, state in zip(ids.tolist(), decoded, strict=True):
            self.rng[e].bit_generator.state = state
        self.component.order_inputs()

    def restore_grasped(self, ids, specs=None):
        # Bundles are generated offline with the full-cycle reference. Never
        # acquire a grasp or step physics inside a reset callback.
        for i, e in enumerate(ids.tolist()):
            index = specs[i]["target_index"] if specs is not None else int(self.rng[e].integers(5))
            if index not in self.grasped_bundles:
                path = self.snapshot_path.with_name(f"grasped_{self.env.num_envs}_{index}.npz")
                import hashlib

                manifest_path = path.with_name(f"grasped_manifest_{self.env.num_envs}.json")
                manifest = json.loads(manifest_path.read_text())
                records = [row for row in manifest["records"] if row["file"] == path.name]
                if manifest.get("schema") != 1 or manifest["worlds"] != self.env.num_envs or len(records) != 1:
                    raise ValueError("grasped snapshot manifest layout mismatch")
                record = records[0]
                if (
                    record["target"] != self.examples[index]["particle_ids"][0]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
                ):
                    raise ValueError("grasped snapshot target/hash mismatch")
                with np.load(path, allow_pickle=False) as saved:
                    bundle = {"identity": saved["identity"].copy()}
                    for key in saved.files:
                        if "/" in key:
                            family, name = key.split("/", 1)
                            bundle.setdefault(family, {})[name] = saved[key].copy()
                self.grasped_bundles[index] = bundle
            sampler_state = self.rng[e].bit_generator.state
            self.restore_complete_state(self.grasped_bundles[index], ids.new_tensor([e]))
            # A new curriculum episode continues its sampler sequence; replay's
            # complete-state restoration still restores every RNG byte.
            self.rng[e].bit_generator.state = sampler_state
            self.env.episode_length_buf[e] = 0
