# SPDX-License-Identifier: Apache-2.0
"""RL-owned PPO safeguards for the BC-initialized liver experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO

REQUIRED_GATES = (
    "dependency",
    "actions",
    "isolation_1_8_16",
    "replay",
    "scoring",
    "bc_parity",
    "zero_lr",
    "critic_invariance",
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GuardedPPO(PPO):
    """Fixed initial std, critic warm-up and full-rollout analytic KL rollback.

    Evaluation checkpoints at actor updates 1, 5, 10 are mandatory. The owning
    training driver must supply the same frozen 20-start evaluator used for BC.
    """

    def __init__(self, *args, critic_warmup_updates=10, qualification_required=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_updates = critic_warmup_updates
        self.completed_updates = self.actor_updates = 0
        self.evaluator = None
        self.baseline_successes = None
        self.history = []
        self.stopped = False
        self.qualification = None
        self.resume_qualifications = []
        self._liver_env = None
        if qualification_required:
            path = os.environ.get("I4H_LIVER_QUALIFICATION", "")
            if not path:
                raise RuntimeError("liver PPO requires the completed qualification manifest")
            self.qualification = json.loads(Path(path).read_text())
            if any(self.qualification.get("gates", {}).get(name) is not True for name in REQUIRED_GATES):
                raise RuntimeError("liver PPO qualification gates are incomplete")
            from i4h_common.paths import workflow_root

            for relative, expected_hash in self.qualification["qualified_source_hashes"].items():
                if file_hash(Path(workflow_root()) / relative) != expected_hash:
                    raise ValueError(f"qualified liver source changed: {relative}")
            bc = Path(os.environ["I4H_LIVER_BC"])
            if file_hash(bc) != self.qualification["bc_checkpoint_sha256"]:
                raise ValueError("qualified BC checkpoint hash mismatch")
            self.actor.load_state_dict(
                torch.load(bc, weights_only=False, map_location=self.device)["actor"], strict=True
            )
        # Keep the complete distribution fixed initially, including grip noise.
        for parameter in self.actor.distribution.parameters():
            parameter.requires_grad_(False)

    def update(self):
        if self.stopped:
            raise RuntimeError("liver PPO stopped after deterministic evaluation regression")
        warmup = self.completed_updates < self.warmup_updates
        next_actor = self.actor_updates + (0 if warmup else 1)
        if not warmup and next_actor in (1, 5, 10) and not callable(self.evaluator):
            raise RuntimeError("actor update requires the frozen 20-start BC comparison evaluator")
        actor_parameters = list(self.actor.parameters())
        old_actor = copy.deepcopy(self.actor.state_dict())
        old_optimizer = {p: copy.deepcopy(self.optimizer.state.get(p)) for p in actor_parameters}
        old_lrs = [group["lr"] for group in self.optimizer.param_groups]
        old_lr = self.learning_rate
        observation = self.storage.observations.flatten(0, 1).clone()
        with torch.no_grad():
            self.actor(observation, stochastic_output=True)
            old_distribution = tuple(value.detach().clone() for value in self.actor.output_distribution_params)
        flags = [p.requires_grad for p in actor_parameters]
        schedule = self.schedule
        if warmup:
            for p in actor_parameters:
                p.requires_grad_(False)
            self.schedule = "fixed"
        try:
            result = super().update()
        finally:
            self.schedule = schedule
            for p, flag in zip(actor_parameters, flags, strict=True):
                p.requires_grad_(flag)
        with torch.no_grad():
            self.actor(observation, stochastic_output=True)
            kl = float(self.actor.get_kl_divergence(old_distribution, self.actor.output_distribution_params).mean())
        rollback = not torch.isfinite(torch.tensor(kl)) or kl > 0.02
        if rollback:
            self.actor.load_state_dict(old_actor)
            for p, state in old_optimizer.items():
                if state is None:
                    self.optimizer.state.pop(p, None)
                else:
                    self.optimizer.state[p] = state
            for group, lr in zip(self.optimizer.param_groups, old_lrs, strict=True):
                group["lr"] = lr
            self.learning_rate = old_lr
        if warmup and any(not torch.equal(value, self.actor.state_dict()[key]) for key, value in old_actor.items()):
            raise RuntimeError("critic warm-up mutated actor/distribution/normalization state")
        self.completed_updates += 1
        if not warmup:
            self.actor_updates = next_actor
        result.update(update_kl=kl, actor_rollback=float(rollback), critic_only=float(warmup))
        if not warmup and self.actor_updates in (1, 5, 10):
            score = self.evaluator(self.actor)
            if score["episodes"] != 20 or score["starts_sha256"] != self.qualification["validation_starts_sha256"]:
                raise RuntimeError("evaluation did not use the same 20-start BC baseline")
            if score["objective"] != self.qualification["objective"]:
                raise RuntimeError("evaluation objective mismatch")
            self.stopped = score["successes"] < self.qualification["bc_successes"] - 2
            result["deterministic_successes"] = score["successes"]
        self.history.append(dict(update=self.completed_updates, actor_update=self.actor_updates, **result))
        if not warmup and self.actor_updates in (1, 5, 10) and hasattr(self, "_validation_folder"):
            torch.save(
                dict(**self.save(), iter=self.completed_updates, infos={"validation_pending": False}),
                self._validation_folder / ("stopped_checkpoint.pt" if self.stopped else "validated_checkpoint.pt"),
            )
        if self.stopped:
            raise RuntimeError("deterministic success fell by more than two episodes against BC; stop training")
        return result

    def save(self):
        result = super().save()
        import random

        import numpy as np

        result["liver_rng"] = dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        )
        if self._liver_env is not None:
            result["liver_environment"] = self._liver_env.unwrapped.capture_complete_state()
        result["liver_safeguards"] = dict(
            completed_updates=self.completed_updates,
            actor_updates=self.actor_updates,
            warmup_updates=self.warmup_updates,
            qualification=self.qualification,
            history=self.history,
            stopped=self.stopped,
            resume_qualifications=self.resume_qualifications,
        )
        return result

    def load(self, loaded_dict, load_cfg, strict):
        if (loaded_dict.get("infos") or {}).get("validation_pending"):
            raise RuntimeError("checkpoint awaited mandatory evaluation; resume a finalized validated checkpoint")
        result = super().load(loaded_dict, load_cfg, strict)
        state = loaded_dict.get("liver_safeguards")
        if state is not None:
            previous = state["qualification"]
            if self.qualification is not None and previous != self.qualification:
                # The constructor verifies all current source hashes and gates.
                # A requalified source revision may continue the same experiment;
                # retain both qualifications and reject changed objectives/pins/starts.
                stable = (
                    "objective",
                    "numerical_revision",
                    "versions",
                    "bc_checkpoint_sha256",
                    "validation_starts_sha256",
                    "bc_successes",
                )
                if any((previous or {}).get(key) != self.qualification.get(key) for key in stable):
                    raise ValueError("resume qualification experiment mismatch")
                self.resume_qualifications.append(previous)
            self.resume_qualifications.extend(state.get("resume_qualifications", []))
            self.completed_updates, self.actor_updates = state["completed_updates"], state["actor_updates"]
            self.warmup_updates, self.history, self.stopped = (
                state["warmup_updates"],
                state["history"],
                state["stopped"],
            )
        if "liver_rng" in loaded_dict:
            import random

            import numpy as np

            rng = loaded_dict["liver_rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if rng["cuda"]:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        if "liver_environment" in loaded_dict and self._liver_env is not None:
            self._liver_env.unwrapped.restore_complete_state(loaded_dict["liver_environment"])
        return result

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        alg._liver_env = env if hasattr(env, "unwrapped") else None
        if alg.qualification is not None:
            objective = env.unwrapped.liver.objective.objective
            if alg.qualification["objective"] != objective:
                raise ValueError("PPO objective differs from its qualified BC baseline")
            alg.evaluator = alg._evaluate_in_fresh_process
        return alg

    def _evaluate_in_fresh_process(self, actor):
        """Keep evaluation resets and clocks out of the training environment."""
        import subprocess
        import sys

        from i4h_arena.medical.liver import ENCODER
        from i4h_common.paths import workflow_root

        root = Path(workflow_root())
        folder = (
            Path(os.environ.get("I4H_LIVER_EXPERIMENT_DIR", root / "outputs/liver_actor_validation"))
            / f"update_{self.actor_updates:02d}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        self._validation_folder = folder
        torch.jit.script(actor.as_jit().cpu()).save(str(folder / "policy.pt"))
        (folder / "contract.json").write_text(
            json.dumps(dict(objective=self.qualification["objective"], encoder=ENCODER), indent=2) + "\n"
        )
        # Preserve a resumable checkpoint before an evaluation can stop learning.
        torch.save(
            dict(**self.save(), iter=self.completed_updates, infos={"validation_pending": True}),
            folder / "checkpoint.pt",
        )
        starts = Path(self.qualification["validation_starts"])
        if file_hash(starts) != self.qualification["validation_starts_sha256"]:
            raise ValueError("validation starts changed since BC qualification")
        command = [
            sys.executable,
            "-m",
            "i4h_rl.liver_experiment",
            "evaluate",
            "--worlds",
            "16",
            "--checkpoint",
            str(folder / "policy.pt"),
            "--starts",
            str(starts),
            "--output",
            str(folder / "evaluation.json"),
            "--headless",
        ]
        with (folder / "evaluation.log").open("w") as log:
            subprocess.run(command, cwd=root, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads((folder / "evaluation.json").read_text())
