# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7/RLinf mapping for the catheter navigation Scene.

This post-trains a checkpoint already fine-tuned on catheter demonstrations.
The state and action widths below are the catheter's, not a generic robot's,
and a checkpoint trained against any other modality will fail to load rather
than train badly -- which is the preferable failure.

Two things differ from the Trocar adapter and are worth stating.

There is one camera. A C-arm is the only view an interventionalist has, so
``extra_view_images`` is empty rather than padded with a second angle that
does not exist clinically. The bridge schema keeps the key so the shape of a
log line does not change between workflows.

Nothing is padded on the action side. The G1 adapter maps 28 controlled
joints into a 43-DoF action and left-pads the rest; every one of the
catheter's four channels is commanded, so the mapping is the identity and a
pad would silently zero a real control.

The observation the policy receives is the checkpoint's, not the Scene's.
The Scene publishes fifteen numbers; the checkpoint declares nine of them,
as ``state: [catheter, carm, target, route]`` at widths 3, 1, 3 and 2. The
four drive channels and the navigation goal reach the policy; tip position
and direction do not, because the reward computes against them environment-
side and a projector sized for nine cannot read fifteen.

Which nine is a property of the checkpoint, not of this file, and handing
GR00T a group its processor does not declare is not an error that raises --
the loader finds no such key and proceeds, so the extra group is dropped in
silence while the ones it wants go missing. That is why
:func:`_assert_checkpoint_matches` reads the groups out of the checkpoint
directory at startup rather than trusting the modality config that ships
beside this module: the two are widened by the same change and agree with
each other whatever the file on disk was trained on.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from i4h_rl.rollout_monitor import RolloutProgressMonitor

logger = logging.getLogger(__name__)
_registered = False

OBS_CONVERTER = "i4h_catheter_carm"
TRAIN_TASK_ID = "I4H-Workflows-Endoluminal-Navigation-RLinf-v0"
EVAL_TASK_ID = "I4H-Workflows-Endoluminal-Navigation-RLinf-Eval-v0"

#: The arm-borne scene post-trains through this same adapter. Its Scene
#: subclasses the armless one and overrides the embodiment, the gravity and the
#: recorded joint columns -- never ``get_observation_cfg`` -- so the six
#: observation terms, the four action channels and the reward are identical.
#: The seven servo'd arm joints appear in recordings and not here: the RL
#: observation reads ``drive_state`` off the action terms, which is four values
#: whether or not something is holding the drive unit.
#:
#: Separate ids rather than reuse. The id does not pick the scene -- the
#: profile's ``scene`` does, through the simulator process -- so sharing would
#: work, but a run would then log under a name that claims the wrong scene.
ARM_TRAIN_TASK_ID = "I4H-Workflows-Endoluminal-Navigation-Arm-RLinf-v0"
ARM_EVAL_TASK_ID = "I4H-Workflows-Endoluminal-Navigation-Arm-RLinf-Eval-v0"

#: Every id this adapter answers for. One environment class serves them all.
TASK_IDS = (TRAIN_TASK_ID, EVAL_TASK_ID, ARM_TRAIN_TASK_ID, ARM_EVAL_TASK_ID)

#: Ordered as the trainer config's ``state`` slices expect. Changing this
#: order silently re-labels GR00T's state groups, so it is asserted below
#: rather than trusted.
STATE_KEYS = ("drive_state", "tip_position", "tip_direction", "target_offset", "route_state")
STATE_WIDTHS = (4, 3, 3, 3, 2)
STATE_DIM = sum(STATE_WIDTHS)

#: Insertion velocity, axial rotation rate, tip bend rate, C-arm orbit rate.
#: These are the modality group names from ``config_catheter.py``, which is
#: what the checkpoint was fine-tuned against. N1.5 emitted them with an
#: ``action.`` prefix and N1.7 emits them bare, so both spellings are accepted
#: below -- the same tolerance upstream's own N1.7 Libero converter applies.
ACTION_KEYS = ("catheter", "carm")
ACTION_WIDTHS = (3, 1)
ACTION_DIM = sum(ACTION_WIDTHS)

#: The video group the checkpoint's processor declares. The camera is named
#: once, in ``config_catheter.py``, from the camera the Scene publishes; a
#: ``_view`` suffix here reached GR00T as an undeclared key and left the
#: policy running on no image at all.
GR00T_VIDEO_KEY = "video.fluoroscopy"

GR00T_LANGUAGE_KEY = "annotation.human.task_description"

#: ``(group, start, stop)`` into the bridge state vector. Tip position and
#: direction remain simulator-only; target and route are policy inputs because
#: the task's reward and termination depend on them.
#: ``i4h_common`` is not on this venv's path, so they are restated here and
#: checked against the registered modality config at startup instead.
GR00T_STATE_GROUPS = (
    ("state.catheter", 0, 3),
    ("state.carm", 3, 4),
    ("state.target", 10, 13),
    ("state.route", 13, 15),
)

#: Number of scalar state values reaching the policy. The source slices are
#: intentionally non-contiguous because tip pose stays simulator-side.
GR00T_STATE_DIM = sum(stop - start for _key, start, stop in GR00T_STATE_GROUPS)


def _to_rgb(image: Any) -> Any:
    return image[..., :3]


def wrap_workflow_observation(obs: dict[str, Any], *, task_description: str, num_envs: int) -> dict[str, Any]:
    """Map current Workflow observation terms to the small RLinf bridge schema.

    The state is concatenated here, in the documented ``STATE_KEYS`` order,
    so that the trainer config's slices have one place to agree with.
    """
    import torch

    policy = obs.get("policy", obs)
    required = ("fluoroscopy_rgb", *STATE_KEYS)
    missing = [name for name in required if name not in policy]
    if missing:
        raise KeyError(f"Workflow catheter observation is missing {missing}; got {sorted(policy)}")
    for name, width in zip(STATE_KEYS, STATE_WIDTHS, strict=True):
        actual = policy[name].shape[-1]
        if actual != width:
            raise ValueError(f"expected catheter {name} width {width}, got {actual}")
    return {
        "main_images": _to_rgb(policy["fluoroscopy_rgb"]),
        # Kept for schema parity with multi-camera workflows; a C-arm has no
        # second view to offer.
        "extra_view_images": None,
        "states": torch.cat([policy[name] for name in STATE_KEYS], dim=-1),
        "task_descriptions": [task_description] * num_envs,
    }


def convert_workflow_obs_to_gr00t(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert the bridge schema to the GR00T N1.7 catheter modality contract."""
    import torch

    main = env_obs["main_images"]
    states = env_obs["states"]
    if not all(isinstance(value, torch.Tensor) for value in (main, states)):
        raise TypeError("Workflow catheter images and states must be torch tensors")
    if states.shape[-1] != STATE_DIM:
        raise ValueError(f"expected catheter state width {STATE_DIM}, got {states.shape[-1]}")
    state = states.unsqueeze(1).cpu().numpy()
    observation = {
        GR00T_VIDEO_KEY: main.unsqueeze(1).cpu().numpy(),
        GR00T_LANGUAGE_KEY: env_obs["task_descriptions"],
    }
    for key, start, stop in GR00T_STATE_GROUPS:
        observation[key] = state[:, :, start:stop]
    return observation


def convert_gr00t_to_workflow_action(action_chunk: dict[str, Any], chunk_size: int = 1) -> np.ndarray:
    """Map GR00T's catheter and C-arm heads into Workflow's 4-DoF action.

    No padding: the concatenation is already the full action space, so an
    unexpected width is an error rather than something to pad around.
    """
    resolved = []
    missing = []
    for key in ACTION_KEYS:
        name = next((candidate for candidate in (key, f"action.{key}") if candidate in action_chunk), None)
        if name is None:
            missing.append(key)
        else:
            resolved.append(name)
    if missing:
        raise KeyError(f"GR00T catheter action is missing {missing}; got {sorted(action_chunk)}")
    controlled = np.concatenate([np.asarray(action_chunk[key])[:, :chunk_size, :] for key in resolved], axis=-1)
    if controlled.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} catheter/C-arm channels, got {controlled.shape[-1]}")
    return controlled


def _register_gr00t_converters(simulation_io: Any) -> None:
    """Register this environment against the pinned RLinf N1.7 registries.

    Each GR00T version keeps its own action registry and reads only that one
    (``gr00t_n1d7/gr00t_action_model.py`` looks up ``ACTION_CONVERSION_N1D7``),
    so registering into the wrong version's dict fails as a missing converter
    at rollout rather than at startup. The observation registry is shared.
    """
    try:
        action_registry = simulation_io.ACTION_CONVERSION_N1D7
    except AttributeError as exc:
        raise RuntimeError("pinned RLinf does not expose the GR00T N1.7 action registry") from exc
    simulation_io.OBS_CONVERSION[OBS_CONVERTER] = convert_workflow_obs_to_gr00t
    action_registry[OBS_CONVERTER] = convert_gr00t_to_workflow_action


#: Where a GR00T fine-tuning run records the groups it normalized against.
#: Written from the dataset before the first optimizer step, so every
#: checkpoint carries it, including one abandoned after a handful of steps.
CHECKPOINT_STATISTICS = "experiment_cfg/dataset_statistics.json"


def _register_catheter_modality(full_cfg: dict[str, Any]) -> None:
    """Register the catheter modality the fine-tuned checkpoint expects.

    N1.7 replaced N1.5's ``data_config_class`` import string with modality
    configs registered against an embodiment tag, so this imports the same
    module the fine-tuning run used instead of restating the groups here. Only
    one config can be registered per tag and ``NEW_EMBODIMENT`` is the only tag
    open to us, so importing a sibling config in the same process would take
    the catheter's place.

    Both guards run: the registered config catches a typo in this module, and
    the checkpoint catches the case the registered config cannot see.
    """
    from i4h_tasks.gr00t_n17.config_catheter import CATHETER_CONFIG

    _assert_contract_matches(CATHETER_CONFIG)
    isaaclab = full_cfg.get("env", {}).get("train", {}).get("isaaclab", {})
    model_path = _resolve_checkpoint_path(full_cfg)
    _assert_checkpoint_matches(
        _checkpoint_contract(model_path, str(isaaclab.get("embodiment_tag", "new_embodiment"))),
        source=model_path,
    )


def _resolve_checkpoint_path(full_cfg: dict[str, Any]) -> str:
    """The fine-tuned checkpoint the run loads.

    The actor and the rollout name it separately, and two different
    checkpoints would train one policy against another's outputs, so they are
    checked for agreement here. An unresolved ``${...}`` on the rollout side
    means it defers to the actor, which is the intended wiring; only a second
    literal path can disagree.
    """
    actor = str(full_cfg.get("actor", {}).get("model", {}).get("model_path", "") or "").strip()
    rollout = str(full_cfg.get("rollout", {}).get("model", {}).get("model_path", "") or "").strip()
    if not actor:
        raise ValueError("trainer config sets no actor.model.model_path to check the contract against")
    if rollout and not rollout.startswith("${") and rollout != actor:
        raise ValueError(f"actor loads {actor} but rollout loads {rollout}")
    return actor


def _checkpoint_contract(model_path: str, embodiment_tag: str) -> dict[str, dict[str, int]]:
    """``{"state": {group: width}, ...}`` the checkpoint was fine-tuned on.

    Read off the normalization statistics because they are the only part of a
    checkpoint that names its groups: the processor config records widths and
    shapes but not what they are called. A group the run never saw has no
    entry to normalize against, and the length of a group's ``min`` vector is
    the width its projector was sized for.
    """
    statistics = Path(model_path).expanduser() / CHECKPOINT_STATISTICS
    if not statistics.is_file():
        raise FileNotFoundError(
            f"no {CHECKPOINT_STATISTICS} under {model_path}; a checkpoint that does not record "
            "what it was fine-tuned on cannot be checked against this adapter"
        )
    recorded = json.loads(statistics.read_text(encoding="utf-8"))
    if embodiment_tag not in recorded:
        raise ValueError(f"{statistics} records embodiments {sorted(recorded)}, not {embodiment_tag!r}")
    tagged = recorded[embodiment_tag]
    return {
        modality: {group: len(values["min"]) for group, values in tagged.get(modality, {}).items()}
        for modality in ("state", "action")
    }


def _assert_checkpoint_matches(contract: dict[str, dict[str, int]], *, source: str) -> None:
    """Fail at startup if the checkpoint was fine-tuned on different groups.

    :func:`_assert_contract_matches` compares this module against the modality
    config that ships beside it, which catches a typo here but nothing about
    the file ``model_path`` points at: one change widens both, so they agree
    with each other whatever is on disk. A checkpoint predating the widening
    loads without complaint and drops ``state.target`` and ``state.route``,
    which is exactly the silent-input failure the other guard exists to stop.

    Names and widths, not order: the statistics are a JSON object and their
    key order is the dataset's, not evidence of how the projectors are laid
    out. Order stays with the guard that reads the modality config, where it
    is declared rather than inferred.
    """
    emitted = {
        "state": {key.split(".", 1)[1]: stop - start for key, start, stop in GR00T_STATE_GROUPS},
        "action": dict(zip(ACTION_KEYS, ACTION_WIDTHS, strict=True)),
    }
    for modality, groups in emitted.items():
        trained = contract.get(modality, {})
        if groups != trained:
            raise ValueError(
                f"catheter adapter emits {modality} groups {groups} but the checkpoint at "
                f"{source} was fine-tuned on {trained}; the difference would be dropped in silence"
            )


def _assert_contract_matches(modality_config: Any) -> None:
    """Fail at startup if this module disagrees with the modality config beside it.

    The config is registered against the embodiment tag and is what names the
    groups at load time, so a renamed video key or a reordered state group
    costs the policy an entire input and shows up only as training that does
    not improve. Order is positional in GR00T, which is why this compares
    sequences rather than sets.

    Scope: this and the config are in the same source tree and move together.
    :func:`_assert_checkpoint_matches` is the one that can tell you the
    checkpoint disagrees.
    """
    expected = {
        "video": [GR00T_VIDEO_KEY],
        "state": [key for key, _start, _stop in GR00T_STATE_GROUPS],
        "action": list(ACTION_KEYS),
        "language": [GR00T_LANGUAGE_KEY],
    }
    for modality, emitted in expected.items():
        declared = list(modality_config[modality].modality_keys)
        # State and video keys carry their modality as a prefix in the
        # observation dict; the config states the bare group name.
        bare = [key.split(".", 1)[1] if key.startswith(f"{modality}.") else key for key in emitted]
        if bare != declared:
            raise ValueError(
                f"catheter adapter emits {modality} groups {bare} but the registered modality "
                f"config declares {declared}; the checkpoint would silently ignore the difference"
            )
    widths = [stop - start for _key, start, stop in GR00T_STATE_GROUPS]
    if widths != [3, 1, 3, 2]:
        raise ValueError("catheter state groups must be 3, 1, 3, and 2 wide to match the checkpoint, " f"got {widths}")


def monitor_step_threshold(cfg) -> int:
    """Steps after which one rollout epoch's metrics are final.

    The smaller of the episode length and the epoch's own step budget, because
    either can be the thing that ends collection. With ``auto_reset: false``
    RLinf runs ``max_steps_per_rollout_epoch // num_action_chunks`` chunk steps
    per epoch and resets at the top of the next one, and it asserts that the
    budget divides by the chunk width, so the budget is exactly how many env
    steps an epoch delivers.

    Reading ``max_episode_steps`` alone is what this used to do, and the two are
    equal in the shipped profiles, so the count landed on the threshold on the
    epoch's final step with nothing to spare. Lower the budget -- the obvious
    knob for a shorter rollout -- and the threshold becomes unreachable, so the
    monitor stops recording entirely and reports no error, since nothing in
    RLinf relates the two keys.
    """
    steps = [int(getattr(cfg, name, 0) or 0) for name in ("max_episode_steps", "max_steps_per_rollout_epoch")]
    reachable = [value for value in steps if value > 0]
    if not reachable:
        raise ValueError("env config declares neither max_episode_steps nor max_steps_per_rollout_epoch")
    return min(reachable)


def monitor_owns_run_dir(seed_offset: int) -> bool:
    """Whether this env worker is the one that may write the run directory.

    ``RolloutProgressMonitor`` writes a fixed ``rollout_progress.json`` and one
    TensorBoard directory, so a second worker would overwrite the first rather
    than add to it. Only one worker exists today -- the backend starts a single
    simulator behind a single bridge socket, and both profiles place ``env`` on
    one node -- so this is a guard against a placement change, not a live fault.
    """
    return int(seed_offset) == 0


def _get_workflow_env_class():
    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    class WorkflowCatheterEnv(IsaaclabBaseEnv):
        def __init__(self, *args, **kwargs):
            self._rollout_monitor = None
            self._rollout_monitor_recorded = False
            self._rollout_monitor_steps = 0
            self._rollout_monitor_threshold = 0
            seed_offset = kwargs.get("seed_offset", args[2] if len(args) > 2 else 0)
            super().__init__(*args, **kwargs)
            run_dir = os.environ.get("I4H_RL_RUN_DIR")
            if (
                run_dir
                and self.isaaclab_env_id in (TRAIN_TASK_ID, ARM_TRAIN_TASK_ID)
                and monitor_owns_run_dir(seed_offset)
            ):
                self._rollout_monitor_threshold = monitor_step_threshold(self.cfg)
                self._rollout_monitor = RolloutProgressMonitor(
                    run_dir,
                    rollouts_per_update=int(self.cfg.rollout_epoch),
                )

        def _init_isaaclab_env(self):
            from i4h_rl.sim_bridge import RemoteIsaacEnv

            self.env = RemoteIsaacEnv.from_environment()
            self.env.reset(seed=self.seed)

        def reset(self, *args, **kwargs):
            obs = super().reset(*args, **kwargs)
            self._rollout_monitor_recorded = False
            self._rollout_monitor_steps = 0
            return obs

        def _wrap_obs(self, obs):
            return wrap_workflow_observation(
                obs,
                task_description=self.task_description,
                num_envs=self.num_envs,
            )

        def _record_metrics(self, step_reward, terminations, infos):
            episode_info = {}
            self.returns += step_reward
            self.success_once = self.success_once | terminations.bool()
            episode_info["success_once"] = self.success_once.clone()
            episode_info["return"] = self.returns.clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
            infos["episode"] = episode_info
            self._rollout_monitor_steps += 1
            if (
                self._rollout_monitor is not None
                and not self._rollout_monitor_recorded
                and self._rollout_monitor_steps >= self._rollout_monitor_threshold
            ):
                self._rollout_monitor.record(
                    mean_return=episode_info["return"].float().mean().item(),
                    mean_reward=episode_info["reward"].float().mean().item(),
                    mean_episode_length=episode_info["episode_len"].float().mean().item(),
                    success_rate=episode_info["success_once"].float().mean().item(),
                )
                self._rollout_monitor_recorded = True
            return infos

        def add_image(self, obs):
            policy = obs.get("policy", obs)
            image = policy.get("fluoroscopy_rgb")
            return None if image is None else _to_rgb(image[0]).cpu().numpy()

        def close(self):
            if self._rollout_monitor is not None:
                self._rollout_monitor.close()
            super().close()

    return WorkflowCatheterEnv


def register() -> None:
    """Register catheter conversion, model loading, and environment factories."""
    global _registered
    if _registered:
        return

    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    env_class = _get_workflow_env_class()
    for task_id in TASK_IDS:
        REGISTER_ISAACLAB_ENVS[task_id] = env_class

    from isaaclab_contrib.rl.rlinf import extension as isaaclab_extension
    from rlinf.models.embodiment.gr00t import simulation_io

    cfg = isaaclab_extension._get_isaaclab_cfg()
    if cfg.get("obs_converter_type") != OBS_CONVERTER:
        raise ValueError(f"expected obs_converter_type={OBS_CONVERTER!r}, got {cfg.get('obs_converter_type')!r}")
    _register_gr00t_converters(simulation_io)
    _register_catheter_modality(isaaclab_extension._load_full_cfg())
    # With no ``data_config_class`` in the trainer config this only registers
    # the embodiment tag and returns, which is what we want: its own model
    # loader hardcodes the N1.5 class, while RLinf's default ``get_model``
    # dispatches on ``model_type`` and so builds the N1.7 one.
    isaaclab_extension._patch_gr00t_get_model(cfg)
    _registered = True
    logger.info("registered Workflow catheter RL tasks: %s", ", ".join(TASK_IDS))
