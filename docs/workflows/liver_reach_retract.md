# Liver reach and retract

`liver_reach_retract` uses Numerical Sim mechanics, the i4h–Isaac Lab host lifecycle, Arena scoring, normal Workflow Tasks and the Workflows RSL-RL path.

The implementation and integration checks are complete. **Full-cycle PPO qualification failed:** the required validation guard stopped the campaign at actor update 5 (12/20 successes versus the same-start 20/20 BC baseline). No further full-cycle actor updates or mechanics tuning were performed. The matched reserved comparison was controller **100/100**, BC **94/100**, and the stopped PPO actor **59/100**. The separate reach-only and grasped-retract curricula each retained 20/20 validation success at actor updates 1, 5 and 10.

Detailed results, per-target Wilson intervals, failures, physical diagnostics and provenance are in [`outputs/liver_import/results.md`](../../outputs/liver_import/results.md), [`comparison.json`](../../outputs/liver_import/comparison.json) and [`provenance.json`](../../outputs/liver_import/provenance.json). These are local experiment artifacts, separate from source control.

## Ownership

- Numerical Sim owns mechanics, contact, both grasp mechanisms, raw probe measurements and numerical continuation.
- The explicitly shipped [`i4h_numerical_liver_host.patch`](../../third_party/i4h_numerical_liver_host.patch) adds pre-capture post-physics callback registration and selected-world restore hooks to the pinned bridge. Ordinary reset retains its original ordering. Snapshot IDs bypass ordinary particle-history/procedure initialization after events.
- [`medical/liver.py`](../../arena/i4h_arena/medical/liver.py) owns the Arena phase reducer, failure latches, physical extrema and fixed 57-feature encoder. Warp callbacks run after both 60 Hz physics steps. Observation, reward and success getters do not advance objective state.
- [`envcfg/liver_reach_retract/`](../../arena/i4h_arena/envcfg/liver_reach_retract/) composes the virtual grasper and TaskBase. The reducer supplies success directly, without a second Arena progress clock or an injected training recorder.
- Controller and exported-policy Tasks read `ctx.scene`, write `ctx.act` and never step physics. The controller owns command hysteresis only; it has no objective phase, deadline or failure latch. Its commands match the pinned reference.
- [`rl/i4h_rl/liver_ppo.py`](../../rl/i4h_rl/liver_ppo.py) owns BC initialization, critic warm-up, fixed initial distribution, KL rollback, mandatory evaluation and checkpoint continuation.

## Fixed experiment contract

Targets are `252, 257, 255, 1688, 1098`. Physics is 60 Hz, control 30 Hz, decimation 2, with eight internal substeps and iterations. Four normalized channels command Cartesian displacement and grip. The displacement vector is capped at 1 mm once in `process_actions()`; both `apply_actions()` calls write the held absolute target. Orientation stays at its demonstrated value; energy and fire stay zero.

Success requires intended-patch acquisition, 0.5 s retention, a frozen +20 mm Y destination, one continuous second within 2 mm, clearing both grasp counts and bindings within 0.5 s of opening, then one second open relaxation. The complete attempt has a 20 s deadline classified as failure. Wrong-patch capture, retention loss, invalid state/workspace and premature opening cannot succeed. Acquisition identities never change during an attempt. Soft-attachment drift and inverted tetrahedra remain measured limitations.

The exported `contract.json` names all 57 feature groups, offsets, dimensions, physical scaling, phase order, action mapping and physical criteria. Actor and critic empirical normalization are disabled. Reward revision `valid_patch_capped_lift_no_retention_v3` preserves valid-patch acquisition, capped lift progress and completion pulses, removes repeated positive retention payment, and accounts for Isaac Lab's `step_dt` multiplier.

## Dependency and input provenance

Workflows starts at `1208154b83180901aedcf8eb23371c9045ea3d88`. Numerical Sim is a detached dependency at `7edecea49204539ca54cf9da736b5bcf597dadff`, with only the recorded bridge patch. Isaac Lab `39394453ceaac256349ad822e675a93485373667`, Arena `196680c28aa53e173d97d9e2dbded9d8d4f981a0`, Newton and installed wheels remain pinned. The dependency has its own Git objects and imports from `third_party/i4h-numerical-sim-7edecea`, independently of the source worktree.

The user-provided commit contains `liver_pinned_grasp_1x.yaml`; the uncommitted generic `case.yaml` is not used. The five manual calibration examples are embedded in [`calibration.json`](../../arena/i4h_arena/envcfg/liver_reach_retract/calibration.json). The existing BC checkpoint and a frozen observation sample were copied into experiment inputs and hashed. No runtime package imports uncommitted recordings or the liver source worktree.

`I4H_DATA` may select the asset root. The local experiment uses the committed assets under the detached dependency's `numerical_sim/data` directory. Snapshot identity includes asset bytes, model/configuration identity, numerical source hashes and dependency identity. A different asset root/configuration must qualify its own compatible bundles.

## Offline snapshots and qualification

Run from the Workflows repository root with the installed Arena environment:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/arena:$PWD/engine:$PWD/common:$PWD/workflows:$PWD/rl"
export I4H_LIVER_SNAPSHOTS="$PWD/outputs/liver_snapshots"
mkdir -p outputs/liver_import

arena/.venv/bin/python scripts/prepare_liver_snapshots.py \
  --worlds 16 --output "$I4H_LIVER_SNAPSHOTS" --headless
arena/.venv/bin/python -m i4h_rl.liver_experiment prepare-grasped \
  --worlds 16 --output "$I4H_LIVER_SNAPSHOTS" --headless
arena/.venv/bin/python -m i4h_rl.liver_experiment qualify \
  --worlds 16 --output outputs/liver_import/isolation_16.json --headless
arena/.venv/bin/python -m i4h_rl.liver_experiment fresh-replay \
  --worlds 16 --observations outputs/liver_import/isolation_16.npz \
  --output outputs/liver_import/fresh_replay_16.json --headless
arena/.venv/bin/python scripts/qualify_liver_grasped_isolation.py \
  --worlds 16 --output outputs/liver_import/grasped_isolation_16.json
```

Repeat with 1 and 8 worlds. Snapshot generation refuses to overwrite existing evidence. It settles offline using completed-physics-step measurements; reset callbacks never step physics. Grasped starts are generated with the reference controller and stored with versioned manifests and file hashes. A bundle from another model/world count, a changed target mapping, or a hash/layout mismatch is rejected.

Ordinary reset and snapshot restore preserve every peer continuation row, grasp identity, probe latch, objective/action history and RNG state. Complete-state HDF5 recording restores numerical continuation after host asset reset/kinematic synchronization. Recorded layouts are not broadcast into incompatible world counts.

## Normal Workflow execution

```bash
./run.sh liver_reach_retract --idle
./run.sh liver_reach_retract --rule-based --episodes 1 --attempts 1 \
  --record controller.hdf5 --record-failures
./run.sh liver_reach_retract --policy \
  --checkpoint outputs/liver_import/normal_export/policy.pt \
  --episodes 1 --attempts 1 --record policy.hdf5 --record-failures
./run.sh liver_reach_retract \
  --replay outputs/liver_import/visible_controller/workflow.hdf5 \
  --episodes 1 --attempts 1 --record replay.hdf5
```

The local exported PPO artifact is the **stopped, regressed** actor retained for review. Its successful single Workflow episode does not supersede the failed training qualification. Full-cycle Workflow success delegates to Arena and rejects curriculum objectives/exports.

The visible diagnostic display draws actual tissue and virtual-grasper sphere geometry. Headless training omits display copies. Viewport captures and complete recordings are retained under `outputs/liver_import/visible_controller/` and canonical `runs/liver_reach_retract/<timestamp>/` directories.

## RL path and safeguards

[`rl/profiles/liver_reach_retract.yaml`](../../rl/profiles/liver_reach_retract.yaml) selects the shared custom host. [`liver_reach_retract_ppo_rsl_rl.yaml`](../../rl/config/liver_reach_retract_ppo_rsl_rl.yaml) declares two 128-unit ELU layers, 16 environments, 256 control ticks per rollout, actor LR `1e-5`, desired KL `.01`, fixed initial std `.02`, and no running observation normalization. Environment counts are qualified separately.

```bash
./train.sh rl show liver_reach_retract
./train.sh rl liver_reach_retract --dry-run
```

`build_liver_qualification.py` constructs the required gate manifest from retained isolation, active-grasp, replay, scoring, encoder/distribution and learning-invariance evidence. Set `I4H_LIVER_BC` to the hashed BC input and `I4H_LIVER_QUALIFICATION` to the appropriate objective's manifest. Training refuses incomplete gates or changed qualified source/BC inputs.

The completed campaigns used ten critic-only updates followed by up to ten actor updates. The actor and distribution are frozen during critic warm-up. A mean update KL above `.02` restores actor/distribution parameters and their Adam state. Deterministic evaluation runs in a fresh simulator process after actor updates 1, 5 and 10, using the exact same 20-start validation manifest as BC. A decrease exceeding two successes stops training and finalizes a stopped checkpoint. Pending-validation checkpoints cannot be resumed.

For the separate curricula, set `I4H_LIVER_OBJECTIVE=reach_only` or `grasped_retract` and use that objective's qualification. Grasped-retract restores qualified offline acquisitions. Curriculum success is explicitly labeled and is rejected when full-cycle scoring is requested.

The standard native export path produces both `policy.pt` and `contract.json`:

```bash
./train.sh rl export liver_reach_retract \
  --checkpoint outputs/liver_import/actor_validation/update_05/stopped_checkpoint.pt \
  --output-dir outputs/liver_import/normal_export
```

Native resume is supported with `./train.sh rl <workflow> --checkpoint <native.pt>`. It restores actor, critic, optimizer, safeguard counters, complete compatible environment state and RNG. Current qualification must pass; requalification may update source hashes while preserving objective, numerical revision, wheel versions, BC hash, validation starts and baseline score. Both qualifications are retained in checkpoint provenance. A stopped full-cycle checkpoint remains stopped; do not resume it.

`verify_liver_export.py` compares native and exported means, log probabilities, std channels and analytic KL on frozen inputs. `verify_liver_resume.py` checks exact hosted environment, actor/distribution and optimizer restoration without a training update.

## Validation outcome

The broad regression suite passed **1377 tests, 3 skipped**. Additional focused lifecycle/controller parity and exact-resume checks passed after source cleanup. Open and actively grasped masked reset/restore passed at 1/8/16 environments. Fresh-process initial observations were exact; the largest subsequent particle discrepancy was 0.778 micrometres against a 10 micrometre tolerance. This does not claim bitwise full-episode trajectories.

The reference controller, HDF5 replay and standard exported stopped-PPO Workflow each completed one visible episode. The source BC's seed-7341 single-world failure is retained and agrees with matched native failure. The reserved 16-world comparison is reported separately with per-target results and Wilson intervals. The failed full-cycle PPO campaign remains the completion limitation.
