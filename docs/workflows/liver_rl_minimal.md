# Minimal liver RL example: tissue grasp and retraction

This example trains a reinforcement-learning policy for `liver_reach_retract`, end to end. It uses Isaac Lab and the Numerical Sim liver. The policy moves a virtual grasper to an intended patch of liver tissue, grasps it, retracts (mobilizes) it 20 mm, holds it, then releases it. One command runs every stage. All artifacts go into one run directory:

```bash
python3 scripts/liver_rl_minimal.py --headless
```

`--dry-run` prints every stage's command without starting Isaac Sim. To continue a run after a failure, pass `--run-dir runs/liver_reach_retract/<ts> --from-stage <stage>`. Resume verifies earlier commands and output hashes against `summary.json`; changed or unverified inputs require rerunning from the earliest affected stage. Downstream scores are invalidated when rerunning. Ctrl+C or SIGTERM stops the active stage and its launcher children.

## Requirements

- An Arena environment from `./setup.sh`.
- Numerical Sim from `isaac-for-healthcare/i4h-physics-simulation`, at the revision pinned by `third_party/setup.sh` (currently `cfc05e0d20dda2a841fdf935b666f47b27163554`). `./setup.sh` installs its packages, including the neutral `i4h-host`, from `third_party/i4h-physics-simulation/numerical_sim`; the standalone application is not needed. Access to this revision requires the physics sync PR to be available in that repository.
- Stock `warp-lang==1.17.0`, installed by setup. No patched wheel is required; the upstream captured-HashGrid issue [Warp #2426](https://github.com/NVIDIA/warp/issues/2426) concerns a path this liver fixture does not use. Historical training results below used the earlier patched binary.
- One NVIDIA RTX GPU. The run uses 8 parallel environments for demonstrations, training and evaluation, and 1 for export and the final launcher run.

## What the task is

The environment is the existing `full_cycle` objective, described in the [liver guide](liver_reach_retract.md):

- **Observation:** 57 features describing the grasper, the target patch, the jaws and the task phase.
- **Action:** 4 channels. Three are a Cartesian step of at most 1 mm per 30 Hz control tick; the fourth is the grip.
- **Success:** reach and grasp the intended patch, retain it, retract it 20 mm, hold for 1 s within 2 mm, release both grasp mechanisms within 0.5 s, then relax for 1 s.
- **Failure:** missing the 20 s deadline, the wrong patch, a premature release, or leaving the workspace.
- **Reward:** dense approach and lift progress, a bonus for a valid grasp and for release, +10 on success and −3 on failure.

## Why behaviour cloning comes first

The full cycle needs millimetre-scale contact timing, so PPO from a random actor does not find it in a short run. The scene already has a scripted reference controller that completes the cycle reliably. The example clones that controller into the RL actor, then lets PPO fine-tune it in the simulator while retaining a frozen copy of the BC policy as a reference. The additional loss is `0.1 × mean KL(BC || actor)` on each PPO minibatch; its gradient is combined with PPO before gradient clipping. The reference remains the original BC policy across checkpoint saves and resumes. This standard "imitation, then RL" recipe needs no hand-made checkpoint.

## Stages

| Stage | Command (run by the driver) | Output |
| --- | --- | --- |
| `evaluation-starts` | `i4h_rl.liver_experiment prepare-starts` | Separate frozen `starts/development.json` (20 starts) and `starts/test.json` (40 by default) |
| `snapshots-8`, `snapshots-1` | `scripts/prepare_liver_snapshots.py --worlds {8,1}` | Settled liver start states, `snapshots/settled_{8,1}.npz` |
| `demonstrations` | `i4h_rl.liver_experiment collect-demos --worlds 8 --episodes 160` | Reference observations and actions from every world, `bc/demonstrations.npz` |
| `behaviour-cloning` | `i4h_rl.liver_experiment train-bc` | MSE fit of the 57→128→128→4 actor, `bc/bc.pt`, plus `bc/bc_training.json` |
| `import` | `i4h_rl.liver_experiment import-bc` | RSL-RL checkpoint `bc/imported/native_bc.pt`, TorchScript `policy.pt`, exact parity report |
| `evaluate-bc` | `i4h_rl.liver_experiment evaluate --episodes 40` | `evaluation/bc.json` |
| `ppo` | `./train.sh rl liver_reach_retract --checkpoint bc/imported/native_bc.pt --num-envs 8 --epochs 30 --set agent.algorithm.class_name=i4h_rl.liver_ppo:MinimalLiverPPO --set agent.algorithm.schedule=fixed` | Native checkpoints and `ppo/run.json` |
| `select-ppo` | Evaluate updates 6, 10, 15 and 30 on development starts; choose earliest maximum full-cycle success | `selection/model_selected.pt`, `selection/selection.json` |
| `export` | `./train.sh rl export liver_reach_retract --checkpoint selection/model_selected.pt` | `export/policy.pt`, `export/contract.json` |
| `evaluate-ppo` | same as `evaluate-bc`, using the export | `evaluation/ppo.json` |
| `run` | `./run.sh liver_reach_retract --policy --checkpoint export --episodes 3 --record` | `run/demos.hdf5` through the normal workflow launcher |

`summary.json` records each stage's command, exit code, duration, output hashes and the BC/PPO scores. Each stage's log is in `logs/<stage>.log`.

`MinimalLiverPPO` is RSL-RL PPO with the liver safeguards: 5 critic-only warm-up iterations for the fresh critic, a fixed exploration std of 0.02, a continuing frozen BC reference (weight 0.1), and a rollback of any update whose analytic KL exceeds 0.02. Unlike `GuardedPPO`, it needs no qualification manifest, no BC hash and no frozen 20-start evaluator. The driver uses a fixed learning rate of `1e-5`. It is example tooling, not a qualified training claim. A native training resume requires the original saved reference and matching coefficient; older PPO checkpoints without that reference cannot resume this recipe. Inference/export of old checkpoints remains supported. The plain `./train.sh rl liver_reach_retract` profile still selects the qualified `GuardedPPO` campaign.

BC and the selected PPO checkpoint are scored on the same 40 held-out starts, cycling through all 5 targets with ±5 mm start jitter. Development and test starts are saved before training and use distinct seeds; test scores never select the checkpoint. Demonstrations use a different start seed. For a different training length (at least 6 updates), the candidates are updates 6, 10, 15 and the final update, restricted to the requested length. Longer training is not assumed to produce a better checkpoint.

## Results

A five-seed investigation at physics pin `5af32b1` compared the same fixed BC actor with and without the continuing BC reference. Both methods used 30 updates, fixed LR `1e-5`, fixed std `.02`, and separate development checkpoint selection. Five fresh training seeds each received 100 distinct held-out starts; all policies saw identical starts within each seed block.

| Policy | Full cycles / 500 | Per-seed range / 100 |
| --- | --- | --- |
| Fixed BC actor | 358 | 69–75 |
| New PPO without continuing BC reference | 445 | 78–100 |
| New PPO with frozen BC reference | 497 | 98–100 |
| Previously saved pilot checkpoint | 500 | 100–100 |

The reference-guided method gained 10.4 percentage points over the control overall and won four of five pairs. Its three failures were workspace exits. The pilot is one existing checkpoint evaluated on new starts, not five additional training runs. This investigation used one fixed BC actor; it did not measure variability from collecting new demonstrations or retraining BC. The maintained pipeline regenerates its own BC actor, so report its own result separately.

Evidence: local `runs/liver_reach_retract/20261124_162601/` contains all 2,000 evaluation episodes, source hashes, selections, training histories and verification. The outer launcher returned 143 despite the worker saving all prescribed results and its completion marker; worker exit status was not captured. No reliability trials were retried or discarded. The maintained implementation preserves the tested gradient; rerun end-to-end validation after source changes.

### Maintained pipeline validation

A fresh run on 2026-11-24 at physics pin `5af32b1` (`20261124_maintained_bc_reference`) regenerated snapshots, 160 demonstrations and its own BC actor, then completed all 30 PPO updates. Development scores were 16/20, 16/20, 15/20 and 20/20 at updates 6, 10, 15 and 30. Selection chose update 30 before the held-out PPO evaluation: **BC 34/40, selected PPO 40/40**, followed by **3/3** normal headless launcher episodes and **1/1** visible episode. Native/export action parity was exact on 4,096 inputs; both native HDF5 recordings passed inspection. This is a single end-to-end integration run, separate from the five-seed study.

The saved reference matched the imported BC weights exactly after all 30 updates; std remained `.02` and optimizer LR remained `1e-5`. The initial launch rejected an unsupported Hydra coefficient override before training. Removing that override used the same `MinimalLiverPPO` default weight of `.1`; the run resumed from the same BC inputs with no score-based tuning. Failed-launch metadata is retained alongside the run.

### Recheck after the neutral-host split

At physics pin `8812708`, fresh run `20261124_8812708_compatibility` regenerated both snapshot sizes and completed 160/160 scripted demonstrations. Its new BC actor scored **7/40**. PPO development scores were 4/20, 8/20, 16/20 and 16/20 at updates 6, 10, 15 and 30; the earliest-best rule selected update 15. Held-out PPO scored **34/40** (six workspace exits), and the normal launcher scored **2/3**. A separate visible episode passed **1/1**; native recordings passed inspection and native/export action parity was exact on 4,096 inputs. All stages completed, but one launcher episode failed; this is not an all-successful rollout.

The unchanged saved pilot passed **100/100** on the same seed-101 holdout specifications used previously; the previous BC policy remained **34/40** on the pipeline test specifications. Both matched their prior scores. These fixed-checkpoint checks support runtime compatibility with `8812708`; they do not establish consistent fresh-BC training.

Fitting BC again on the prior stored demonstrations reproduced the prior actor bit for bit. Newly collected demonstrations differed slightly in trajectory length and produced different BC weights. The earlier five-seed campaign held BC fixed, so it did not establish reliability when regenerating demonstrations and retraining BC. No hyperparameters, physical success criteria or failure thresholds were adjusted in this recheck.

### Historical recipe without a continuing reference

Two saved runs on 2026-11-23 completed the pipeline using the earlier private `caff73b` source. They used identical starts for BC and PPO within each run:

| Run directory under `runs/liver_reach_retract/` | BC full cycles | PPO full cycles | Final launcher |
| --- | --- | --- | --- |
| `20261123_183204` | 27/40 | 26/40 | 3/3 |
| `20261123_193948` | 31/40 | 20/40 | 2/3 |

Neither run demonstrates PPO improvement. These are historical results, not validation of the current physics pin; regenerate snapshots and rerun after setup for current evidence.

### Stock Warp validation

Normal setup now uses the public `warp-lang==1.17.0` wheel. At physics pin `8812708`, stock-Warp validation regenerated one- and eight-world snapshots, scored the unchanged saved pilot **100/100** on the same seed-101 holdout specifications, and completed a six-update PPO smoke (five critic-only updates and one retained actor update). This verifies the liver path without the local HashGrid patch; it is not a repeat of the full five-seed or fresh-BC training study. Local evidence is under `runs/liver_reach_retract/20261124_stock_warp/`.

## Watch the policy

```bash
I4H_LIVER_SNAPSHOTS=runs/liver_reach_retract/<ts>/snapshots \
  ./run.sh liver_reach_retract --policy --checkpoint runs/liver_reach_retract/<ts>/export --episodes 3
```

## Limits

- Availability of the pinned Numerical Sim revision depends on publication of the physics sync PR. Do not substitute the older private source or reuse its snapshots.
- Snapshots are tied to their exact world count (1, 8 or 16) and to the pinned sources. Regenerate them after any source change; the driver makes fresh ones in each run directory.
- PPO fine-tuning is short and is not guaranteed to improve on BC. Report the regenerated BC and selected PPO scores side by side. The five-seed evidence covers one liver, one BC actor, five target patches and ±5 mm jitter; it does not establish generalization to new anatomy or real tissue.
- Existing tissue inversions and attachment drift remain recorded mechanics limitations. Policy success does not qualify the mechanics. The full-cycle success criteria, action cap and deadline are unchanged.
- `rl/i4h_rl/liver_ppo.py` is part of the qualified campaign's source hashes. Requalify before resuming that campaign on this source.
