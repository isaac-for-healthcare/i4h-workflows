# Native workflow consolidation results

This is historical evidence, including its original source and release references. Current setup uses private merged Numerical Sim `caff73b12a77abd6157771678ee0a57206771ac4` without the bridge patch; see the [current liver guide](liver_reach_retract.md). Public-install readiness remains deferred.

The source integration checkpoint is `3180effd`. Merge `4ca34336` preserves the complete `mallan/lab-native-workflows` history through `9ba858f8` and ports the liver and static-aorta integrations onto its native Arena architecture. Current public main's missing-component launch fix is also merged and adapted to direct interpreter execution.

## Current checks

- Lightweight workflow/contracts suite after merging the latest main launcher fix: **449 passed, 2 skipped**. The focused new and native launcher selection also passed **90 tests**.
- Isolated RL project: **219 passed, 1 skipped**, including liver PPO/evaluation/resume tests with declared CPU development dependencies.
- Arena plus focused liver RL tests: **1039 passed, 1 skipped**. After the final idle-action adjustment, the native liver and shared policy selection passed **21 tests**.
- Pre-commit hooks passed on the integration delta. Setup patches for the custom Arena host entry point, window closing, WBC configuration and numerical bridge were checked against the pinned checkouts.
- Native controller: one full-cycle episode succeeded, recorded through Isaac Lab recorder terms. The recording has **141 four-channel actions**, observations and complete `initial_state/i4h` families: numerical, probe, task, identity and RNG.
- Fresh-process native replay completed both visibly and headlessly. Headless replay needs the existing setup-managed Isaac Lab keyboard compatibility patch. Completion alone is not a claim of bitwise full-trajectory equality.
- Exported stopped-PPO actor: one full-cycle native episode succeeded at seed 7341. This does not supersede its failed training qualification.
- Static-aorta native demo: one unscored episode completed with the preserved 115-command schedule and explicit timeout. The initial NoTask-based port had no timeout and was stopped; its replacement completed normally.

Runs and raw binary evidence remain outside Git. Historical evidence is preserved under the existing local `outputs` path, backed by the qualification-evidence directory. The earlier source and numerical experiment recordings were not rewritten or added to Git.

## Review and release boundaries

Normal execution uses YAML workflow declarations, Arena policies and Isaac Lab recording/replay; the removed workflow engine is not restored. Online training stays in `rl/`. Numerical construction/reset uses the custom host in both stock Arena tools and training registration. The tablecloth embodiment declares the registry name required by the newer Arena pin.

Numerical Sim internal PR #19 is consolidated at `1e9ffd9`. Workflows intentionally retains `7edecea` plus its bridge compatibility patch until a reviewed public numerical snapshot is available. Before public release, publish that numerical snapshot, repin all dependency references, remove the carried bridge patch, regenerate compatible snapshots/qualification manifests, and qualify the final pair from a fresh clone. Public checkpoint/snapshot distribution is separate from this source update.

The earlier 100-start results remain controller 100/100, BC 94/100 and stopped PPO 59/100. Full-cycle PPO stopped at actor update 5 after validation regressed from 20/20 to 12/20. Neither that training campaign nor the complete 1/8/16 qualification campaign was repeated by this consolidation.

The PR's pre-existing skill-signature and protocol follow-ups remain separate. Legacy HDF5 recordings and frozen reports retain their original format and source provenance.
