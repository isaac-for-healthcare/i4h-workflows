# Merged private Numerical Sim qualification

PR #292 follow-up uses private Numerical Sim `caff73b12a77abd6157771678ee0a57206771ac4` without the carried bridge patch. Private repository access is required; public-install readiness remains deferred.

Validation on Linux x86-64 and RTX PRO 6000 Blackwell:

| Check | Result |
| --- | --- |
| Complete CPU/coverage command | 814 passed, 3 skipped |
| Focused Arena regressions | 91 passed |
| Pinned pre-commit hooks | Passed |
| Fresh scoped Arena setup and locked CI uv 0.11.8 sync | Passed |
| Approved aorta/Warp hashes and installed numerical sources | Passed |
| Native bridge reset/restore and acquired-grasp isolation | Passed in 1/8/16 worlds |
| Native liver full-cycle recording | 1/1 success; 146 four-channel actions |
| Native liver replay | One episode completed |
| Static-aorta diagnostics | Unscored; 115 actions and fluoroscopy frames |
| Headless patient fluoroscopy | C-arm sweep PASS; automatic LeRobot conversion passed |
| Legacy headless recording smoke | Passed |
| In-Kit CUDA/Vulkan dependency checks | Both passed |
| NV-BASE and eval schemas | Zero errors; 15 schemas passed |

`python3 scripts/run_tests.py --suite ci --coverage` runs the complete CPU command after `bash scripts/setup_rl_cpu.sh`. Scoped simulator setup uses `I4H_SETUP_PROJECTS='common arena' I4H_THIRD_PARTY_TARGET=arena ./setup.sh`. Normal setup uses index Warp; the historical patched-Warp results remain historical evidence. See [Warp #2426](https://github.com/NVIDIA/warp/issues/2426) for the known captured-HashGrid issue. GPU checks run one simulator at a time. Native strict snapshots must be regenerated for this source pair; historical snapshots correctly fail identity validation.

The local evidence directory is `runs/pr292_qualification/20261120_112500/`; its report records exact commands, source revisions, raw logs, recording paths and earlier failures separately from final passing results. The patient validation fixture preserves the existing s0011 HU array and scan affine while writing the native schema-2 bundle in its run directory. Historical sources and reports were preserved.

NV-BASE executed skillspector and passed with 16 understood warnings: 14 optional test-coverage notices and two robot-coordinate GPS false positives. Required tools and eval datasets were present. Replay completion does not establish bitwise full-trajectory equality. This qualification does not claim training/PPO improvements, XR, public-install readiness or validation of unrelated served-policy checkpoints.

## Review follow-up

The private Numerical Sim source is a temporary pin and a public-release blocker. An approved public snapshot must be supplied, repinned and qualified before public release. The legacy endoluminal navigation and arm workflows remain on their existing solver; the separate static-aorta workflow does not complete their migration.

Arena installs the pinned GR00T N1.6 client directly from GitHub; a fresh locked sync needs network access. The redundant Arena checkout was removed, while the isolated N1.6 policy target keeps its local checkout. Scoped setup and CI-version locked synchronization passed with the local checkout moved aside, without recreating it.

The converter accepts expanded and filtered sibling HDF5 files sharing the original capture's `run.json`. It validates the patient manifest hash and retains explicit overrides. The hash covers only the manifest, not referenced volume or mesh contents. Both real expansion/filter outputs converted automatically without a patient override; filtering used a fixture verdict, not a model judgment.

Numerical scenes intentionally select `I4H_DATA` from their pinned declarations. Batch liver tools forward `--headless` when requested; fixed batch callers supply it. GPU smoke artifacts are retained under `runs/<workflow>/` and require deliberate cleanup when no longer needed.

Follow-up commands, results and local PR/reply drafts are under `runs/pr292_qualification/20261120_120613/`. Earlier qualification results above are historical evidence for the preceding three commits.

Review follow-up validation: **820 CPU tests passed, 3 existing skips**, 157 focused launcher/setup checks, 44 dataset tests, pinned pre-commit hooks and NV-BASE/schema validation passed. No assertions or exclusions were weakened. GPU mechanics were unchanged; the preceding GPU evidence retains its original source revision.
