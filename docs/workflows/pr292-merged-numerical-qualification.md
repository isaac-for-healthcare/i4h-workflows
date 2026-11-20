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

`python3 scripts/run_tests.py --suite ci --coverage` runs the complete CPU command after `bash scripts/setup_rl_cpu.sh`. Scoped simulator setup uses `I4H_SETUP_PROJECTS='common arena' I4H_THIRD_PARTY_TARGET=arena I4H_WARP_WHEEL=<approved wheel> ./setup.sh`. GPU checks run one simulator at a time. Native strict snapshots must be regenerated for this source pair; historical snapshots correctly fail identity validation.

The local evidence directory is `runs/pr292_qualification/20261120_112500/`; its report records exact commands, source revisions, raw logs, recording paths and earlier failures separately from final passing results. The patient validation fixture preserves the existing s0011 HU array and scan affine while writing the native schema-2 bundle in its run directory. Historical sources and reports were preserved.

NV-BASE executed skillspector and passed with 16 understood warnings: 14 optional test-coverage notices and two robot-coordinate GPS false positives. Required tools and eval datasets were present. Replay completion does not establish bitwise full-trajectory equality. This qualification does not claim training/PPO improvements, XR, public-install readiness or validation of unrelated served-policy checkpoints.
