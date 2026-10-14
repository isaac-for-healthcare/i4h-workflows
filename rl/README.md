# RL CPU regression tests

Run from the repository root:

```bash
bash scripts/setup_rl_cpu.sh
python scripts/run_tests.py --suite rl
```

The setup command provisions RLinf at `a4b6abe205d7942f45cf3e8843c3e72ce818729d` and applies the checked-in patch idempotently. It does not provision simulators or download models. The RL dev extra pins Hydra 1.3.2, ImageIO 2.34.2, PyAV 12.3.0, and Torch 2.9.0, using CPU Torch wheels on Linux x86_64.

The `ci` selection includes this suite. Draft PRs run it in a dedicated job; ready PRs and main run it in the CPU coverage job. CI requires all four Trocar regression modules to execute without skips. Locally, missing optional RLinf source or codec/collective dependencies skip their checks explicitly. The N1.7 profile path is always checked; installed N1.7 runtime availability is a separate optional check.

Across RLinf profiles, optional training and periodic evaluation recordings warn on encoder or writer-close failures. Standalone evaluation requires a decodable video when requested with `--video` or effective `env.eval.video_cfg.save_video=true`; video failures return nonzero with a video diagnostic. `--set` uses Hydra 1.3.2 in both dry-run preflight and launch, including `+` additions and `++` forced overrides.
