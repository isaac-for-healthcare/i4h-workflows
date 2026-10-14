# tools/tactile

Helpers for fingertip tactile recordings in LeRobot datasets, independent of
any policy stack. Companion of [`tasks/gr00t_n17_tactile`](../../tasks/gr00t_n17_tactile/README.md).

```bash
# Which tactile columns a dataset declares, and whether the GR00T loader will read them.
uv run --project tools/tactile i4h-tactile inspect /data/h2_sharpa_pads_lerobot

# One episode's ten fingertip maps next to their force traces, as a video (follow-up).
uv run --project tools/tactile i4h-tactile viz /data/h2_sharpa_pads_lerobot --episode 3

# The capture-to-LeRobot conversion, which lives with the recording pipeline.
uv run --project tools/tactile i4h-tactile convert /data/raw/pads /data/h2_sharpa_pads_lerobot --repo-id lab/h2-pads
```

`inspect` is complete; `viz` raises until the code follow-up; `convert` prints
the invocation of the public converter rather than reimplementing it.
