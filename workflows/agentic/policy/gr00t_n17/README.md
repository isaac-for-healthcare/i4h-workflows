# gr00t_n17 — GR00T N1.7

uv project for the GR00T N1.7 stack (transformers 4.57.3, diffusers 0.35.1).
Hosts backends whose checkpoints target the N1.7 model layout.

| Env | Model |
|---|---|
| `scissor_pick_and_place` | `nvidia/SO_ARM_Starter_Gr00tN17` |

```bash
workflows/agentic/policy/gr00t_n17/setup.sh                 # clone Isaac-GR00T @ 4b1dca9 + uv sync
workflows/agentic/policy/gr00t_n17/run.sh --env scissor_pick_and_place
```

Add a new N1.7-stack env by setting `policy.stack: gr00t_n17` in its env YAML.

For TRT acceleration on SO-ARM (~40x speedup):

```bash
workflows/agentic/policy/gr00t_n17/scripts/build_trt_engines.sh
```
