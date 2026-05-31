# openpi_pi0 — openpi PI0

uv project for the [openpi](https://github.com/Physical-Intelligence/openpi)
PI0 stack (JAX-based). Hosts ultrasound liver-scan PI0 inference.

| Env | Model |
|---|---|
| `ultrasound_liver_scan` | `nvidia/Liver_Scan_Pi0_Cosmos_Rel` |

```bash
workflows/agentic/policy/openpi_pi0/setup.sh                                 # ensure shared third_party + uv sync
workflows/agentic/policy/openpi_pi0/run.sh --env ultrasound_liver_scan
```

Setup mirrors `tools/env_setup/install_pi0.sh`:

* Uses `workflows/agentic/third_party/openpi-581e07d` @ `581e07d73`.
* Patches `src/openpi/training/utils.py` (type-hint loosening).
* Patches `pyproject.toml` to bump `jax[cuda12]` to `0.5.3` for sm_120
  (Blackwell) compat.
* Copies `scripts/train.py` and `scripts/compute_norm_stats.py` into
  `src/openpi/` so the module imports resolve.
* `uv sync` installs `openpi` and `openpi-client` editable.

Policy inference mirrors
`workflows/robotic_ultrasound/scripts/policy/pi0/runners.py`: PI0
`infer({room, wrist, joint_positions}) -> chunk(50,6)`. Actions are
**relative** joint positions — the runtime must integrate them against the
current pose.
