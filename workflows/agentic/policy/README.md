# Agentic Policy

One uv subproject per foundation-model stack. The top-level `run.sh` routes
by env id using `policy.stack` in `config/environments/<env>.yaml`; per-stack
policy backends are built from the same YAML routing fields.

| Subproject | Envs |
|---|---|
| [`gr00t_n15/`](gr00t_n15/) — GR00T N1.5   | `scissor_pick_and_place`, `assemble_trocar` |
| [`gr00t_n16/`](gr00t_n16/) — GR00T N1.6   | `locomanip_tray_pick_and_place`, `locomanip_push_cart` |
| [`openpi_pi0/`](openpi_pi0/) — openpi PI0   | `ultrasound_liver_scan` |

```bash
workflows/agentic/policy/setup.sh                          # set up every subproject (or pass names to pick)
workflows/agentic/policy/run.sh --list-envs                # show env -> subproject mapping + policy language
workflows/agentic/policy/run.sh --env <env_id> [args...]   # dispatch
```

## Examples

Start the matching policy daemon for an environment:

```bash
workflows/agentic/policy/run.sh --env scissor_pick_and_place
workflows/agentic/policy/run.sh --env locomanip_tray_pick_and_place
workflows/agentic/policy/run.sh --env locomanip_push_cart
workflows/agentic/policy/run.sh --env assemble_trocar
workflows/agentic/policy/run.sh --env ultrasound_liver_scan
```

Start only the policy needed for the active environment; validation and e2e runs do not launch every policy daemon.

```bash
workflows/agentic/policy/run.sh --env scissor_pick_and_place --ensure --log /tmp/policy.log
```

Default health ports are configured per env YAML. Policy inference traffic
uses per-env Zenoh topics, not TCP inference ports.

For a new env, add routing to its YAML instead of editing `policy/run.sh` or a
stack `policy/registry.py`:

```yaml
policy:
  stack: gr00t_n16
  infer_module: policy.template_env.infer.infer  # optional; defaults to policy.<env>.infer.infer
  train_module: policy.template_env.train.train  # optional; defaults to policy.<env>.train.train
```
