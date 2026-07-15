# Local i4h agent

You drive the Isaac for Healthcare `workflows/agentic/` pipeline. The `i4h-*`
skills hold the exact, tested procedure for every task — file layout, env ids,
policy wiring, and the APIs to call. Your own guesses about these are usually
wrong, so always work through the matching skill.

## STEP 0 (mandatory): load the matching skill before touching anything

Your **first action every session** is to call the `skill` tool to load the i4h
skill that matches the request — before any `read`, `write`, `glob`, `grep`, or
`bash`. Pick it from this table:

| Request is about | Load this skill |
|---|---|
| Create / add a new environment or task (fork robot + scene + policy) | `i4h-workflow-create` |
| Edit an existing env's scene / cameras / task in place | `i4h-workflow-scene-edit` |
| Install / sync deps or third-party checkouts | `i4h-workflow-setup` |
| Record teleop demos | `i4h-workflow-dataset-teleop` |
| Replay an episode | `i4h-workflow-dataset-replay` |
| Expand demos (mimic) | `i4h-workflow-dataset-mimic` |
| VLM success labels / episode filtering | `i4h-workflow-dataset-annotate` |
| Convert HDF5 → LeRobot | `i4h-workflow-dataset-convert` |
| Fine-tune a policy | `i4h-workflow-finetune` |
| Roll out / validate a policy | `i4h-workflow-validate` |
| Run the whole pipeline end-to-end | `i4h-workflow-e2e` |
| Open the LeRobot dataset viewer | `i4h-lerobot-viz` |
| Unsure / want the overview and router | `i4h-workflow` |

For cleanup prompts such as `Stop all`, do not load a stage skill. Run
`workflows/agentic/stop.sh all` from the repo root and report what was stopped.
Do not invent `run.sh stop`.

Then **follow the loaded skill exactly**: create only the files it lists, use the
**env id it specifies** (do not rename it), and complete its checklists. Do not
invent extra files, duplicate or copy a policy package, or refactor shared
modules (e.g. `_humanoid_base.py`, package `__init__.py`) unless the skill tells
you to. If a skill links another skill with `[[name]]`, load that one too.

## STEP LAST (mandatory): verify your own output, then repair

You are **not done when the files are written — only when the checks pass.** Before
your final message, loop until clean:

1. **Run the skill's verification.** Re-open the loaded skill (and the reference it
   sent you to) and run its "Verify"/checklist/"known gaps" items against the files
   you actually wrote — including any grep block it gives you. Run it; don't eyeball it.
2. **Don't trust shallow checks.** `--list-envs`/`--dry-run` resolve the YAML and
   import only the env wrapper (which defers its task/asset imports), and `py_compile`
   only checks syntax — so **none of them import your task/asset modules.** A missing
   import, a camelCase cfg kwarg, or a helper used above its definition passes all
   three silently and only crashes on a real build. Use the skill's greps for those,
   and treat the first real build (e.g. the bridge) as the import check.
3. **Visual + build validation: run the turnkey check (one command).** You are not
   vision-capable and the bridge is long-running, so don't hand-orchestrate it — run
   `./local-agent/validate-env.sh <env_id>`. It does `arena --dry-run`, launches and
   waits for the bridge, captures the viewport, asks the VL model for a scene verdict,
   stops the bridge, and prints `RESULT: PASS` or `RESULT: FAIL` (+ the VL `issues`).
   This is the check that catches geometry bugs every static check misses (a tray sunk
   under the table, a toppled robot, a missing asset). Treat `FAIL` as a real failure
   to fix; do **not** skip it or claim it "requires a human / GPU" — the VL model is
   the human's eyes, and the GPU host is available. **It takes several minutes** (cold
   Isaac bridge start) — run it **plainly and let it finish**; do **not** wrap it in a
   short `timeout` (e.g. `timeout 120`) or it gets killed before the bridge is ready.
4. **Repair and re-run.** For every failing check, fix the **source** and run the
   check again. Loop — never stop on a known failure.
5. **Report only what you actually ran, and don't call static-only a pass.** Quote the
   real command and its real output. A green `--dry-run`/`--list-envs`/`py_compile` is
   **not** validation — they don't import your modules. If you didn't run a real build,
   report the work as **NOT yet validated** (static-only) and say why; never present
   static checks as "all validation passed." A fabricated ✓ is worse than an honest
   "couldn't run the build here."

## Operating rules

1. Stay in `workflows/agentic/` unless the user asks otherwise.
2. Keep answers short and direct.
3. Do not use todo tracking tools. Keep the plan in your response text if needed.
4. Use only the `bash` tool for shell commands. Script names such as `run.sh`, `vllm.sh`, `stop.sh`, and `setup.sh` are file paths/commands, never tool names.
5. Use `run.sh` only to start/run/list/dry-run a component. To stop a component, use its `stop.sh` or `workflows/agentic/stop.sh ... --env <env>`; never invent `run.sh stop`. **To stop the scene-edit bridge, the ONLY correct way is to stop the arena: `workflows/agentic/arena/stop.sh --env <env>`.** Do **not** kill the bridge/Isaac process, send Ctrl-C, or `curl` a `/stop`/`/shutdown` endpoint — there is no such endpoint; stopping the arena is what stops the bridge.
6. For Isaac Sim, arena, policy, replay, finetune, and validation commands, redirect full output to log files under the run directory.
7. Do not paste full Isaac logs into chat or tool results. Summarize with targeted `rg` matches, artifact paths, success/failure counts, and at most the final 80 log lines.
8. Avoid `tee` for high-volume workflow logs unless the user explicitly asks to stream them.
9. The bash `timeout` must be an **integer number of milliseconds** (e.g. `600000`) — never a float/decimal like `600000.0` or a fractional value; opencode rejects those with a schema error. For long-running commands (`bridge.sh start`, `validate-env.sh`, a bridge build) just **omit** the timeout and let them finish.
10. When a skill provides a readiness helper such as `policy/run.sh --ensure`, `vllm.sh ensure`, or `vllm.sh wait`, run that helper exactly with the `bash` tool. Do not add your own `sleep`, backoff, polling loop, `curl`, or `docker ps` checks.
11. **`/tmp` is auto-rejected by the harness** — any command that reads or writes a `/tmp` path (including an output redirect like `cmd > /tmp/out.txt` or a `cat /tmp/...`) fails immediately with `external_directory (/tmp/*); auto-rejecting`. Never touch `/tmp`. Put run artifacts, temp files, scratch output, and summaries under `workflows/agentic/runs/<run>/`; set `TMPDIR` there if needed. **This includes every bridge `/script` payload you generate** — write them to `workflows/agentic/runs/<run>/scripts/` (same place logs/captures go), never `/tmp` or an ad-hoc path you invent. If you want to capture a command's output to a file, redirect to `${RUN_DIR}/...`, never `/tmp` — but for `bridge.sh start` don't redirect at all (run it bare).
12. When a skill includes a checklist or "known gaps" section, verify those items against the files before launching Isaac/bridge validation. Fix mismatches in source first.
13. **Never run destructive git** — no `git checkout <path>`, `git restore`, `git reset`, `git stash`, or `git clean` (the harness blocks them). They discard uncommitted work, including the user's skill/source edits. If a change you made is wrong, **undo it with a targeted edit** (remove exactly what you added) — never wield git to revert. Only ever edit files the task scopes you to.

## Edit mode (the scene-edit bridge)

**Default = LIVE-ONLY.** "Edit the scene" / "edit mode" always means edit through the
bridge and **never modify source or restart the bridge** — the user does not need to
say "live mode" / "don't change source"; assume it. Bake to source **only** if the
user explicitly says `bake`/`save`/`persist`/`commit` as a final step; otherwise (incl.
"exit without baking") just stop the bridge and leave source untouched.

**Do ONLY what the user asks — launching edit mode is NOT a cue to start editing.**
"Run/open the env in edit mode" (with no specific edit) means: launch the bridge,
confirm it's ready (e.g. `GET /objects`), then **STOP and report it's ready, awaiting
edit instructions.** Perform a scene edit **only** when the user explicitly requests
that edit in the current prompt. Do **not** invent, preempt, or "helpfully" apply
edits (e.g. moving the robot, adding a cube) — and do **not** treat the README's
"Edit Scene" example list, other docs, the skill's recipes, or prior runs as a
to-do list. Those are reference; the user's current prompt is the only instruction.

When asked to run an env "in edit mode" / open the `--bridge`:

1. **Launch DETACHED:** `./local-agent/bridge.sh start <env_id>` — it backgrounds the bridge, waits until ready, and prints `RUN_DIR=...`. Run it **plainly and EXACTLY — no output redirection** (no `> file`, no `| tee`, no trailing `; cat ...`, and **no short `timeout`**). It already sends the noisy Isaac boot log to the run dir and prints just a few status lines to stdout, so there's nothing to capture.
   Redirecting it — **especially to `/tmp` (e.g. `> /tmp/out.txt`)** — makes the launch **fail**: the harness auto-rejects `/tmp`, so the command errors before the bridge starts. **Never** run `arena/run.sh --env <env> --bridge` directly in your shell either: foreground, it blocks your (tmux) shell forever, floods stdout with Isaac logs, and the bridge dies with the call. `bridge.sh start` (bare) is the only correct launch.
2. **Edit LIVE only**, via the env-specific bridge URL printed by `./local-agent/bridge.sh start <env_id>` (or `BRIDGE_URL="$(workflows/agentic/arena/run.sh bridge-url --env <env_id>)"`). Never modify source files while the bridge runs. For a `/script` edit, write the Python snippet to `<RUN_DIR>/scripts/<name>.py` (never `/tmp`), then `curl -s -X POST -H 'Content-Type: application/json' -d '{"path":"<abs path>"}' "$BRIDGE_URL/script"`. Load the `i4h-workflow-scene-edit` skill for the live-edit API, helper list, and verified snippet recipes (e.g. add-a-prim).
3. **Verify** each edit: `GET /object?name=<key>` (live pose/bbox) and `POST /capture` into `<RUN_DIR>/captures`; judge the capture (use `vlcheck.py` for the visual check).
4. **Bake (only when the user explicitly says bake/save/persist/commit):** write the live changes into the env's **own** source files + its YAML — follow the skill's bake checklist exactly. **Never edit OR create a file under `third_party/`, or touch anything shared by sibling envs** — that includes the shared **G1 embodiment** (`arena/.../embodiments/g1.py` and `third_party/.../embodiments/g1/g1.py`), a shared `modality.json`, and the single-cam `config.py`. `third_party/` is **gitignored**, so edits/new files there are invisible to `git status` and silently corrupt the other envs. Define a new camera **env-locally** (in the env's assets / task `get_scene_cfg`/`modify_env_cfg`, or a per-instance `embodiment.camera_config` in the env's `get_env`).
   For a normal interactive bake, stop the bridge after collecting live bake state and write source from that state; do not fresh-relaunch. Run the **fresh-source validation gate** `./local-agent/validate-bake.sh <env>` only when the user explicitly asks for validation/onboarding/readiness/ready-to-commit work; it fresh-relaunches and checks YAML recording wiring, env build, recorded policy camera keys, camera renders, robot sanity, and that no shared/tracked file outside the env's own files changed. Green `--dry-run`/`py_compile` do **NOT** prove a bake loads; only the gate does. **It must print `RESULT: PASS`. A `FAIL` is real — fix the SOURCE and re-run; never explain it away. Do not validate against the still-running edit session** (it holds your live edits and gives a false pass).
5. **Exit:** `./local-agent/bridge.sh stop <env_id>` (= `workflows/agentic/arena/stop.sh`). "Exit without baking" = just stop — no source writes, no `/bake`.
