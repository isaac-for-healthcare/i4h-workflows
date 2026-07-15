# Skill Validation

This document is the source of truth for validating the agent skills in this repository before onboarding. Run these checks from the repository root.

The canonical skill catalog is `skills/`. `.claude/skills` and `.codex/skills` are symlinks to the same directory, but validation commands should target `skills/` directly.

## When To Run

Run the recommended local validation and eval dataset schema check when changing any file under `skills/`.

Also run them when changing skill routing docs such as `AGENTS.md`, skill references in `workflows/agentic/README.md`, or the Quick Run golden prompts that the eval datasets are based on.

For source-only workflow changes outside `skills/`, use the workflow-specific checks for that change. If the source change affects a skill prompt, example, command, or expected behavior, update the relevant skill/eval and run this validation too.

## Tool Setup

Use an `nv-base` installation on `PATH`, or create a dedicated virtualenv. The commands below are intentionally version-light: they require `nv-base validate`, the `nv_base.evaluation.dataset_validator.DatasetValidator` import, and the `skillspector` CLI for security scanning.

Preferred `uv tool` setup:

```bash
export NV_PYPI_URL=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple

uv tool install --index-url "$NV_PYPI_URL" nv-base
uv tool install --index-url "$NV_PYPI_URL" skillspector
```

Virtualenv setup:

```bash
export NV_BASE_VENV="$HOME/.venvs/nvbase"
export NV_PYPI_URL=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple

python3 -m venv "$NV_BASE_VENV"
"$NV_BASE_VENV/bin/python" -m pip install -U pip
"$NV_BASE_VENV/bin/python" -m pip install -U --extra-index-url "$NV_PYPI_URL" nv-base
"$NV_BASE_VENV/bin/python" -m pip install -U --extra-index-url "$NV_PYPI_URL" skillspector
export PATH="$NV_BASE_VENV/bin:$PATH"
```

If your CI job publishes a required minimum `nv-base` version, use that version or newer for final onboarding validation. Do not record a local version in this file unless the repo also pins that version.

## Verify Toolchain

```bash
command -v nv-base
nv-base --version

command -v skillspector
skillspector --version
```

If `skillspector` is missing, NV-BASE may still run but the security scan can be skipped with warnings. For onboarding readiness, install `skillspector` and rerun instead of accepting a skipped scan.

If using a virtualenv, also check dependency consistency:

```bash
"${NV_BASE_VENV:-$HOME/.venvs/nvbase}/bin/python" -m pip check
```

## Recommended Local Validation

Use this command for the normal pre-push check. It disables LLM-backed scanning, which avoids transmitting skill contents to external services while still running local NV-BASE checks.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

nv-base validate skills \
  -r cli json \
  -o /tmp/i4h-workflows-internal-nvbase \
  --catalog-path "$PWD" \
  --no-dedup \
  --no-llm \
  -c
```

Flag notes:

- `-r cli json` prints a terminal report and writes a machine-readable JSON report.
- `-o /tmp/i4h-workflows-internal-nvbase` keeps generated reports out of the repository.
- `--catalog-path "$PWD"` points NV-BASE at the repository root, which contains the `skills/` catalog.
- `--no-dedup` skips inter-skill deduplication.
- `--no-llm` avoids LLM-backed security and quality analysis.
- `-c` continues after failures so the report includes all findings.

The JSON report is written as:

```text
/tmp/i4h-workflows-internal-nvbase/nv-base-output-YYYYMMDDHHMMSS.json
```

## Tracked-Only Validation

`nv-base validate` scans files on disk. If locally generated artifacts or caches appear under `skills/`, the working-tree validation can report findings that CI will not see.

Before treating a finding as real, confirm the file is tracked:

```bash
git ls-files --error-unmatch <path-from-the-finding>
```

To validate only git-tracked files while still including staged and working-tree edits to tracked files, copy the tracked file list to a temporary directory and validate there:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

tmp="$(mktemp -d)"
git ls-files -z | rsync -a --from0 --files-from=- ./ "$tmp/"
nv-base validate "$tmp/skills" \
  -r cli json \
  -o "$tmp/_report" \
  --catalog-path "$tmp" \
  --no-dedup \
  --no-llm \
  -c
rm -rf "$tmp"
```

This is a local approximation of CI's clean checkout. It excludes untracked files, including untracked eval datasets, so run `git status --short` first and stage intended new files before using this path for readiness.

## Inspect The Report

Find the latest JSON report:

```bash
latest_report="$(ls -t /tmp/i4h-workflows-internal-nvbase/nv-base-output-*.json | head -1)"
echo "$latest_report"
```

Check the top-level result:

```bash
jq '{overall_passed,total_warnings,total_errors,severity_counts}' "$latest_report"
```

Summarize failing validators:

```bash
jq -r '
  .results[]
  | select(.passed == false)
  | [.validator, (.summary.errors // 0), (.summary.warnings // 0)]
  | @tsv
' "$latest_report"
```

Summarize repeated findings:

```bash
jq -r '
  [.results[] | select(.findings) | .findings[]
    | {category, severity, check_name, message}]
  | group_by(.category + "|" + .check_name + "|" + .message)[]
  | {count:length, category:.[0].category, severity:.[0].severity, check:.[0].check_name, message:.[0].message}
  | [.count,.category,.severity,.check,.message]
  | @tsv
' "$latest_report"
```

Look for skipped-tool warnings:

```bash
jq -r '
  .. | scalars
  | select(type == "string" and test("skillspector not installed|Could not read|WARNING|skipped"; "i"))
' "$latest_report"
```

## Eval Dataset Schema Check

Every onboarded skill in this repository should have an eval dataset at `skills/<skill>/evals/evals.json`. These datasets should cover the Quick Run golden prompts in `workflows/agentic/README.md` and any skill-table examples intended to trigger a specific skill.

Repo-specific rule: entries in `skills/<skill>/evals/evals.json` should set `expected_skill` to that same skill name. Use `null` only for direct non-skill prompts such as `Stop all`.

NV-BASE Tier 3 expects each eval dataset to use the NV-ACES schema:

- top-level JSON array
- each entry has `id`
- each entry has `question`
- each entry has `expected_skill`, either a skill name or `null`
- each entry has `ground_truth`
- each entry has `expected_behavior`, a list of strings

Run this offline schema check from the repository root. It resolves the Python interpreter used by the installed `nv-base` console script, so it works when NV-BASE was installed as a `uv tool`.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

NV_BASE_BIN="$(command -v nv-base)"
NV_BASE_PY="$(python3 - "$NV_BASE_BIN" <<'PY'
from pathlib import Path
import sys

script = Path(sys.argv[1])
first_line = script.read_text(encoding="utf-8").splitlines()[0]
if not first_line.startswith("#!"):
    raise SystemExit(f"cannot resolve Python interpreter from {script}")
print(first_line[2:].strip())
PY
)"

"$NV_BASE_PY" - <<'PY'
from pathlib import Path
import json
from nv_base.evaluation.dataset_validator import DatasetValidator

skill_dirs = sorted(path for path in Path("skills").iterdir() if (path / "SKILL.md").is_file())
files = sorted(Path("skills").glob("*/evals/evals.json"))
if not files:
    print("NO_EVAL_DATASETS")
    raise SystemExit(1)

missing = [path for path in skill_dirs if not (path / "evals/evals.json").is_file()]
if missing:
    for path in missing:
        print(f"MISSING_EVAL_DATASET {path}/evals/evals.json")
    raise SystemExit(1)

validator = DatasetValidator()
failed = False
for path in files:
    skill_name = path.parts[1]
    result = validator.validate(path)
    print(("PASS " if result.passed else "FAIL ") + str(path))
    if not result.passed:
        failed = True
        for finding in result.findings:
            print("   -", finding.severity, finding.check_name, finding.message)
    data = json.loads(path.read_text(encoding="utf-8"))
    for index, entry in enumerate(data):
        expected_skill = entry.get("expected_skill")
        if expected_skill not in (skill_name, None):
            failed = True
            print(f"   - ERROR expected_skill_mismatch entry {index}: expected_skill should be {skill_name!r} or null, got {expected_skill!r}")

raise SystemExit(1 if failed else 0)
PY
```

Expected result for this repository: one `PASS` line per `skills/*/evals/evals.json` file, no `FAIL` lines, and no `NO_EVAL_DATASETS`.

If the check prints `NO_EVAL_DATASETS`, the eval files are missing or the command is being run from the wrong directory. Fix that before onboarding.

## Optional Networked Validation

Dropping `--no-llm` enables LLM-backed analysis and may run live Tier 3 agent evaluation when eval datasets are present and the installed NV-BASE version supports it. This can send skill contents to configured internal model/security services and is slower than the offline path. Run it only when the repository owner explicitly approves that data flow.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

nv-base validate skills \
  -r cli json \
  -o /tmp/i4h-workflows-internal-nvbase \
  --catalog-path "$PWD" \
  --no-dedup \
  -c
```

If this hangs during `skillspector`, reports connection errors, or cannot access internal evaluation services, use the recommended `--no-llm` local validation command and report that networked validation was not run.

## Acceptance Criteria

Before reporting skill work as ready:

- `nv-base validate skills ... --no-llm -c` completes and writes a JSON report.
- The latest JSON report has `overall_passed: true` and `total_errors: 0`.
- The eval dataset schema check prints `PASS` for every `skills/*/evals/evals.json` file.
- No skill directory with `SKILL.md` is missing `evals/evals.json`.
- `skillspector` is installed and the report does not indicate that the security scan was skipped.
- Every Quick Run natural-language prompt in `workflows/agentic/README.md` is represented in an eval dataset, or intentionally covered as a direct command with `expected_skill: null`.
- Any remaining warnings are understood, documented in the handoff, and not caused by missing evals, missing metadata, broken links, invalid JSON, script lint failures, or skipped required tooling.

## Common Warning Triage

Known low-risk warnings can be accepted only after inspecting the latest report:

- `gps_coordinates` PII warnings can be false positives when robot coordinate or scale tuples are mistaken for GPS data.
- `External Transmission` warnings can be false positives for documented calls to the localhost scene-edit bridge, such as `http://127.0.0.1:<port>/...`.
- `quality_discoverability` warnings may be acceptable when shortening a skill `description` would make skill triggering less accurate.

Do not accept warnings for missing eval datasets, missing metadata, invalid eval schema, broken relative links, unreadable files, missing `skillspector`, or failed script linting.

## Updating Evals

When a Quick Run prompt in `workflows/agentic/README.md` changes, update the matching `skills/*/evals/evals.json` entry in the same change. The `question` should use the prompt text or a close representative prompt, `expected_skill` should name the skill that should trigger, and `expected_behavior` should list the concrete behaviors the agent must perform.

When a skill is intentionally removed, remove its eval dataset and all references to that skill in `AGENTS.md`, `workflows/agentic/README.md`, and related routing docs.
