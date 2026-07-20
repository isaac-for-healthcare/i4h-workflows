#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Turnkey env validation for the LOCAL (blind) agent — ONE command, manages the whole flow:
#   arena --dry-run (env-module/discovery import check)
#   -> launch the bridge with ensure-bridge, wait for "scene-edit bridge ready"
#   -> capture the viewport
#   -> ask the local VL model (local-agent/vlcheck.py) for a scene verdict
#   -> stop the bridge, print PASS/FAIL.
#
# The VL step is the BLIND local agent's substitute for eyes. A vision-capable CLI agent
# (Claude/Codex) does NOT need the local VLM — it should capture + read the JPEG with its
# own model. If the local VLM is unreachable this script DEFERS the visual check (exit 3)
# rather than reporting a false scene FAIL.
#
# Usage: ./local-agent/validate-env.sh <env_id> [extra rubric text]
# Exit:  0 = PASS, 1 = build/import/dry-run error, 2 = scene FAIL (geometry or VL pass:false),
#        3 = VL-DEFERRED (build+geometry OK, local VLM unreachable — judge the capture yourself).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${1:?usage: validate-env.sh <env_id> [rubric]}"
EXTRA="${2:-}"
A="$ROOT/workflows/agentic"
RUN="$A/runs/validate_${ENV}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN/captures"
LOG="$RUN/bridge.log"
BRIDGE_URL="$("$A/arena/run.sh" bridge-url --env "$ENV")"
export BRIDGE_URL

RUBRIC="Validate this robot manipulation scene from the top-down viewport. Reply with ONLY a JSON object {\"pass\":true|false,\"issues\":[\"...\"]}. Require: every task object and every destination container is present and resting ON the table surface (NOT missing, NOT sunk below/under the tabletop, NOT floating in the air); destinations are distinct and separated; the robot is upright and faces the table, not toppled or intersecting it. A container buried under the tabletop counts as MISSING. ${EXTRA}"

cleanup() { "$A/arena/stop.sh" --env "$ENV" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "[validate] 1/4 arena --dry-run (env-module import / discovery check)"
if ! "$A/arena/run.sh" --env "$ENV" --dry-run >"$RUN/dryrun.log" 2>&1 || ! grep -qi "dry run ok" "$RUN/dryrun.log"; then
  echo "[validate] FAIL: arena --dry-run did not pass:"
  grep -niE "Error|Traceback|No module|cannot import|not defined|line [0-9]+, in" "$RUN/dryrun.log" | head -20
  exit 1
fi
echo "[validate]     dry-run ok"

echo "[validate] 2/4 launching bridge (this is the real import + scene build)..."
"$A/arena/run.sh" ensure-bridge --env "$ENV" --log "$LOG" --timeout 360
if [ "$?" != 0 ]; then
  echo "[validate] FAIL: bridge never became ready (build/import error):"
  grep -niE "failed with|Error|Traceback|No module|cannot import|not defined|unexpected keyword|line [0-9]+, in" "$LOG" | head -25
  exit 1
fi
echo "[validate]     bridge ready"

echo "[validate] 3/5 geometry check (prop bottoms vs tabletop — catches partial sinking the eye/VL miss)"
geo="$(python3 - <<'PY'
import json, os, urllib.request
BASE = os.environ["BRIDGE_URL"]
def get(url):
    return json.load(urllib.request.urlopen(url, timeout=30))
def names():
    d = get(f"{BASE}/objects")
    out = []
    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("name"), str): out.append(x["name"])
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(d)
    return sorted(set(out))
def bbox(n):
    try:
        return get(f"{BASE}/object?name={n}")["result"].get("bbox")
    except Exception:
        return None
SKIP = ("table", "ground", "terrain", "robot")
is_prop = lambda n: not (any(s in n.lower() for s in SKIP) or "light" in n.lower() or "cam" in n.lower())
tb = bbox("table")
if not tb:
    print("WARN: no table bbox; skipping geometry check"); raise SystemExit(0)
top = tb["max"][2]
issues = []
for n in names():
    if not is_prop(n): continue
    b = bbox(n)
    if not b: continue
    lo = b["min"][2]; gap = lo - top
    tag = "SUNK" if gap < -0.005 else ("floating" if gap > 0.06 else "ok")
    print(f"  {n}: bottom={lo:.3f} tabletop={top:.3f} gap={gap:+.3f} {tag}")
    if tag == "SUNK": issues.append(f"{n} sunk {abs(gap)*1000:.0f}mm")
if issues:
    print("GEOM_FAIL: " + "; ".join(issues))
PY
)"
echo "$geo"
geo_fail=0; echo "$geo" | grep -q "GEOM_FAIL:" && geo_fail=1

echo "[validate] 4/5 capturing viewport"
curl -sS "$BRIDGE_URL/capture" -H 'Content-Type: application/json' \
  -d '{"output_dir":"'"$RUN"'/captures","viewport":true}' >/dev/null 2>&1
vp="$(ls -t "$RUN"/captures/*viewport*.jpg 2>/dev/null | head -1)"
[ -n "$vp" ] || { echo "[validate] FAIL: no viewport capture produced"; exit 1; }

echo "[validate] 5/5 VL scene check on $vp"
verdict="$(python3 "$ROOT/local-agent/vlcheck.py" --image "$vp" --prompt "$RUBRIC" 2>&1)"
echo "[validate] VL verdict: $verdict"
echo "[validate] capture: $vp"
# The local VLM is the BLIND local agent's eyes. If it's unreachable, don't read that as a
# scene FAIL — defer the visual judgment: a vision-capable CLI agent (Claude/Codex) reads
# the capture with its own model; a local (blind) agent must bring up the VLM and re-run.
if echo "$verdict" | grep -qiE 'VLCHECK ERROR|connection refused|failed to|timed out|name or service not known|\[Errno'; then
  echo "[validate] VL UNAVAILABLE (local VLM not reachable) — visual check DEFERRED. Capture: $vp"
  echo "[validate] RESULT: VL-DEFERRED — structural+geometry $([ $geo_fail = 0 ] && echo OK || echo FAILED). Vision-capable agent: read $vp with your own model. Local (blind) agent: start the VLM (see local-agent/config.env) and re-run."
  [ "$geo_fail" = 0 ] && exit 3 || exit 2
fi
vl_pass=0; echo "$verdict" | grep -qiE '"pass"[[:space:]]*:[[:space:]]*true' && vl_pass=1
if [ "$geo_fail" = 0 ] && [ "$vl_pass" = 1 ]; then
  echo "[validate] RESULT: PASS"; exit 0
fi
echo "[validate] RESULT: FAIL — geometry_ok=$([ $geo_fail = 0 ] && echo yes || echo NO) vl_pass=$([ $vl_pass = 1 ] && echo yes || echo NO). Fix source (prop z below tabletop, footprint overlap, missing/floating asset) and re-run."
exit 2
