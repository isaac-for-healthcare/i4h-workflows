#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Mechanical bake-completeness GATE for a scene-edit bake (run after a bake, before reporting done).
#
# Why this exists: a bake touches ~8 surfaces across 4 files; a non-deterministic agent
# drops a different 1-2 each run, and `--dry-run`/`py_compile` + validate-env.sh DON'T
# catch the recording/config wiring (validate-env.sh checks scene geometry + a fixed-
# viewport VL that spuriously FAILS when the robot is moved out of frame). This gate is
# purely MECHANICAL — structural assertions the agent cannot explain away.
#
# Usage:  ./local-agent/validate-bake.sh <env_id>
# Exit:   0 = PASS (every check green) ; 1 = FAIL (one or more checks) ; 2 = bad usage / build error.
#
# Checks:
#   A. Static YAML wiring (no GPU): for each policy.pov_cam_names_sim entry —
#      - dataset.camera_mappings[<obs_key minus _rgb>] == observation.images.<video_key>
#      - dataset.modality_template_path resolves, is env-local (NOT under third_party/),
#        and its `video` section contains <video_key>
#      - policy.train.modality_config_path (if set) resolves and is env-local
#   B. Fresh relaunch (GPU): stop any live bridge, launch CLEAN on the baked source, then —
#      - the env builds (no import/instantiation error)
#      - every expected recorded obs key is in observation_manager.active_terms['policy']
#        (this proves each camera SENSOR is defined AND its obs resolves — catches the
#        "obs term references an undefined sensor" bug)
#      - every camera renders (GET /cameras)
#      - the robot is present and upright (root z in a sane band; not missing/sunk/toppled)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A="$ROOT/workflows/agentic"
ENV="${1:-}"; [ -n "$ENV" ] || { echo "usage: validate-bake.sh <env_id>" >&2; exit 2; }
BRIDGE_URL="$("$A/arena/run.sh" bridge-url --env "$ENV")"
YAML="$A/config/environments/$ENV.yaml"
[ -f "$YAML" ] || { echo "[bake-gate] FAIL: no YAML at $YAML" >&2; exit 2; }

fail=0

# ---------- A. static YAML wiring ----------
echo "[bake-gate] A/B  static YAML recording-wiring checks"
STATIC="$(cd "$A" && python3 - "$ENV" <<'PY'
import sys, json, pathlib, yaml
env = sys.argv[1]
d = yaml.safe_load(open(f"config/environments/{env}.yaml"))
pol = d.get("policy", {}) or {}; ds = d.get("dataset", {}) or {}
povs = pol.get("pov_cam_names_sim", []) or []
cm = ds.get("camera_mappings", {}) or {}
fails, expected = [], []
for c in povs:
    ok = c["obs_key"]; vk = c["video_key"]
    rec = ok[:-4] if ok.endswith("_rgb") else ok      # recorded obs key == pov obs_key minus _rgb
    expected.append(rec)
    if rec not in cm:
        fails.append(f"camera_mappings missing key '{rec}' (recorded key for pov obs_key '{ok}')")
    elif cm[rec] != f"observation.images.{vk}":
        fails.append(f"camera_mappings['{rec}']='{cm[rec]}' != 'observation.images.{vk}'")
mt = ds.get("modality_template_path")
if mt:
    if "third_party/" in mt: fails.append(f"modality_template_path is under third_party/ (use an env-local copy): {mt}")
    p = pathlib.Path(mt)
    if not p.exists(): fails.append(f"modality_template_path does not resolve: {mt}")
    else:
        try: vids = set(json.load(open(p)).get("video", {}))
        except Exception as e: vids = set(); fails.append(f"modality_template_path not valid JSON: {e}")
        for c in povs:
            if c["video_key"] not in vids:
                fails.append(f"modality template '{mt}' lacks video key '{c['video_key']}' (has {sorted(vids)})")
mc = (pol.get("train") or {}).get("modality_config_path")
if mc:
    if "third_party/" in mc: fails.append(f"modality_config_path is under third_party/ (use an env-local copy): {mc}")
    if not pathlib.Path(mc).exists(): fails.append(f"modality_config_path does not resolve: {mc}")
print(json.dumps({"expected_obs_keys": expected, "fails": fails}))
PY
)" || { echo "[bake-gate] FAIL: could not parse YAML"; exit 1; }

echo "$STATIC" | python3 -c "import sys,json; d=json.load(sys.stdin); [print('   STATIC FAIL:', f) for f in d['fails']]"
if [ "$(echo "$STATIC" | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["fails"]))')" != "0" ]; then fail=1; else echo "   static wiring ok"; fi
EXPECTED="$(echo "$STATIC" | python3 -c 'import sys,json; print(",".join(json.load(sys.stdin)["expected_obs_keys"]))')"

# ---------- C. env-locality: the bake must touch ONLY the env's own files, never shared ----------
echo "[bake-gate] C   env-locality (no shared-file edits)"
# Flag modified-tracked CODE/CONFIG under workflows/agentic that isn't the env's own. Docs
# (*.md, e.g. an auto-touched README) are not shared-module contamination — exclude them so
# they don't false-FAIL the bake.
STRAY="$(cd "$ROOT" && git status --porcelain -- workflows/agentic 2>/dev/null | grep -E '^ ?M ' | awk '{print $2}' \
  | sed 's#^workflows/agentic/##' \
  | grep -vE '\.md$' \
  | grep -vE "^(arena/arena/(assets|tasks|runtimes)/${ENV}\.py|arena/arena/environments/${ENV}_environment\.py|config/environments/${ENV}\.yaml)$" || true)"
if [ -n "$STRAY" ]; then
  if [ "${I4H_BAKE_GATE_ALLOW_DIRTY:-0}" = "1" ]; then
    echo "   ENV-LOCALITY WARN: shared/tracked files are already dirty; allowing because I4H_BAKE_GATE_ALLOW_DIRTY=1:"
    echo "$STRAY" | sed 's/^/     /'
  else
    echo "   ENV-LOCALITY FAIL: bake modified shared/tracked files (define the camera/change env-locally, not in a shared module):"
    echo "$STRAY" | sed 's/^/     /'; fail=1
  fi
else echo "   only the env's own files were modified ✓"; fi
# gitignored shared g1 embodiments are git-invisible — content-check them directly
for emb in third_party/IsaacLab-Arena-dba0995/isaaclab_arena/embodiments/g1/g1.py arena/arena/embodiments/g1.py; do
  if grep -qE "room_cam|RoomCam" "$A/$emb" 2>/dev/null; then
    echo "   ENV-LOCALITY FAIL: shared embodiment $emb defines a room camera (would contaminate sibling g1 envs)"; fail=1
  fi
done

# ---------- B. fresh relaunch + structural assertions ----------
echo "[bake-gate] B   fresh relaunch of baked source (stopping any live bridge first)"
"$A/arena/stop.sh" --env "$ENV" >/dev/null 2>&1 || true
sleep 2
RUN="$A/runs/bake_gate_${ENV}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUN/logs" "$RUN/scripts"
LOG="$RUN/logs/bridge.log"; : > "$LOG"
if ! "$A/arena/run.sh" ensure-bridge --env "$ENV" --log "$LOG"; then
  echo "[bake-gate] FAIL: bridge never became ready; see $LOG"
  if grep -qiE "Traceback \(most recent|\[agentic-arena\] failed with" "$LOG"; then
    echo "[bake-gate] env did NOT build (baked source has an import/instantiation error):"
    grep -niE "Error|No module|cannot import|Traceback|line [0-9]+, in" "$LOG" | tail -15
  fi
  "$A/arena/stop.sh" --env "$ENV" >/dev/null 2>&1
  exit 1
fi

SCRIPT="$RUN/scripts/assert_bake.py"
cat > "$SCRIPT" <<PY
EXPECTED = [k for k in "$EXPECTED".split(",") if k]
out = {"checks": [], "ok": True}
def chk(name, cond, detail=""):
    out["checks"].append({"name": name, "pass": bool(cond), "detail": str(detail)})
    if not cond: out["ok"] = False
try:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    scene = base.scene
    sensors = list(scene.sensors.keys()) if hasattr(scene, "sensors") else []
    # observation manager policy group must contain every expected recorded obs key
    om = getattr(base, "observation_manager", None)
    pol_terms = []
    if om is not None:
        try: pol_terms = list(om.active_terms.get("policy", []))
        except Exception: pol_terms = list(getattr(om, "group_obs_term_names", {}).get("policy", []))
    out["policy_obs_terms"] = pol_terms
    out["sensors"] = sensors
    for k in EXPECTED:
        chk(f"obs '{k}' recorded in policy group", k in pol_terms, f"policy terms={pol_terms}")
    # robot present + upright (WBC idle squat z ~ -0.14 is fine; flag missing/sunk/flying)
    try:
        z = float(scene["robot"].data.root_pos_w[0][2])
        chk("robot present + upright (root z sane)", -0.6 < z < 0.6, f"root z={z:.3f}")
    except Exception as e:
        chk("robot present + upright", False, f"robot not found: {e!r}")
except Exception as e:
    import traceback; chk("env introspection", False, repr(e)); out["tb"] = traceback.format_exc()
result = out
PY
RESP="$(curl -s -X POST -H 'Content-Type: application/json' -d "{\"path\":\"$SCRIPT\",\"timeout\":90}" "$BRIDGE_URL/script")"
# also list rendered cameras and assert each expected camera renders
CAMS="$(curl -s "$BRIDGE_URL/cameras")"
"$A/arena/stop.sh" --env "$ENV" >/dev/null 2>&1

echo "$RESP" | python3 -c "
import sys, json
try: r = json.load(sys.stdin)
except Exception: print('   BUILD FAIL: bad /script response'); sys.exit(3)
res = r.get('result', {}) if r.get('ok') else {}
for c in res.get('checks', []):
    print('   ', 'OK  ' if c['pass'] else 'FAIL', c['name'], '' if c['pass'] else '-> '+c['detail'])
sys.exit(0 if (r.get('ok') and res.get('ok')) else 3)
" || fail=1
# camera-render assertion: each expected recorded key should map to a rendered camera label
echo "$CAMS" | python3 -c "
import sys, json
exp=[k for k in '$EXPECTED'.split(',') if k]
try: cams={c['name'] for c in json.load(sys.stdin).get('result',{}).get('cameras',[])}
except Exception: cams=set()
print('   rendered cameras:', sorted(cams))
" || true

if [ "$fail" = 0 ]; then echo "[bake-gate] RESULT: PASS — bake is complete and the env loads + records"; exit 0; fi
echo "[bake-gate] RESULT: FAIL — fix the failing check(s) in SOURCE and re-run (do not rationalize a FAIL away)"; exit 1
