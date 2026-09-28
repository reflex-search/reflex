#!/usr/bin/env bash
# REF-225 autonomous orchestrator.
# Removes the agent from the critical path for the ~2.5h arm-A run: waits for the
# run to finish, computes the validity gate (robustly), and — if Phase 1 PASSES
# (>=12 tasks median num_turns >= 4) — auto-launches Phase 2 (arm A+B matrix).
# The Phase 2 launch is a *mechanical* consequence of a gate pass, honoring the
# pre-registration (no human discretion that could introduce p-fishing).
#
# The gate computation + Phase 2 launch do NOT require Paperclip auth, so they
# survive run-JWT expiry. A completion comment is attempted best-effort with
# whatever PAPERCLIP_API_KEY was in the environment at launch (may be expired by
# completion — that's fine; the durable STATUS file is the source of truth).
#
# Usage (launch detached, capturing the current run's env):
#   setsid nohup bash benches/efficacy/ref225-orchestrator.sh \
#       >benches/efficacy/results/ref225-orchestrator.log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS="$SCRIPT_DIR/results"
STATUS_FILE="$RESULTS/ref225-orchestrator-status.txt"
ARM_A_PID_FILE="$RESULTS/ref225-arm-a.pid"

cd "$REPO_ROOT"

log() { echo "[orchestrator $(date -u +%H:%M:%SZ)] $*"; }

log "started; waiting for arm-A run to complete"

# 1. Wait for the arm-A run to finish (by PID if known, else by log marker).
ARM_A_PID=""
[ -f "$ARM_A_PID_FILE" ] && ARM_A_PID=$(cat "$ARM_A_PID_FILE" 2>/dev/null)
# No PID file means nothing to wait for: a hardcoded fallback PID could match an
# unrelated process after a reboot.
[ -z "$ARM_A_PID" ] && log "no $ARM_A_PID_FILE; not waiting"

WAITED=0
MAX_WAIT=18000   # 5h ceiling
while [ -n "$ARM_A_PID" ] && kill -0 "$ARM_A_PID" 2>/dev/null; do
  sleep 60
  WAITED=$((WAITED + 60))
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    log "TIMEOUT waiting for arm-A (pid $ARM_A_PID)"; break
  fi
done
log "arm-A process no longer running (waited ${WAITED}s)"

# Give the filesystem a moment to flush the last trial.
sleep 5

# 2. Compute the validity gate (robust to incomplete trials).
log "computing validity gate"
GATE_OUT=$(python3 benches/efficacy/ref225-validity-gate.py 2>&1)
GATE_CODE=$?
echo "$GATE_OUT"

{
  echo "REF-225 orchestrator status — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gate_exit_code: $GATE_CODE  (0=PASS, 1=FAIL, 2=undecidable)"
  echo ""
  echo "$GATE_OUT"
} > "$STATUS_FILE"

# 3. Branch on the gate result.
if [ "$GATE_CODE" -eq 0 ]; then
  log "PHASE 1 PASSES — launching Phase 2 (arm A+B matrix)"
  echo "" >> "$STATUS_FILE"
  echo "DECISION: Phase 1 PASSED -> launching Phase 2 at $(date -u +%H:%M:%SZ)" >> "$STATUS_FILE"
  setsid nohup bash benches/efficacy/run-ref225-phase2.sh \
      >"$RESULTS/ref225-phase2.log" 2>&1 &
  PHASE2_PID=$!
  echo "$PHASE2_PID" > "$RESULTS/ref225-phase2.pid"
  log "Phase 2 launched, pid $PHASE2_PID"
  echo "phase2_pid: $PHASE2_PID" >> "$STATUS_FILE"
elif [ "$GATE_CODE" -eq 1 ]; then
  log "PHASE 1 FAILS — null result stands; NOT launching Phase 2"
  echo "" >> "$STATUS_FILE"
  echo "DECISION: Phase 1 FAILED -> publish null; Phase 2 NOT launched." >> "$STATUS_FILE"
else
  log "gate undecidable (some tasks missing data) — arm-A may have ended early"
  echo "" >> "$STATUS_FILE"
  echo "DECISION: UNDECIDABLE — investigate arm-A completeness before deciding." >> "$STATUS_FILE"
fi

# 4. Best-effort Paperclip comment (may fail if the run JWT has expired).
if [ -n "${PAPERCLIP_API_URL:-}" ] && [ -n "${PAPERCLIP_API_KEY:-}" ] && [ -n "${PAPERCLIP_TASK_ID:-}" ]; then
  log "attempting completion comment (best-effort)"
  BODY="## Arm-A Complete — Orchestrator Result

Gate exit code: $GATE_CODE (0=PASS, 1=FAIL, 2=undecidable).

\`\`\`
$GATE_OUT
\`\`\`

Status file: \`benches/efficacy/results/ref225-orchestrator-status.txt\`"
  PAYLOAD=$(python3 -c "import json,sys; print(json.dumps({'body': sys.stdin.read()}))" <<< "$BODY")
  HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
      "$PAPERCLIP_API_URL/api/issues/$PAPERCLIP_TASK_ID/comments" \
      -H "Authorization: Bearer $PAPERCLIP_API_KEY" \
      -H "Content-Type: application/json" -d "$PAYLOAD")
  log "comment POST http_status=$HTTP"
  echo "comment_post_http: $HTTP" >> "$STATUS_FILE"
fi

log "orchestrator done"
