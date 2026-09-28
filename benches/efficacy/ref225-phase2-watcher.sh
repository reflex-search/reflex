#!/usr/bin/env bash
# REF-225 Phase 2 completion watcher.
# Waits for the Phase 2 run to finish, runs the FROZEN 13-valid-task analysis
# (ref225-phase2-analysis.py — the authoritative result, not the Phase 2 script's
# inline all-16 block), writes a durable status file, and posts a best-effort
# Paperclip comment so the agent is woken to report the final verdicts.
#
# Usage (launch detached):
#   setsid nohup bash benches/efficacy/ref225-phase2-watcher.sh \
#       >benches/efficacy/results/ref225-phase2-watcher.log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS="$SCRIPT_DIR/results"
STATUS_FILE="$RESULTS/ref225-phase2-analysis-status.txt"
P2_PID_FILE="$RESULTS/ref225-phase2.pid"

cd "$REPO_ROOT"
log() { echo "[p2-watcher $(date -u +%H:%M:%SZ)] $*"; }

P2_PID=""
[ -f "$P2_PID_FILE" ] && P2_PID=$(cat "$P2_PID_FILE" 2>/dev/null)
[ -z "$P2_PID" ] && { echo "no $P2_PID_FILE; nothing to watch" >&2; exit 1; }

log "watching Phase 2 pid $P2_PID"
WAITED=0; MAX_WAIT=28800   # 8h ceiling
while kill -0 "$P2_PID" 2>/dev/null; do
  sleep 60; WAITED=$((WAITED+60))
  [ "$WAITED" -ge "$MAX_WAIT" ] && { log "TIMEOUT"; break; }
done
log "Phase 2 process ended (waited ${WAITED}s)"
sleep 5

# Ensure metrics + recall CSVs exist (Phase 2's own hook builds them; rebuild defensively).
log "extracting metrics + recall"
python3 benches/efficacy/extract_metrics.py \
  --results-dir benches/efficacy/results/ \
  --out benches/efficacy/results/metrics_ref225.csv 2>&1 | tail -3 || true
python3 benches/efficacy/score_recall_ref225.py \
  --tasks benches/efficacy/tasks/iteration-forcing.yaml \
  --results-dir benches/efficacy/results/ \
  --out benches/efficacy/results/recall_ref225.csv 2>&1 | tail -3 || true

log "running frozen 13-valid-task analysis"
ANALYSIS=$(python3 benches/efficacy/ref225-phase2-analysis.py 2>&1)
echo "$ANALYSIS"
{
  echo "REF-225 Phase 2 authoritative analysis — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "$ANALYSIS"
} > "$STATUS_FILE"

# Best-effort completion comment.
if [ -n "${PAPERCLIP_API_URL:-}" ] && [ -n "${PAPERCLIP_API_KEY:-}" ] && [ -n "${PAPERCLIP_TASK_ID:-}" ]; then
  BODY="## Phase 2 Complete — Authoritative Analysis (13 valid tasks)

\`\`\`
$ANALYSIS
\`\`\`

Status file: \`benches/efficacy/results/ref225-phase2-analysis-status.txt\`
Note: analysis restricted to the 13 tasks that passed the arm-A validity gate;
3 tasks excluded pre-unblinding (frozen in commit f0f87b3)."
  PAYLOAD=$(python3 -c "import json,sys; print(json.dumps({'body': sys.stdin.read()}))" <<< "$BODY")
  HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
      "$PAPERCLIP_API_URL/api/issues/$PAPERCLIP_TASK_ID/comments" \
      -H "Authorization: Bearer $PAPERCLIP_API_KEY" \
      -H "Content-Type: application/json" -d "$PAYLOAD")
  log "comment POST http_status=$HTTP"
fi
log "p2-watcher done"
