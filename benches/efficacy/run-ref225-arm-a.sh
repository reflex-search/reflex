#!/usr/bin/env bash
# REF-225: Phase 1b arm-A validation — iteration-forcing task set
# Runs arm A only (grep/glob, no MCP) on all 16 iteration-forcing tasks
# with N=8 trials per task, to check validity gate: median turns >= 4.
#
# Run detached so agent heartbeats don't consume the subprocess output.
# Usage:
#   bash benches/efficacy/run-ref225-arm-a.sh [--dry-run] [--n N]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$SCRIPT_DIR/results/ref225-arm-a.log"

N=8
MODEL="${MODEL:-claude-sonnet-5}"
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    --n=*) N="${1#--n=}"; shift ;;
    --n) N="${2:-8}"; shift 2 ;;
    --model=*) MODEL="${1#--model=}"; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    *) shift ;;
  esac
done

mkdir -p "$SCRIPT_DIR/results"

echo "=== REF-225 Phase 1b: Arm-A validation run ===" | tee -a "$LOG_FILE"
echo "  Arm:   A (grep/glob only; no MCP)" | tee -a "$LOG_FILE"
echo "  N:     $N trials per task" | tee -a "$LOG_FILE"
echo "  Model: $MODEL" | tee -a "$LOG_FILE"
echo "  Tasks: 16 iteration-forcing tasks (iteration-forcing.yaml)" | tee -a "$LOG_FILE"
echo "  Log:   $LOG_FILE" | tee -a "$LOG_FILE"
echo "  Goal:  Verify median arm-A assistant_turns >= 4 before unblinding arm-B" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# All 16 pre-registered iteration-forcing task IDs (frozen at git commit 3e50d61)
IF_TASKS=(
  # comprehension (Reflex)
  "reflex-comp-query-path"
  "reflex-comp-trigram-extract"
  "reflex-comp-deleted-file"
  "reflex-comp-mcp-dispatch"
  # comprehension (ripgrep)
  "ripgrep-comp-binary"
  "ripgrep-comp-type-flag"
  "ripgrep-comp-parallel"
  # comprehension (tokio)
  "tokio-comp-work-steal"
  "tokio-comp-task-abort"
  "tokio-comp-io-driver"
  # transitive_dep (Reflex)
  "reflex-trans-content-store"
  "reflex-trans-regex-trigrams"
  # cross_module
  "reflex-cm-language-dispatch"
  "ripgrep-cm-printer-chain"
  "ripgrep-cm-stats-tracking"
  "tokio-cm-spawn-chain"
)

echo "Running: arm A × ${#IF_TASKS[@]} tasks × $N trials" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$REPO_ROOT"
python3 benches/efficacy/runner.py \
  --arms A \
  --tasks "${IF_TASKS[@]}" \
  --n "$N" \
  --model "$MODEL" \
  --skip-build \
  --skip-index \
  $DRY_RUN \
  2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Runner complete at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

if [ -z "$DRY_RUN" ]; then
  echo "" | tee -a "$LOG_FILE"
  echo "=== Extracting arm-A metrics ===" | tee -a "$LOG_FILE"
  python3 benches/efficacy/extract_metrics.py \
    --results-dir benches/efficacy/results/ \
    --out benches/efficacy/results/metrics_ref225_arm_a.csv \
    2>&1 | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
  echo "=== Arm-A validity check: median assistant_turns per task ===" | tee -a "$LOG_FILE"
  python3 - <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import csv, statistics, pathlib
csv_path = pathlib.Path("benches/efficacy/results/metrics_ref225_arm_a.csv")
if not csv_path.exists():
    print("ERROR: metrics CSV not found")
    exit(1)
rows = list(csv.DictReader(open(csv_path)))
arm_a = [r for r in rows if r.get("arm") == "A"]
IF_TASK_IDS = [
    "reflex-comp-query-path", "reflex-comp-trigram-extract", "reflex-comp-deleted-file",
    "reflex-comp-mcp-dispatch", "ripgrep-comp-binary", "ripgrep-comp-type-flag",
    "ripgrep-comp-parallel", "tokio-comp-work-steal", "tokio-comp-task-abort",
    "tokio-comp-io-driver", "reflex-trans-content-store", "reflex-trans-regex-trigrams",
    "reflex-cm-language-dispatch", "ripgrep-cm-printer-chain", "ripgrep-cm-stats-tracking",
    "tokio-cm-spawn-chain",
]
passed = []
failed = []
for tid in IF_TASK_IDS:
    task_rows = [r for r in arm_a if r.get("task_id") == tid]
    turns_vals = []
    for r in task_rows:
        try:
            turns_vals.append(int(r.get("assistant_turns", 0)))
        except (ValueError, TypeError):
            pass
    if not turns_vals:
        print(f"  {tid}: NO DATA")
        failed.append(tid)
        continue
    med = statistics.median(turns_vals)
    status = "PASS" if med >= 4 else "FAIL"
    print(f"  {tid}: median_turns={med:.1f} n={len(turns_vals)} -> {status}")
    if med >= 4:
        passed.append(tid)
    else:
        failed.append(tid)
print()
print(f"VALIDITY GATE RESULT: {len(passed)}/{len(IF_TASK_IDS)} tasks pass (median turns >= 4)")
if len(passed) >= 12:
    print("-> PHASE 1 PASSES: proceed to Phase 2 (arm-A + arm-B full matrix)")
else:
    print(f"-> PHASE 1 FAILS ({len(passed)} < 12): publish null result")
    print("   Null: 'We could not construct iteration-forcing tasks that arm-A cannot")
    print("   answer in <=3 turns. Parity result from REF-222 is robust.'")
PYEOF
fi
