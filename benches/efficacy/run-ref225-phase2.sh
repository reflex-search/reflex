#!/usr/bin/env bash
# REF-225: Phase 2 powered A/B — iteration-forcing task set
# Runs BOTH arms A+B on all 16 pre-registered iteration-forcing tasks.
#
# Prerequisites:
#   1. Phase 1 arm-A validity gate has PASSED (run-ref225-arm-a.sh completed,
#      metrics_ref225_arm_a.csv shows >=12 tasks with median turns >= 4).
#   2. rfx binary is built and current (CARGO_TARGET_DIR or ./target/release/rfx).
#
# This script re-runs arm A plus arm B in a single matrix pass.
# Arm-A trials that already exist are SKIPPED by the runner (resume-safe).
# Arm-B trials start fresh since B results don't exist yet after Phase 1.
#
# Primary endpoints (Holm-Bonferroni, pre-registered REF-225 spec):
#   assistant_turns, total_tool_calls, total_tokens
#
# Usage (run detached):
#   nohup bash benches/efficacy/run-ref225-phase2.sh [--dry-run] [--n N] \
#       >benches/efficacy/results/ref225-phase2.log 2>&1 &
#   echo $! > benches/efficacy/results/ref225-phase2.pid
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$SCRIPT_DIR/results/ref225-phase2.log"

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

echo "=== REF-225 Phase 2: Powered A/B — iteration-forcing tasks ===" | tee -a "$LOG_FILE"
echo "  Arms:  A B" | tee -a "$LOG_FILE"
echo "  N:     $N trials per arm × task" | tee -a "$LOG_FILE"
echo "  Model: $MODEL" | tee -a "$LOG_FILE"
echo "  Tasks: 16 iteration-forcing tasks (iteration-forcing.yaml)" | tee -a "$LOG_FILE"
echo "  Log:   $LOG_FILE" | tee -a "$LOG_FILE"
echo "  Primary endpoints: assistant_turns, total_tool_calls, total_tokens (Holm-Bonferroni)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Verify binary hygiene
RFX_BIN="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/rfx"
if [ ! -x "$RFX_BIN" ]; then
  echo "ERROR: rfx binary not found at $RFX_BIN" | tee -a "$LOG_FILE"
  exit 1
fi

BUILD_SHA=$("$RFX_BIN" --version 2>/dev/null | head -1 || echo "unknown")
echo "  rfx binary: $RFX_BIN" | tee -a "$LOG_FILE"
echo "  rfx version: $BUILD_SHA" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Verify Phase 1 gate was passed before unblinding arm B
ARM_A_CSV="$SCRIPT_DIR/results/metrics_ref225_arm_a.csv"
if [ -z "$DRY_RUN" ] && [ ! -f "$ARM_A_CSV" ]; then
  echo "ERROR: Phase 1 arm-A metrics CSV not found: $ARM_A_CSV" | tee -a "$LOG_FILE"
  echo "       Run run-ref225-arm-a.sh first and confirm PHASE 1 PASSES." | tee -a "$LOG_FILE"
  exit 1
fi

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

echo "Running matrix: ${#IF_TASKS[@]} tasks × $N trials × 2 arms" | tee -a "$LOG_FILE"
echo "  (Arm-A trials already completed by Phase 1 will be SKIPPED — runner is resume-safe)" | tee -a "$LOG_FILE"
echo "Started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$REPO_ROOT"
python3 benches/efficacy/runner.py \
  --arms A B \
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
  echo "=== Extracting Phase 2 metrics ===" | tee -a "$LOG_FILE"
  python3 benches/efficacy/extract_metrics.py \
    --results-dir benches/efficacy/results/ \
    --out benches/efficacy/results/metrics_ref225.csv \
    2>&1 | tee -a "$LOG_FILE"

  echo "" | tee -a "$LOG_FILE"
  echo "=== Recall guardrail: file-coverage recall per arm (rubric tasks) ===" | tee -a "$LOG_FILE"
  python3 benches/efficacy/score_recall_ref225.py \
    --tasks benches/efficacy/tasks/iteration-forcing.yaml \
    --results-dir benches/efficacy/results/ \
    --out benches/efficacy/results/recall_ref225.csv \
    2>&1 | tee -a "$LOG_FILE"
  # Guardrail check: arm B median recall must not be materially below arm A
  python3 - <<'RECALLEOF' 2>&1 | tee -a "$LOG_FILE"
import csv, statistics
from pathlib import Path
p = Path("benches/efficacy/results/recall_ref225.csv")
if p.exists():
    rows = list(csv.DictReader(open(p)))
    a = [float(r["recall"]) for r in rows if r["arm"] == "A"]
    b = [float(r["recall"]) for r in rows if r["arm"] == "B"]
    if a and b:
        ma, mb = statistics.median(a), statistics.median(b)
        delta = mb - ma
        # Guardrail: fail if B recall drops more than 0.05 (5pp) below A
        status = "PASS" if delta >= -0.05 else "FAIL — arm B sacrifices coverage"
        print(f"RECALL GUARDRAIL: arm A median={ma:.3f}, arm B median={mb:.3f}, "
              f"delta={delta:+.3f} -> {status}")
RECALLEOF

  echo "" | tee -a "$LOG_FILE"
  echo "=== Phase 2 statistical analysis (Wilcoxon signed-rank + Holm-Bonferroni) ===" | tee -a "$LOG_FILE"
  python3 - <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import csv, sys
from pathlib import Path

# Import the harness stats module (deterministic Wilcoxon + bootstrap CI)
sys.path.insert(0, str(Path("benches/efficacy")))
from stats import wilcoxon_signed_rank, paired_bootstrap_ratio_ci, median

csv_path = Path("benches/efficacy/results/metrics_ref225.csv")
if not csv_path.exists():
    print("ERROR: metrics CSV not found")
    sys.exit(1)

rows = list(csv.DictReader(open(csv_path)))

IF_TASK_IDS = [
    "reflex-comp-query-path", "reflex-comp-trigram-extract", "reflex-comp-deleted-file",
    "reflex-comp-mcp-dispatch", "ripgrep-comp-binary", "ripgrep-comp-type-flag",
    "ripgrep-comp-parallel", "tokio-comp-work-steal", "tokio-comp-task-abort",
    "tokio-comp-io-driver", "reflex-trans-content-store", "reflex-trans-regex-trigrams",
    "reflex-cm-language-dispatch", "ripgrep-cm-printer-chain", "ripgrep-cm-stats-tracking",
    "tokio-cm-spawn-chain",
]

ENDPOINTS = ["assistant_turns", "total_tool_calls", "total_tokens"]
ALPHA = 0.05

def get_trial_pairs(endpoint):
    """Return paired (a_val, b_val) per task×trial for Wilcoxon signed-rank.
    Pairs on task_id + trial_num so same task/trial is compared across arms."""
    a_map = {}
    b_map = {}
    for r in rows:
        if r.get("task_id") not in IF_TASK_IDS:
            continue
        v = r.get(endpoint, "").strip()
        if not v:
            continue
        try:
            fv = float(v)
        except ValueError:
            continue
        key = (r["task_id"], r.get("trial", ""))
        if r.get("arm") == "A":
            a_map[key] = fv
        elif r.get("arm") == "B":
            b_map[key] = fv
    pairs = [(a_map[k], b_map[k]) for k in a_map if k in b_map]
    return pairs

print("PRIMARY ENDPOINTS — paired Wilcoxon signed-rank + 95% bootstrap CI on ratio")
print(f"  K=3 tests, Holm-Bonferroni FWER control at α={ALPHA}")
print(f"  {'Endpoint':<25} {'A med':>8} {'B med':>8} {'ratio':>7}  {'p':>6}  {'95% CI':>16}  adj-sig")
print("  " + "-"*85)

results = []
for ep in ENDPOINTS:
    pairs = get_trial_pairs(ep)
    if not pairs:
        print(f"  {ep:<25}  NO DATA")
        continue
    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    wres = wilcoxon_signed_rank(a_vals, b_vals)
    ci = paired_bootstrap_ratio_ci(pairs, n_boot=2000, level=0.95, seed=42)
    med_a = median(a_vals)
    med_b = median(b_vals)
    ratio = med_b / med_a if med_a > 0 else float('inf')
    results.append(dict(ep=ep, med_a=med_a, med_b=med_b, ratio=ratio,
                        p=wres.p_value, ci=ci, n=wres.n))

# Holm-Bonferroni correction (sort by p, compare to alpha/(K-rank+1))
results_sorted = sorted(results, key=lambda r: r["p"])
K = len(results_sorted)
for rank, res in enumerate(results_sorted):
    threshold = ALPHA / (K - rank)
    res["holm_sig"] = res["p"] < threshold
    res["holm_threshold"] = threshold

# Print in original endpoint order
ep_order = {ep: i for i, ep in enumerate(ENDPOINTS)}
for res in sorted(results, key=lambda r: ep_order.get(r["ep"], 99)):
    ci = res["ci"]
    ci_str = f"[{ci.low:.3f}, {ci.high:.3f}]"
    sig = "* (sig)" if res["holm_sig"] else "ns"
    print(f"  {res['ep']:<25} {res['med_a']:>8.1f} {res['med_b']:>8.1f} {res['ratio']:>7.3f}  {res['p']:>6.4f}  {ci_str:>16}  {sig}")

print()
all_sig = all(r["holm_sig"] for r in results)
all_ns = all(not r["holm_sig"] for r in results)
directions = ["B<A" if r["ratio"] < 1 else ("B>A" if r["ratio"] > 1 else "tie") for r in results]
consistent = len(set(directions)) == 1
print(f"  Co-primary direction consistency: {'YES ('+directions[0]+')' if consistent else 'NO — mixed'}")
if all_sig and consistent and directions[0] == "B<A":
    print("  VERDICT: B WINS — Reflex reduces iterations; all 3 endpoints significant")
elif all_sig and consistent and directions[0] == "B>A":
    print("  VERDICT: B LOSES — Reflex increases iterations; all 3 endpoints significant")
elif all_ns:
    print("  VERDICT: PARITY — no significant difference on any endpoint (Holm-adjusted)")
elif not consistent:
    print("  VERDICT: MIXED — endpoint directions disagree; report individually")
else:
    print("  VERDICT: INDETERMINATE — some endpoints significant, some not; see individual p-values")

print()
print("Full data in: benches/efficacy/results/metrics_ref225.csv")
PYEOF
fi
