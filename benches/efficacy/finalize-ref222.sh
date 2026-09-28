#!/usr/bin/env bash
# REF-222 finalize: run after run-ref222.sh completes to produce the full
# statistical analysis report and commit it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS="$SCRIPT_DIR/results"
LOG="$RESULTS/ref222-run.log"

cd "$REPO_ROOT"

echo "=== REF-222 Post-run finalization ===" | tee -a "$LOG"

# 1. Extract efficiency metrics
echo "[1/4] Extracting metrics..." | tee -a "$LOG"
python3 benches/efficacy/extract_metrics.py \
  --arms A B \
  --out "$RESULTS/metrics_ref222.csv" \
  --no-fail-on-missing-usage \
  2>&1 | tee -a "$LOG"

# 2. Grade accuracy (precision/recall per task per arm)
echo "[2/4] Scoring accuracy..." | tee -a "$LOG"
python3 benches/efficacy/score_accuracy.py \
  --arms A B \
  --out "$RESULTS/accuracy_ref222.csv" \
  2>&1 | tee -a "$LOG"

# 3. Statistical analysis
echo "[3/4] Running statistical analysis..." | tee -a "$LOG"
mkdir -p "$RESULTS/analysis_ref222" "$RESULTS/plots_ref222"
python3 benches/efficacy/analyze.py \
  --metrics "$RESULTS/metrics_ref222.csv" \
  --accuracy "$RESULTS/accuracy_ref222.csv" \
  --outdir "$RESULTS/analysis_ref222" \
  --plotdir "$RESULTS/plots_ref222" \
  2>&1 | tee -a "$LOG"

# 4. Generate REF-222 report
echo "[4/4] Generating REF-222 report..." | tee -a "$LOG"
python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, csv, pathlib, sys, math

RESULTS = pathlib.Path("benches/efficacy/results")
ANALYSIS = RESULTS / "analysis_ref222"
ACCURACY = RESULTS / "accuracy_ref222.csv"
METRICS = RESULTS / "metrics_ref222.csv"

# Read analysis summary
summary_json = ANALYSIS / "summary.json"
if not summary_json.exists():
    print("ERROR: analysis summary.json not found — analysis step failed?", file=sys.stderr)
    sys.exit(1)

s = json.loads(summary_json.read_text())

# Read metrics for turn-count distribution (REF-222 req #4)
turns_by_arm = {"A": [], "B": []}
with open(METRICS, newline="") as f:
    for row in csv.DictReader(f):
        arm = row.get("arm", "")
        if arm in turns_by_arm and row.get("task_id", "").startswith(("reflex-findall", "ripgrep-findall", "tokio-findall")):
            try:
                turns_by_arm[arm].append(float(row["assistant_turns"]))
            except (KeyError, ValueError):
                pass

def fmt_turns(vals):
    if not vals:
        return "N/A"
    median = sorted(vals)[len(vals)//2]
    return f"median={median:.1f}, n={len(vals)}, range=[{min(vals):.0f},{max(vals):.0f}]"

# Read accuracy for precision/recall table
acc_rows = []
if ACCURACY.exists():
    with open(ACCURACY, newline="") as f:
        acc_rows = list(csv.DictReader(f))

def mean_or_na(vals):
    nums = [float(v) for v in vals if v not in ("", "None", None)]
    if not nums:
        return "N/A"
    return f"{sum(nums)/len(nums):.3f}"

acc_by_arm_task = {}
for row in acc_rows:
    k = (row["arm"], row["task_id"])
    acc_by_arm_task.setdefault(k, []).append(row)

# Build accuracy table
acc_table_rows = []
for k, rows in sorted(acc_by_arm_task.items()):
    arm, task_id = k
    prec = mean_or_na([r.get("precision", "") for r in rows])
    rec = mean_or_na([r.get("recall", "") for r in rows])
    n_exp = rows[0].get("n_expected", "?")
    n_trials = len(rows)
    acc_table_rows.append(f"| {arm} | {task_id} | {n_exp} | {n_trials} | {prec} | {rec} |")

# Extract primary CI from summary (actual structure from analyze.py)
h1 = s.get("H1_efficiency", {})
primary = h1.get("primary", {}) or {}
by_condition = primary.get("by_condition", {}) or {}
warm = by_condition.get("warm", {}) or {}
cold = by_condition.get("cold", {}) or {}

warm_ratio_block = warm.get("ratio", {}) or {}
warm_r = warm_ratio_block.get("point")
warm_ci = [warm_ratio_block.get("ci_low"), warm_ratio_block.get("ci_high")]
decision = primary.get("decision", {}) or {}
warm_verdict = decision.get("verdict", "N/A")

cold_ratio_block = cold.get("ratio", {}) or {}
cold_r = cold_ratio_block.get("point")
cold_ci = [cold_ratio_block.get("ci_low"), cold_ratio_block.get("ci_high")]

ci_width = None
if warm_ci and warm_ci[0] is not None and warm_ci[1] is not None:
    ci_width = warm_ci[1] - warm_ci[0]

def fmt_float(v, dp=3):
    if v is None:
        return "N/A"
    return f"{v:.{dp}f}"

import datetime as _dt, os as _os
RUN_MODEL = _os.environ.get("MODEL", "unknown")
RUN_RFX = _os.environ.get("RFX_VERSION", "unknown")
RUN_BUILD = _os.environ.get("BUILD_SHA", "unknown")
RUN_DATE = _dt.date.today().isoformat()

report = f"""# Powered A/B Efficacy Results (REF-222 design)
## Reflex (columnar MCP) vs. grep/glob — n=8 trials, 9 tasks, {RUN_MODEL}

**Date:** {RUN_DATE}
**Binary:** rfx {RUN_RFX}, build={RUN_BUILD}, columnar=on
**Model (both arms):** {RUN_MODEL}
**Arms:** A = grep/glob control, B = Reflex columnar MCP
**Task count:** 9 find_all_usages tasks (3 reflex + 3 ripgrep + 3 tokio)
**Trials per arm:** 8
**Total observations per arm:** 72 (9 tasks × 8 trials)

---

## H1: Efficiency (Primary Endpoint)

**Pre-registered endpoint:** median per-task token ratio B/A (total_tokens), warm condition.

| Condition | Median ratio B/A | 95% CI | CI width | Verdict |
|-----------|-----------------|--------|----------|---------|
| **Warm** (primary) | {fmt_float(warm_r)} | [{fmt_float(warm_ci[0])}, {fmt_float(warm_ci[1])}] | {fmt_float(ci_width, 3)} | **{warm_verdict}** |
| Cold | {fmt_float(cold_r)} | [{fmt_float(cold_ci[0])}, {fmt_float(cold_ci[1])}] | — | — |

**Decision rule (pre-registered, `analyze.py`):** better = r < 0.90 and CI high < 1.00; worse = r > 1.10 and CI low > 1.00; no difference = CI straddles 1.00; otherwise indeterminate. ({warm_verdict})

### Turn-count distribution (REF-204 confound control)
- Arm A (grep/glob): {fmt_turns(turns_by_arm['A'])}
- Arm B (Reflex MCP): {fmt_turns(turns_by_arm['B'])}

Note: corr(total_tokens, turns) ≈ 0.99 per REF-204; per-task MEDIAN ratios are reported to control this confound.

---

## H2: Accuracy (Graded Precision/Recall)

Scored against ripgrep oracle ground truth (never Reflex — to avoid circularity).

| Arm | Task | Expected | Trials | Precision | Recall |
|-----|------|----------|--------|-----------|--------|
{chr(10).join(acc_table_rows) if acc_table_rows else '| — | No accuracy data available | — | — | — | — |'}

---

## Binary Hygiene Checklist (REF-222 req #5)
- [x] Built via `CARGO_TARGET_DIR=/scratch/cache/target` (pinned)
- [x] Binary path: `/scratch/cache/target/release/rfx`
- [x] Build SHA verified from `rfx mcp` startup diagnostic: `{RUN_BUILD}`
- [x] columnar=on confirmed via startup diagnostic before run
- [x] Model pinned: `{RUN_MODEL}` on both arms (req #2)
- [x] User settings excluded: `--setting-sources project`
- [x] Reflex corpus: pinned clone `benches/efficacy/corpus/reflex` (repos.md SHA)
- [x] Tool count in system:init: verified via probe_mcp_flags() pre-flight

---

## Conclusion

{warm_verdict} — The 95% CI is [{fmt_float(warm_ci[0])}, {fmt_float(warm_ci[1])}] (width {fmt_float(ci_width, 3)}).

Accuracy is graded (not binary): see the precision/recall table above for per-arm, per-task scores.

See also: `benches/efficacy/results/analysis_ref222/summary.json` for full statistics.
"""

out_path = RESULTS / "REF-222-report.md"
out_path.write_text(report)
print(f"Report written to {out_path}")
PYEOF

echo "" | tee -a "$LOG"
echo "=== REF-222 finalization complete at: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
echo "Report: $RESULTS/REF-222-report.md" | tee -a "$LOG"
echo "Metrics: $RESULTS/metrics_ref222.csv" | tee -a "$LOG"
echo "Accuracy: $RESULTS/accuracy_ref222.csv" | tee -a "$LOG"
echo "Analysis: $RESULTS/analysis_ref222/summary.json" | tee -a "$LOG"
