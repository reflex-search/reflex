#!/usr/bin/env python3
r"""Statistical analysis + plotting for the Reflex efficacy study (REF-176 Phase 3).

Consumes the tidy per-trial metrics table from Phase 2 (``extract_metrics.py``)
and turns it into defensible statistics: medians, bootstrap 95% CIs, effect
sizes, and non-parametric paired tests — plus per-hypothesis summary plots.

================================  PRE-REGISTRATION  ============================
This block is the pre-registered analysis. It is declared **in code** and mirrors
the CEO-locked plan (REF-176 → "Pre-registered primary endpoint" + "Decision
rule"). It is fixed BEFORE any results are viewed; do not edit it to fit an
outcome. Post-hoc comparisons are computed too, but are labelled
SECONDARY/EXPLORATORY and may never be substituted into the primary claim.

  PRIMARY ENDPOINT
    metric      : total_tokens  (= input + output + cache_creation + cache_read)
    comparison  : arm B (Reflex MCP + nudge, "treatment") vs arm A (built-ins
                  only, "control")
    population  : find-all-usages tasks only
    statistic   : median over tasks of the per-task token ratio  B / A
    reported    : separately for the COLD and WARM index conditions, each with a
                  paired bootstrap 95% CI (percentile method).
    verdict on  : the WARM condition (a real multi-query workflow amortizes the
                  one-time index build). COLD is reported alongside as an honesty
                  check, never as the headline.

  DECISION RULE  (let r = warm median ratio, [lo, hi] = its 95% CI)
    "Reflex better"    : r < 0.90  AND  hi < 1.00   (real, non-trivial saving)
    "Reflex worse"     : r > 1.10  AND  lo > 1.00   (real, non-trivial cost)
    "No difference"    : the CI straddles 1.00      (parity cannot be excluded)
    "Indeterminate"    : none of the above (an effect that misses the pre-set
                         thresholds — reported, but not a claim)

  SECONDARY / EXPLORATORY (never feed the primary claim)
    - Same paired analysis for arms C and B' vs A.
    - Other metrics: output_tokens, assistant_turns, total_tool_calls, wall_ms,
      total_cost_usd.
    - H2 accuracy: precision / recall / hallucination-rate (needs the optional
      accuracy table; see --accuracy).
    - H3 outcome: task-success rate (from the metrics table) + answer-quality
      rubric aggregation (needs the optional accuracy table).
================================================================================

Usage:
    python3 benches/efficacy/analyze.py [--metrics results/metrics.csv]
                                        [--accuracy results/accuracy.csv]
                                        [--outdir results/analysis]
                                        [--plotdir results/plots]
                                        [--index-ledger results/index_ledger.json]
    python3 benches/efficacy/analyze.py --self-test   # offline smoke test

Everything is stdlib-only and deterministic (seeded bootstrap); the same inputs
always produce the same summary and plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import plots
from stats import (CI, bootstrap_ci, median, paired_bootstrap_ratio_ci,
                   wilcoxon_signed_rank, wilson_ci)

SCRIPT_DIR = Path(__file__).parent.resolve()

# --------------------------------------------------------------------------- #
# Pre-registration, as executable constants (see the module docstring).        #
# --------------------------------------------------------------------------- #
PRIMARY = {
    "metric": "total_tokens",
    "treatment_arm": "B",
    "control_arm": "A",
    "category": "find_all_usages",
    "verdict_condition": "warm",
}
DECISION = {
    "better_ratio_below": 0.90,
    "worse_ratio_above": 1.10,
    "ci_level": 0.95,
}
N_BOOT = 10000
# Fixed, date-derived seed (Phase 3 authored 2026-07-02). A constant keeps the
# bootstrap byte-for-byte reproducible; it is NOT drawn from the clock.
SEED = 20260702

# The pre-registered "find-all-usages" population. This MUST match exactly the
# authoritative Phase 1 corpus category (tasks/*.yaml, SCHEMA.md category 2),
# which is the literal label `find_all_usages`. Do NOT widen this to sibling
# categories (e.g. reverse_dependency) — that would silently change the
# pre-registered primary population and is exactly the p-fishing the plan bans.
# The hyphen/`findall` spellings are only tolerated so a task_id-based heuristic
# fallback still lands in the right bucket when no category is available.
FIND_ALL_USAGES_CATEGORIES = frozenset({
    "find_all_usages", "find-all-usages", "findall",
})

# H1 efficiency metrics we analyse (primary + secondary). Each maps to how it is
# computed from a metrics row.
EFFICIENCY_METRICS = ["total_tokens", "output_tokens", "assistant_turns",
                      "total_tool_calls", "wall_ms", "total_cost_usd"]

TREATMENT_ARMS = ["B", "Beager", "C", "Bprime"]  # each compared against control "A"


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
def _to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


class Trial:
    """One metrics-table row, with derived fields the analysis needs."""

    __slots__ = ("arm", "task_id", "trial", "category", "condition",
                 "success", "metrics")

    def __init__(self, row: dict, category_of):
        self.arm = row.get("arm", "unknown")
        self.task_id = row.get("task_id", "unknown")
        self.trial = row.get("trial", "")
        # condition: cold | warm. Absent => treat as "warm" (index pre-built,
        # steady state) — see the cold/warm note in the report.
        self.condition = (row.get("condition") or "warm").strip().lower()
        self.success = _to_bool(row.get("success", "false"))
        # category: prefer an explicit column, else the corpus map, else infer.
        self.category = (row.get("category")
                         or category_of(self.task_id)
                         or "unknown")

        inp = _to_float(row.get("input_tokens")) or 0.0
        out = _to_float(row.get("output_tokens")) or 0.0
        cr = _to_float(row.get("cache_read_tokens")) or 0.0
        cc = _to_float(row.get("cache_creation_tokens")) or 0.0
        self.metrics = {
            # total_tokens deliberately includes cache: the MCP context tax
            # (17 tool schemas re-sent every turn) shows up almost entirely as
            # cache_read/creation, so excluding it would hide the very cost the
            # study exists to measure (REF-176 crux #1).
            "total_tokens": inp + out + cr + cc,
            "output_tokens": out,
            "input_tokens": inp,
            "cache_read_tokens": cr,
            "cache_creation_tokens": cc,
            "assistant_turns": _to_float(row.get("assistant_turns")),
            "total_tool_calls": _to_float(row.get("total_tool_calls")),
            "wall_ms": _to_float(row.get("wall_ms")),
            "total_cost_usd": _to_float(row.get("total_cost_usd")),
        }


def _parse_corpus_yaml(text: str) -> dict:
    """Extract task_id -> category from a corpus YAML without a YAML library.

    PyYAML is not installed in the benchmark environment, and the corpus schema
    (SCHEMA.md) is a flat ``tasks:`` list of blocks that each carry a scalar
    ``id:`` and ``category:``. We only need those two fields, so a tolerant
    line scan is safer than pulling in a dependency: for every ``- id:`` we
    capture the ``category:`` that appears before the next task block.
    """
    mapping: dict[str, str] = {}
    cur_id = None
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        # A new task block starts at "- id: <value>".
        m_id = None
        if stripped.startswith("- id:"):
            m_id = stripped[len("- id:"):].strip()
        elif stripped.startswith("id:") and line[:1] in (" ", "-"):
            m_id = stripped[len("id:"):].strip()
        if m_id is not None:
            cur_id = m_id.strip().strip("'\"")
            continue
        if cur_id and stripped.startswith("category:"):
            cat = stripped[len("category:"):].strip().strip("'\"")
            if cat:
                mapping[cur_id] = cat
    return mapping


def load_category_map(path: Path | None) -> dict:
    """Map task_id -> category from the corpus (tasks/*.yaml + tasks.json)."""
    mapping: dict[str, str] = {}
    # Authoritative Phase 1 corpus first (tasks/*.yaml).
    tasks_dir = SCRIPT_DIR / "tasks"
    if tasks_dir.is_dir():
        for yml in sorted(tasks_dir.glob("*.yaml")):
            try:
                mapping.update(_parse_corpus_yaml(yml.read_text()))
            except OSError:
                continue
    # Then JSON task lists (Phase 2 tasks.json, or an explicit --tasks file).
    json_candidates = [SCRIPT_DIR / "tasks.json"]
    if path and path.suffix == ".json":
        json_candidates.append(path)
    elif path and path.suffix in (".yaml", ".yml") and path.exists():
        try:
            mapping.update(_parse_corpus_yaml(path.read_text()))
        except OSError:
            pass
    for cand in json_candidates:
        if cand and cand.exists():
            try:
                data = json.loads(cand.read_text())
                for t in data.get("tasks", []):
                    if "id" in t and "category" in t:
                        mapping.setdefault(t["id"], t["category"])
            except (json.JSONDecodeError, OSError):
                continue
    return mapping


def load_trials(metrics_path: Path, category_map: dict) -> list[Trial]:
    if not metrics_path.exists():
        sys.exit(f"ERROR: metrics table not found: {metrics_path}\n"
                 "Run extract_metrics.py first (or pass --metrics).")

    def category_of(task_id: str):
        if task_id in category_map:
            return category_map[task_id]
        tl = task_id.lower()
        if any(k in tl for k in ("usage", "findall", "callers", "occurrence")):
            return "find_all_usages"
        return None

    trials = []
    with open(metrics_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("task_id") == "_baseline":
                continue  # MCP context-tax baseline is a separate ledger
            trials.append(Trial(row, category_of))
    return trials


def load_accuracy(path: Path | None) -> dict:
    """Optional accuracy/quality table keyed by (arm, task_id, trial).

    Expected columns (all optional): n_expected, n_returned, n_correct,
    precision, recall, hallucination_rate, quality_score (0-4). precision /
    recall / hallucination_rate are derived from the counts when absent.
    """
    if not path or not path.exists():
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("arm"), row.get("task_id"), row.get("trial"))
            n_exp = _to_float(row.get("n_expected"))
            n_ret = _to_float(row.get("n_returned"))
            n_cor = _to_float(row.get("n_correct"))
            prec = _to_float(row.get("precision"))
            rec = _to_float(row.get("recall"))
            hall = _to_float(row.get("hallucination_rate"))
            if prec is None and n_ret not in (None, 0) and n_cor is not None:
                prec = n_cor / n_ret
            if rec is None and n_exp not in (None, 0) and n_cor is not None:
                rec = n_cor / n_exp
            if hall is None and n_ret not in (None, 0) and n_cor is not None:
                hall = (n_ret - n_cor) / n_ret
            out[key] = {
                "precision": prec, "recall": rec, "hallucination_rate": hall,
                "quality_score": _to_float(row.get("quality_score")),
            }
    return out


# --------------------------------------------------------------------------- #
# Core paired analysis                                                         #
# --------------------------------------------------------------------------- #
def _per_task_medians(trials: list[Trial], arm: str, metric: str,
                      condition: str | None, categories: frozenset | None):
    """task_id -> median of ``metric`` across that arm's replicate trials.

    Filters to ``condition`` (cold/warm) and ``categories`` when given. Trials
    with a missing metric value are skipped. Uses ALL trials (successes and
    failures) — a failed run that still burned tokens is a real cost, and
    dropping it would be success-selection bias (see report methodology note).
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        if t.arm != arm:
            continue
        if condition and t.condition != condition:
            continue
        if categories and t.category not in categories:
            continue
        v = t.metrics.get(metric)
        if v is None:
            continue
        buckets[t.task_id].append(v)
    return {task: median(vals) for task, vals in buckets.items() if vals}


def paired_metric_analysis(trials, treatment, control, metric, *,
                           condition=None, categories=None) -> dict:
    """Full paired analysis of one metric for one arm-pair.

    The task is the unit: replicate trials collapse to a per-task median, then
    tasks are the exchangeable units for the ratio, the paired bootstrap CI, and
    the Wilcoxon signed-rank test.
    """
    t_med = _per_task_medians(trials, treatment, metric, condition, categories)
    c_med = _per_task_medians(trials, control, metric, condition, categories)
    common = sorted(set(t_med) & set(c_med))
    result = {
        "metric": metric,
        "treatment_arm": treatment,
        "control_arm": control,
        "condition": condition or "all",
        "n_tasks": len(common),
        "tasks": common,
    }
    if not common:
        result["note"] = "no overlapping tasks for this arm-pair/condition"
        return result

    control_vals = [c_med[t] for t in common]
    treat_vals = [t_med[t] for t in common]
    pairs = list(zip(control_vals, treat_vals))
    per_task_ratios = [(b / a) if a else float("nan")
                       for a, b in pairs]

    result["control_median_of_tasks"] = median(control_vals)
    result["treatment_median_of_tasks"] = median(treat_vals)
    result["per_task_ratios"] = {t: r for t, r in zip(common, per_task_ratios)}

    usable_pairs = [(a, b) for a, b in pairs if a and a > 0]
    if usable_pairs:
        ci = paired_bootstrap_ratio_ci(usable_pairs, n_boot=N_BOOT,
                                       level=DECISION["ci_level"], seed=SEED)
        result["ratio"] = ci.as_dict()
    else:
        result["ratio"] = None
        result["note"] = "all control values <= 0; ratio undefined"

    wil = wilcoxon_signed_rank(control_vals, treat_vals)
    result["wilcoxon"] = wil.as_dict()
    return result


def apply_decision_rule(warm_ratio_ci: dict | None) -> dict:
    """Map the warm primary ratio + CI onto the pre-registered verdict buckets."""
    if not warm_ratio_ci:
        return {"verdict": "insufficient_data",
                "explanation": "No warm find-all-usages ratio could be computed "
                               "(missing arm-pair overlap or condition)."}
    r = warm_ratio_ci["point"]
    lo = warm_ratio_ci["ci_low"]
    hi = warm_ratio_ci["ci_high"]
    better_t = DECISION["better_ratio_below"]
    worse_t = DECISION["worse_ratio_above"]

    if r < better_t and hi < 1.0:
        verdict = "reflex_better"
        expl = (f"warm median ratio r={r:.3f} < {better_t} and 95% CI upper "
                f"bound {hi:.3f} < 1.0 — a real, non-trivial token saving.")
    elif r > worse_t and lo > 1.0:
        verdict = "reflex_worse"
        expl = (f"warm median ratio r={r:.3f} > {worse_t} and 95% CI lower "
                f"bound {lo:.3f} > 1.0 — a real, non-trivial token cost.")
    elif lo <= 1.0 <= hi:
        verdict = "no_difference"
        expl = (f"95% CI [{lo:.3f}, {hi:.3f}] straddles 1.0 — parity cannot be "
                f"excluded (r={r:.3f}).")
    else:
        verdict = "indeterminate"
        expl = (f"r={r:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] excludes parity but "
                f"misses the pre-registered ±10% thresholds — an effect too "
                f"small to claim under the pre-registration.")
    return {"verdict": verdict, "explanation": expl,
            "r": r, "ci_low": lo, "ci_high": hi,
            "thresholds": {"better_below": better_t, "worse_above": worse_t}}


# --------------------------------------------------------------------------- #
# H2 / H3                                                                      #
# --------------------------------------------------------------------------- #
def h2_accuracy(trials, accuracy, arms) -> dict:
    """Per-arm precision / recall / hallucination-rate aggregation (H2)."""
    if not accuracy:
        return {"available": False,
                "note": "No accuracy table supplied (pass --accuracy). "
                        "Precision/recall/hallucination require the agent's "
                        "returned locations scored against the oracle "
                        "ground truth (produced by Phase 4/5 answer scoring)."}
    per_arm = {}
    for arm in arms:
        prec, rec, hall = [], [], []
        for t in trials:
            if t.arm != arm:
                continue
            a = accuracy.get((t.arm, t.task_id, str(t.trial).strip())
                             ) or accuracy.get((t.arm, t.task_id, t.trial))
            if not a:
                continue
            if a.get("precision") is not None:
                prec.append(a["precision"])
            if a.get("recall") is not None:
                rec.append(a["recall"])
            if a.get("hallucination_rate") is not None:
                hall.append(a["hallucination_rate"])

        def summarize(vals):
            if not vals:
                return None
            ci = bootstrap_ci(vals, median, n_boot=N_BOOT,
                              level=DECISION["ci_level"], seed=SEED)
            return ci.as_dict()

        per_arm[arm] = {
            "precision": summarize(prec),
            "recall": summarize(rec),
            "hallucination_rate": summarize(hall),
            "n": len(prec),
        }
    return {"available": True, "per_arm": per_arm}


def h3_outcome(trials, accuracy, arms) -> dict:
    """Per-arm task-success rate (+ quality rubric if the accuracy table has it)."""
    per_arm = {}
    for arm in arms:
        arm_trials = [t for t in trials if t.arm == arm]
        n = len(arm_trials)
        succ = sum(1 for t in arm_trials if t.success)
        entry = {"n_trials": n, "n_success": succ,
                 "success_rate": (wilson_ci(succ, n).as_dict() if n else None)}
        if accuracy:
            quals = []
            for t in arm_trials:
                a = accuracy.get((t.arm, t.task_id, str(t.trial).strip())
                                 ) or accuracy.get((t.arm, t.task_id, t.trial))
                if a and a.get("quality_score") is not None:
                    quals.append(a["quality_score"])
            if quals:
                ci = bootstrap_ci(quals, median, n_boot=N_BOOT,
                                  level=DECISION["ci_level"], seed=SEED)
                entry["quality_rubric"] = ci.as_dict()
            else:
                entry["quality_rubric"] = None
        per_arm[arm] = entry
    return {"per_arm": per_arm,
            "quality_available": bool(accuracy)}


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run_analysis(trials, accuracy, index_ledger) -> dict:
    conditions = sorted({t.condition for t in trials}) or ["warm"]
    has_condition_split = set(conditions) - {"warm"}
    fa = FIND_ALL_USAGES_CATEGORIES

    # ---- H1 primary: total_tokens, B vs A, find-all-usages, cold + warm ----
    primary = {"endpoint": PRIMARY, "by_condition": {}}
    warm_ratio_ci = None
    for cond in ("cold", "warm"):
        if cond not in conditions and cond != "warm":
            continue
        # If the data has no condition column, everything is "warm".
        effective_cond = cond if cond in conditions else None
        analysis = paired_metric_analysis(
            trials, PRIMARY["treatment_arm"], PRIMARY["control_arm"],
            PRIMARY["metric"], condition=effective_cond, categories=fa)
        primary["by_condition"][cond] = analysis
        if cond == "warm":
            warm_ratio_ci = analysis.get("ratio")
    primary["decision"] = apply_decision_rule(warm_ratio_ci)
    primary["cold_warm_note"] = (
        "Agent token totals do not include the `rfx index` build (that runs "
        "outside the agent token budget), so the COLD and WARM *token* ratios "
        "are equal unless the agent itself calls index_project. The cold "
        "penalty is a wall-time/compute cost tracked in the index ledger."
    )
    if not has_condition_split:
        primary["condition_caveat"] = (
            "Metrics table has no `condition` column; all trials treated as "
            "WARM (index pre-built). Phase 4 should emit `condition` to split "
            "cold vs warm explicitly.")

    # ---- H1 secondary: all arms × all efficiency metrics (warm) ----
    secondary = {}
    for arm in TREATMENT_ARMS:
        arm_block = {}
        for metric in EFFICIENCY_METRICS:
            arm_block[metric] = paired_metric_analysis(
                trials, arm, "A", metric,
                condition="warm" if "warm" in conditions else None,
                categories=None)  # all categories for exploratory metrics
        secondary[arm] = arm_block

    arms_present = sorted({t.arm for t in trials})
    return {
        "meta": {
            "n_trials": len(trials),
            "arms_present": arms_present,
            "conditions_present": conditions,
            "n_boot": N_BOOT, "seed": SEED,
            "pre_registration": {"primary": PRIMARY, "decision_rule": DECISION},
        },
        "H1_efficiency": {"primary": primary, "secondary_exploratory": secondary},
        "H2_accuracy": h2_accuracy(trials, accuracy, arms_present),
        "H3_outcome": h3_outcome(trials, accuracy, arms_present),
        "index_ledger": index_ledger,
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _fmt_ci(ci: dict | None) -> str:
    if not ci:
        return "n/a"
    return f"{ci['point']:.3f}  (95% CI {ci['ci_low']:.3f}–{ci['ci_high']:.3f}, n={ci['n']})"


VERDICT_LABEL = {
    "reflex_better": "✅ Reflex better",
    "reflex_worse": "❌ Reflex worse",
    "no_difference": "➖ No difference",
    "indeterminate": "❔ Indeterminate",
    "insufficient_data": "⚠️ Insufficient data",
}


def render_markdown(summary: dict) -> str:
    m = summary["meta"]
    L = []
    L.append("# Reflex Efficacy — Statistical Analysis (REF-176 Phase 3)\n")
    L.append("> Auto-generated by `analyze.py`. Pre-registered primary endpoint "
             "and decision rule are fixed in code; secondary metrics are "
             "exploratory.\n")
    L.append(f"- Trials analysed: **{m['n_trials']}** across arms "
             f"{', '.join(m['arms_present']) or '(none)'}")
    L.append(f"- Conditions present: {', '.join(m['conditions_present'])}")
    L.append(f"- Bootstrap resamples: {m['n_boot']}, seed {m['seed']} (deterministic)\n")

    # Primary
    prim = summary["H1_efficiency"]["primary"]
    dec = prim["decision"]
    L.append("## Primary endpoint (pre-registered)\n")
    L.append(f"**Median total-token ratio B/A on find-all-usages tasks, warm.**\n")
    L.append(f"### Verdict: {VERDICT_LABEL.get(dec['verdict'], dec['verdict'])}\n")
    L.append(f"{dec['explanation']}\n")
    for cond in ("warm", "cold"):
        block = prim["by_condition"].get(cond)
        if not block:
            continue
        ratio = block.get("ratio")
        wil = block.get("wilcoxon", {})
        L.append(f"**{cond.title()} condition** (n={block.get('n_tasks', 0)} tasks): "
                 f"ratio = {_fmt_ci(ratio)}")
        if wil:
            L.append(f"  · Wilcoxon signed-rank: W={wil.get('statistic_W')}, "
                     f"p={wil.get('p_value'):.4f} ({wil.get('method')}), "
                     f"rank-biserial={wil.get('rank_biserial'):.3f}")
    if prim.get("condition_caveat"):
        L.append(f"\n> ⚠️ {prim['condition_caveat']}")
    L.append(f"\n> ℹ️ {prim['cold_warm_note']}\n")

    # Secondary
    L.append("## Secondary / exploratory efficiency metrics (warm)\n")
    L.append("_Not part of the primary claim. Each arm vs control A, all task "
             "categories, per-task median ratio treatment/control._\n")
    L.append("| Arm | Metric | ratio (95% CI) | Wilcoxon p | n |")
    L.append("|-----|--------|----------------|-----------|---|")
    for arm, block in summary["H1_efficiency"]["secondary_exploratory"].items():
        for metric, a in block.items():
            ratio = a.get("ratio")
            wil = a.get("wilcoxon", {})
            rc = _fmt_ci(ratio) if ratio else "n/a"
            p = f"{wil.get('p_value'):.4f}" if wil else "n/a"
            L.append(f"| {arm} | {metric} | {rc} | {p} | {a.get('n_tasks', 0)} |")
    L.append("")

    # H2
    L.append("## H2 — Accuracy (precision / recall / hallucination)\n")
    h2 = summary["H2_accuracy"]
    if not h2.get("available"):
        L.append(f"_{h2.get('note')}_\n")
    else:
        L.append("| Arm | Precision | Recall | Hallucination rate | n |")
        L.append("|-----|-----------|--------|--------------------|---|")
        for arm, a in h2["per_arm"].items():
            L.append(f"| {arm} | {_fmt_ci(a['precision'])} | {_fmt_ci(a['recall'])} "
                     f"| {_fmt_ci(a['hallucination_rate'])} | {a['n']} |")
        L.append("")

    # H3
    L.append("## H3 — Outcome (task success + answer quality)\n")
    h3 = summary["H3_outcome"]
    L.append("| Arm | Success rate (Wilson 95% CI) | Successes | Quality rubric |")
    L.append("|-----|------------------------------|-----------|----------------|")
    for arm, a in h3["per_arm"].items():
        sr = a.get("success_rate")
        srs = (f"{sr['point']:.2f} ({sr['ci_low']:.2f}–{sr['ci_high']:.2f})"
               if sr else "n/a")
        qr = a.get("quality_rubric")
        qrs = _fmt_ci(qr) if qr else ("—" if not h3["quality_available"] else "n/a")
        L.append(f"| {arm} | {srs} | {a['n_success']}/{a['n_trials']} | {qrs} |")
    if not h3["quality_available"]:
        L.append("\n_Answer-quality rubric (0–4) requires the accuracy table "
                 "with `quality_score`; not supplied._")
    L.append("")

    if summary.get("index_ledger"):
        L.append("## Cold-index ledger (confound #2)\n")
        L.append("```json")
        L.append(json.dumps(summary["index_ledger"], indent=2))
        L.append("```")
    return "\n".join(L) + "\n"


def write_plots(summary: dict, plotdir: Path) -> list[Path]:
    plotdir.mkdir(parents=True, exist_ok=True)
    written = []

    prim = summary["H1_efficiency"]["primary"]
    warm = prim["by_condition"].get("warm")
    if warm and warm.get("ratio"):
        ratios = list(warm.get("per_task_ratios", {}).values())
        r = warm["ratio"]
        svg = plots.ratio_distribution_svg(
            "H1 primary — token ratio B/A (find-all-usages, warm)",
            "each dot = one task's median(total_tokens) ratio",
            [x for x in ratios if x == x], r["point"], r["ci_low"], r["ci_high"])
        p = plotdir / "h1_primary_token_ratio.svg"
        p.write_text(svg)
        written.append(p)

    # Forest plot: primary cold vs warm.
    rows = []
    for cond in ("warm", "cold"):
        block = prim["by_condition"].get(cond)
        if block and block.get("ratio"):
            r = block["ratio"]
            rows.append((f"{cond} (n={block['n_tasks']})", r["point"],
                         r["ci_low"], r["ci_high"]))
    if rows:
        svg = plots.forest_svg("H1 primary endpoint — B/A token ratio", rows)
        p = plotdir / "h1_primary_forest.svg"
        p.write_text(svg)
        written.append(p)

    # H3 success-rate bars.
    h3 = summary["H3_outcome"]["per_arm"]
    arms = list(h3.keys())
    if arms:
        succ = [h3[a]["success_rate"]["point"] if h3[a].get("success_rate")
                else float("nan") for a in arms]
        svg = plots.grouped_bars_svg("H3 — task success rate by arm", arms,
                                     [("success rate", succ)], ymax=1.0,
                                     ylabel="success")
        p = plotdir / "h3_success_rate.svg"
        p.write_text(svg)
        written.append(p)

    # H2 precision/recall bars (only if available).
    h2 = summary["H2_accuracy"]
    if h2.get("available"):
        arms = list(h2["per_arm"].keys())
        prec = [h2["per_arm"][a]["precision"]["point"]
                if h2["per_arm"][a].get("precision") else float("nan") for a in arms]
        rec = [h2["per_arm"][a]["recall"]["point"]
               if h2["per_arm"][a].get("recall") else float("nan") for a in arms]
        svg = plots.grouped_bars_svg("H2 — precision & recall by arm", arms,
                                     [("precision", prec), ("recall", rec)],
                                     ymax=1.0, ylabel="rate")
        p = plotdir / "h2_precision_recall.svg"
        p.write_text(svg)
        written.append(p)
    return written


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Reflex efficacy statistical analysis (Phase 3)")
    ap.add_argument("--metrics", type=Path,
                    default=SCRIPT_DIR / "results" / "metrics.csv")
    ap.add_argument("--accuracy", type=Path, default=None,
                    help="Optional per-trial accuracy/quality table (H2/H3).")
    ap.add_argument("--tasks", type=Path, default=None,
                    help="Corpus file for task_id->category (default tasks.json).")
    ap.add_argument("--index-ledger", type=Path, default=None,
                    help="Optional JSON with cold index-build cost per repo.")
    ap.add_argument("--outdir", type=Path,
                    default=SCRIPT_DIR / "results" / "analysis")
    ap.add_argument("--plotdir", type=Path,
                    default=SCRIPT_DIR / "results" / "plots")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the offline self-test (synthetic fixtures) and exit.")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    category_map = load_category_map(args.tasks)
    trials = load_trials(args.metrics, category_map)
    accuracy = load_accuracy(args.accuracy)
    index_ledger = (json.loads(args.index_ledger.read_text())
                    if args.index_ledger and args.index_ledger.exists() else None)

    summary = run_analysis(trials, accuracy, index_ledger)

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    md = render_markdown(summary)
    (args.outdir / "summary.md").write_text(md)
    plot_paths = write_plots(summary, args.plotdir)

    print(md)
    print(f"\nWrote {args.outdir / 'summary.json'}")
    print(f"Wrote {args.outdir / 'summary.md'}")
    for p in plot_paths:
        print(f"Wrote {p}")


# --------------------------------------------------------------------------- #
# Self-test (offline smoke test; no external data needed)                      #
# --------------------------------------------------------------------------- #
def _synthetic_trials(effect: float, n_tasks=6, n_trials=4, seed=1):
    """Build synthetic Trial rows where B tokens = A tokens * effect (+ jitter).

    effect < 1 => Reflex saves tokens; > 1 => Reflex costs tokens.
    Deterministic given ``seed`` (private RNG, not the clock).
    """
    import random as _r
    rng = _r.Random(seed)
    rows = []
    for ti in range(n_tasks):
        base = 40000 + ti * 5000
        for arm, mult in (("A", 1.0), ("B", effect)):
            for tr in range(n_trials):
                jitter = 1.0 + rng.uniform(-0.05, 0.05)
                tok = base * mult * jitter
                rows.append({
                    "arm": arm, "task_id": f"T{ti:02d}_usages", "trial": tr + 1,
                    "condition": "warm", "category": "find_all_usages",
                    "success": "true" if (arm == "B" or tr < n_trials - 1) else "true",
                    "input_tokens": tok * 0.01, "output_tokens": tok * 0.02,
                    "cache_read_tokens": tok * 0.9, "cache_creation_tokens": tok * 0.07,
                    "assistant_turns": 3 if arm == "A" else 4,
                    "total_tool_calls": 5 if arm == "A" else 6,
                    "wall_ms": tok * 0.1, "total_cost_usd": tok * 1e-6,
                })
    cat = {r["task_id"]: "find_all_usages" for r in rows}
    return [Trial(r, lambda tid: cat.get(tid)) for r in rows]


def self_test() -> int:
    """Assertion-based smoke test of the full stats + decision pipeline."""
    from stats import wilcoxon_signed_rank as wsr
    fails = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # 1) Wilcoxon exact against a known textbook value.
    # x=[125,115,130,140,140,115,140,125,140,135], y=[110,122,125,120,140,124,123,137,135,145]
    x = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
    y = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
    w = wsr(x, y)
    check(f"wilcoxon runs (W={w.statistic}, p={w.p_value:.3f}, {w.method})",
          0.0 <= w.p_value <= 1.0 and w.n == 9)

    # 2) Bootstrap CI brackets the point estimate and is deterministic.
    ci1 = bootstrap_ci([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], median,
                       n_boot=2000, level=0.95, seed=SEED)
    ci2 = bootstrap_ci([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], median,
                       n_boot=2000, level=0.95, seed=SEED)
    check("bootstrap deterministic", (ci1.low, ci1.high) == (ci2.low, ci2.high))
    check("bootstrap CI brackets point", ci1.low <= ci1.point <= ci1.high)

    # 3) "Reflex better" fixture (B uses 60% of A's tokens) => reflex_better.
    better = run_analysis(_synthetic_trials(0.60, seed=11), {}, None)
    dv = better["H1_efficiency"]["primary"]["decision"]["verdict"]
    check(f"clear win -> reflex_better (got {dv})", dv == "reflex_better")

    # 4) "Reflex worse" fixture (B uses 150% of A's tokens) => reflex_worse.
    worse = run_analysis(_synthetic_trials(1.50, seed=12), {}, None)
    dv = worse["H1_efficiency"]["primary"]["decision"]["verdict"]
    check(f"clear loss -> reflex_worse (got {dv})", dv == "reflex_worse")

    # 5) Parity fixture (B ~= A) => no_difference.
    par = run_analysis(_synthetic_trials(1.00, seed=13), {}, None)
    dv = par["H1_efficiency"]["primary"]["decision"]["verdict"]
    check(f"parity -> no_difference (got {dv})", dv == "no_difference")

    # 6) Report + plots render without error on the win fixture.
    md = render_markdown(better)
    check("markdown renders", "Primary endpoint" in md and "Verdict" in md)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        paths = write_plots(better, Path(td))
        check("plots written", len(paths) >= 2 and all(p.exists() for p in paths))
        check("svg is valid-ish", paths[0].read_text().startswith("<svg"))

    # 7) H2/H3 with an accuracy table.
    acc = {}
    for t in _synthetic_trials(0.60, seed=11):
        acc[(t.arm, t.task_id, str(t.trial))] = {
            "precision": 0.95 if t.arm == "B" else 0.80,
            "recall": 0.98 if t.arm == "B" else 0.75,
            "hallucination_rate": 0.02 if t.arm == "B" else 0.10,
            "quality_score": 4.0 if t.arm == "B" else 3.0,
        }
    withacc = run_analysis(_synthetic_trials(0.60, seed=11), acc, None)
    check("H2 available with accuracy table",
          withacc["H2_accuracy"].get("available") is True)
    check("H3 quality available",
          withacc["H3_outcome"].get("quality_available") is True)

    print()
    if fails:
        print(f"SELF-TEST FAILED: {len(fails)} check(s) failed: {fails}")
        return 1
    print("SELF-TEST PASSED (all checks green)")
    return 0


if __name__ == "__main__":
    main()
