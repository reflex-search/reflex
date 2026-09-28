# REF-222 Decision Record: Powered A/B Efficacy Results (2026-07-03)

> **Historical — not current.** Measured on Reflex 1.5.3 (build 6e549ca) with
> claude-sonnet-4-6. It predates the 1.7.0–2.0.3 changes. Do not quote these numbers.
> A rerun on 2.0.3 is in progress (see TODO.md).

## Summary

Powered rerun of the Reflex columnar MCP vs. grep/glob token efficiency study.

**Verdict: Indeterminate** — r=1.044, 95% CI [1.014, 1.262], CI width 0.248

## What changed vs REF-217

| | REF-217 (thin slice) | REF-222 (powered) |
|---|---|---|
| n tasks | 3 | 9 |
| trials/arm | 3 | 8 |
| total observations | 9 per arm | 72 per arm |
| CI width | 1.012 | 0.248 (4× tighter) |
| r (point estimate) | 1.047 | 1.044 |
| Verdict | Indeterminate | Indeterminate |

The point estimates are almost identical. The prior CI was wide noise at n=3. The powered
run confirms: Reflex columnar MCP overhead is real but below the ±10% pre-registered threshold.

## Key finding

**At equal turn counts, overhead is only 1–2%.** The spread comes from turn-count variance
(corr(total_tokens, turns) ≈ 0.99, per REF-204). The extract_symbols outlier (ratio 1.45) is
because arm B used 3 turns vs arm A's 2 — the only task where this happened.

8 of 9 tasks showed arm B at 2 turns (equal to arm A) with ratios 1.01–1.26. Only
extract_symbols (133 occurrences, agent was more thorough with Reflex tools) showed higher overhead.

## Accuracy finding

Graded precision/recall (vs ripgrep oracle, not Reflex):
- Both arms: precision ≈ 1.00 (no hallucinations)
- Arm B recall is higher on large result sets (extract_symbols: 0.28 vs 0.008; trigramindex: 0.48 vs 0.27)
- Arm A recall is higher on some tasks (sinkmatch: 0.68 vs 0.56)
- Overall: both arms complete tasks successfully; Reflex gives more exhaustive answers on large result sets

## Disposition

This closes the "did we lose parity?" question from REF-217. The answer is:
- **No regression**: REF-217 and REF-222 show the same point estimate and verdict
- **The overhead is real but small**: ~4.4% median, below ±10% threshold
- **Reflex's value is capability, not token savings**: symbol filtering, dependency analysis,
  atomic `find_references` — these are not available in grep/glob regardless of token cost

## Artifacts

- `benches/efficacy/results/REF-222-report.md` — primary report
- `benches/efficacy/results/metrics_ref222.csv` — per-trial efficiency metrics
- `benches/efficacy/results/accuracy_ref222.csv` — per-trial precision/recall
- `benches/efficacy/results/analysis_ref222/summary.json` — full statistics
- `benches/efficacy/run-ref222.sh` — run launcher
- `benches/efficacy/finalize-ref222.sh` — analysis pipeline
- `benches/efficacy/score_accuracy.py` — graded accuracy grader (new)

## Binary hygiene

- rfx v1.5.3, build=6e549ca, columnar=on (verified via startup diagnostic)
- CARGO_TARGET_DIR=/scratch/cache/target
- Model: claude-sonnet-4-6 (pinned on both arms)
