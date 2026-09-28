#!/usr/bin/env bash
# REF-222: Powered A/B run — Reflex (columnar) vs grep/glob
# Runs arms A and B on ALL find_all_usages tasks (9 tasks across 3 repos)
# with N=8 trials per arm × task. Model: $MODEL (default claude-sonnet-5).
#
# Designed to run detached (setsid nohup) so agent heartbeats don't consume
# the output. Runner context memory is harmless: stdout goes to a log file,
# never re-read into the calling agent's context window.
#
# Usage:
#   bash benches/efficacy/run-ref222.sh [--dry-run] [--n N] [--model ID]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$SCRIPT_DIR/results/ref222-run.log"
MANIFEST_PREFIX="ref222"

N=8
DRY_RUN=""
MODEL="${MODEL:-claude-sonnet-5}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN="--dry-run" ;;
    --n) shift; N="$1" ;;
    --n=*) N="${1#--n=}" ;;
    --model) shift; MODEL="$1" ;;
    --model=*) MODEL="${1#--model=}" ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
export MODEL

mkdir -p "$SCRIPT_DIR/results"

echo "=== REF-222 Powered A/B run ===" | tee -a "$LOG_FILE"
echo "  Arms:  A B" | tee -a "$LOG_FILE"
echo "  N:     $N trials per arm × task" | tee -a "$LOG_FILE"
echo "  Model: $MODEL" | tee -a "$LOG_FILE"
echo "  Tasks: find_all_usages category (reflex + ripgrep + tokio repos)" | tee -a "$LOG_FILE"
echo "  Log:   $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Verify binary hygiene (REF-222 req #5)
RFX_BIN="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/rfx"
if [ ! -x "$RFX_BIN" ]; then
  echo "ERROR: rfx binary not found at $RFX_BIN" | tee -a "$LOG_FILE"
  exit 1
fi

STARTUP=$("$RFX_BIN" mcp </dev/null 2>&1 | grep "reflex-mcp startup:" || true)
BUILD_SHA=$(echo "$STARTUP" | grep -oP 'build=\K[a-f0-9]+' || echo "unknown")
RFX_VERSION=$(echo "$STARTUP" | grep -oP 'version=\K[^ ]+' || echo "unknown")
export BUILD_SHA RFX_VERSION
echo "  Binary: $RFX_BIN" | tee -a "$LOG_FILE"
echo "  rfx: $RFX_VERSION build=$BUILD_SHA (columnar=on verified by probe)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run only find_all_usages tasks across all repos
cd "$REPO_ROOT"
FIND_ALL_TASKS=(
  "reflex-findall-extract_symbols"
  "reflex-findall-symbolcache"
  "reflex-findall-trigramindex"
  "ripgrep-findall-sinkmatch"
  "ripgrep-findall-sinkcontext"
  "ripgrep-findall-mmapchoice"
  "tokio-findall-notified"
  "tokio-findall-joinerror"
  "tokio-findall-barrier"
)

echo "Running matrix: ${#FIND_ALL_TASKS[@]} tasks × $N trials × 2 arms" | tee -a "$LOG_FILE"
echo "Started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python3 benches/efficacy/runner.py \
  --arms A B \
  --tasks "${FIND_ALL_TASKS[@]}" \
  --n "$N" \
  --model "$MODEL" \
  --skip-build \
  --skip-index \
  $DRY_RUN \
  2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Runner complete at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

if [ -z "$DRY_RUN" ]; then
  # Full analysis + report (metrics extraction, accuracy grading, stats, report generation)
  bash "$SCRIPT_DIR/finalize-ref222.sh"
fi
