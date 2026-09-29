#!/usr/bin/env bash
# Performance gates for the incremental-index work.
#
#   perf.sh <rfx> <label> [tree] [runs] [update_paths_timing]
#
# On a scratch clone (default /scratch/cache/k8s-incremental; never the original):
#   cold      rm -rf .reflex; rfx index
#   nochange  rfx index with nothing changed
#   edit1     append one line to one file; rfx index      (then reverted, not timed)
#   library   with the 5th argument (the `update_paths_timing` example binary): a
#             1-file edit through Indexer::update_paths until searchable, then an
#             add and a delete (see examples/update_paths_timing.rs)
# Each line: scenario, wall seconds, peak RSS, and the 1-minute load average at start.
# The background symbol pass is allowed to finish before every measurement.
# `RUST_LOG=info` phase lines of one 1-file edit go to <label>.edit1.log.
set -euo pipefail

BIN="$1"; LABEL="$2"; TREE="${3:-/scratch/cache/k8s-incremental}"; RUNS="${4:-3}"; LIB="${5:-}"
TIME=/run/current-system/sw/bin/time
OUT="${PERF_OUT:-/scratch/cache/incremental-perf}"
mkdir -p "$OUT"
REPORT="$OUT/$LABEL.txt"
cd "$TREE"

wait_symbols() {
  local deadline=$((SECONDS + 600))
  local status
  while status="$("$BIN" index status 2>/dev/null || true)"; [[ "$status" == *Running* ]]; do
    ((SECONDS < deadline)) || { echo "symbol pass still running" >&2; return; }
    sleep 0.5
  done
}

measure() { # <scenario> <cmd...>
  local name="$1"; shift
  wait_symbols
  local load; load="$(cut -d' ' -f1 /proc/loadavg)"
  "$TIME" -f "%e %M" -o "$OUT/.time" "$@" >/dev/null 2>&1
  local wall rss; read -r wall rss <"$OUT/.time"
  printf '%-9s wall=%6.2fs  rss=%7d KB  load=%s\n' "$name" "$wall" "$rss" "$load" | tee -a "$REPORT"
}

# A file every tree has: the first tracked Go / Rust file in sorted order.
EDIT_FILE="$(git ls-files '*.go' '*.rs' | LC_ALL=C sort | sed -n 1p)"

{
  echo "# $LABEL  $(date -u +%Y-%m-%dT%H:%M:%SZ)  $("$BIN" --version)  tree=$TREE  nproc=$(nproc)"
  echo "# edit file: $EDIT_FILE"
} | tee -a "$REPORT"

for _ in $(seq "$RUNS"); do
  rm -rf .reflex
  measure cold "$BIN" index --quiet
done
for _ in $(seq "$RUNS"); do
  measure nochange "$BIN" index --quiet
done
for i in $(seq "$RUNS"); do
  echo "// incremental probe $i" >>"$EDIT_FILE"
  measure edit1 "$BIN" index --quiet
  git checkout -- "$EDIT_FILE"
  wait_symbols
  "$BIN" index --quiet >/dev/null 2>&1
done

# Phase lines for one 1-file edit.
wait_symbols
echo "// incremental probe log" >>"$EDIT_FILE"
RUST_LOG=info "$BIN" index --quiet >"$OUT/$LABEL.edit1.log" 2>&1 || true
git checkout -- "$EDIT_FILE"
wait_symbols
"$BIN" index --quiet >/dev/null 2>&1
echo "# phase log: $OUT/$LABEL.edit1.log" | tee -a "$REPORT"

# The library path, in-process.
if [[ -n "$LIB" ]]; then
  wait_symbols
  echo "# library path (update_paths), load at start $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$REPORT"
  RUST_LOG=info "$LIB" . "$EDIT_FILE" "$RUNS" 2>"$OUT/$LABEL.library.log" | tee -a "$REPORT"
  git checkout -- "$EDIT_FILE"
  echo "# library phase log: $OUT/$LABEL.library.log" | tee -a "$REPORT"
fi
