#!/usr/bin/env bash
# Timing gates for auto-update (every command updates a stale index first).
#
#   auto_update.sh <rfx> <label> [tree] [runs]
#
# On a scratch clone (default /scratch/cache/k8s-incremental; never the original),
# with an index already built by <rfx>:
#   query_fresh   rfx query <token> --json          nothing changed
#   query_edit    append one line to one file; rfx query <that line's token> --json
#                 (the query updates the index, then answers); reverted after
#   deps_fresh    rfx deps <file> --json            nothing changed
#   hotspots      rfx analyze --hotspots --json     nothing changed
#   mcp_edit      an `rfx mcp` session: one warm search, then per run an edit and a
#                 search_code for its token (request latency, the session's peak RSS)
# Each line: scenario, wall seconds, peak RSS, and the 1-minute load average at start.
set -euo pipefail

BIN="$1"; LABEL="$2"; TREE="${3:-/scratch/cache/k8s-incremental}"; RUNS="${4:-5}"
TIME=/run/current-system/sw/bin/time
OUT="${PERF_OUT:-/scratch/cache/incremental-perf}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
REPORT="$OUT/$LABEL.auto_update.txt"
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
  "$TIME" -f "%e %M" -o "$OUT/.time" "$@" >"$OUT/.stdout" 2>/dev/null || true
  local wall rss; read -r wall rss <"$OUT/.time"
  printf '%-11s wall=%6.3fs  rss=%7d KB  load=%s\n' "$name" "$wall" "$rss" "$load" | tee -a "$REPORT"
}

EDIT_FILE="$(git ls-files '*.go' '*.rs' | LC_ALL=C sort | sed -n 1p)"
{
  echo "# $LABEL  $(date -u +%Y-%m-%dT%H:%M:%SZ)  $("$BIN" --version)  tree=$TREE  nproc=$(nproc)"
  echo "# edit file: $EDIT_FILE"
} | tee -a "$REPORT"

git checkout -- "$EDIT_FILE"
"$BIN" index --quiet >/dev/null 2>&1
for _ in $(seq "$RUNS"); do
  measure query_fresh "$BIN" query NewController --json --limit 20
done
for i in $(seq "$RUNS"); do
  echo "// zz_auto_update_probe_$i" >>"$EDIT_FILE"
  measure query_edit "$BIN" query "zz_auto_update_probe_$i" --json
  grep -q '"can_trust_results":true' "$OUT/.stdout" || echo "  (not fresh!)" | tee -a "$REPORT"
  git checkout -- "$EDIT_FILE"
  "$BIN" index --quiet >/dev/null 2>&1
done
for _ in $(seq "$RUNS"); do
  measure deps_fresh "$BIN" deps "$EDIT_FILE" --json
done
for _ in $(seq "$RUNS"); do
  measure hotspots "$BIN" analyze --hotspots --json
done

wait_symbols
echo "# mcp session, load at start $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$REPORT"
python3 "$HERE/mcp_edit_latency.py" "$BIN" "$EDIT_FILE" "$RUNS" | tee -a "$REPORT"
git checkout -- "$EDIT_FILE"
"$BIN" index --quiet >/dev/null 2>&1
