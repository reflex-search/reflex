#!/usr/bin/env bash
# Install smoke test (task 1.5 of .context/AGENT_EDGE_PLAN.md).
#
# In a clean container: install rfx from npm, run the first CLI query and the first
# MCP call on a fresh clone of this repository (tracked files only), and time each
# step. Optionally install the Claude Code plugins from this checkout's marketplace.
#
#   scripts/install-smoke.sh [image] [npm-spec]
#     image     default node:24-bookworm (needs node, npm and git)
#     npm-spec  default reflex-search (for example reflex-search@2.3.1)
#   PLUGINS=1 also installs @anthropic-ai/claude-code and the two plugins.
#
# Needs docker and network access to npm (and to GitHub, where the npm package
# downloads the rfx binary). The repository is mounted read-only at /src.
set -euo pipefail

image="${1:-node:24-bookworm}"
spec="${2:-reflex-search}"
repo="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm -i \
  -v "$repo:/src:ro" \
  -e SPEC="$spec" -e PLUGINS="${PLUGINS:-0}" \
  "$image" sh -s <<'EOS'
set -eu
step() { printf '\n== %s\n' "$*"; }
now() { date +%s.%N; }
took() { awk -v a="$1" -v b="$(now)" 'BEGIN { printf "   took %.2f s\n", b - a }'; }

step "environment"
uname -m; cat /etc/os-release | grep PRETTY_NAME; node --version; npm --version
command -v git >/dev/null || { echo "git missing"; exit 1; }
git config --global --add safe.directory '*'

step "npm install -g $SPEC"
t=$(now); npm install -g "$SPEC" 2>&1 | tail -3; took "$t"
command -v rfx; rfx --version

step "corpus: git clone of the mounted repository"
t=$(now)
git clone -q /src /work/cli
git clone -q /src /work/mcp
took "$t"
echo "   files: $(git -C /work/cli ls-files | wc -l)"

step "first CLI query (builds the index): rfx query update_if_stale --count"
cd /work/cli
t=$(now); rfx query update_if_stale --count 2>&1 | tail -2; took "$t"

step "second CLI query (index fresh): rfx query update_if_stale --count"
t=$(now); rfx query update_if_stale --count 2>&1 | tail -1; took "$t"

step "first MCP call on a clone with no index: initialize + search_code"
cd /work/mcp
t=$(now)
printf '%s\n' \
  '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_code","arguments":{"pattern":"update_if_stale","mode":"count"}}}' \
  | rfx mcp 2>/tmp/mcp.err | tail -1 | cut -c1-200
took "$t"
grep -E "reflex-mcp (startup|index)" /tmp/mcp.err | cut -c1-200 || true

step "second MCP call (index fresh)"
t=$(now)
printf '%s\n' \
  '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_code","arguments":{"pattern":"update_if_stale","mode":"count"}}}' \
  | rfx mcp 2>/dev/null | tail -1 | cut -c1-200
took "$t"

if [ "$PLUGINS" = 1 ]; then
  step "Claude Code plugins from the local marketplace (/src)"
  t=$(now); npm install -g @anthropic-ai/claude-code 2>&1 | tail -1; took "$t"
  claude --version
  t=$(now)
  claude plugin marketplace add /src 2>&1 | tail -2
  claude plugin install reflex@reflex 2>&1 | tail -2
  claude plugin install reflex-mcp@reflex 2>&1 | tail -2
  took "$t"
  claude plugin list 2>&1 | grep -iA3 reflex || true
  cd /work/cli && claude mcp list 2>&1 | grep -i reflex || true
fi
EOS
