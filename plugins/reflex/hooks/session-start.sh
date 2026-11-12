#!/bin/sh
# SessionStart hook of the reflex plugin. Its stdout becomes session context.
# Prints the routing rules only when rfx is on PATH: never route to a missing tool.
command -v rfx >/dev/null 2>&1 || exit 0
cat "$(dirname "$0")/routing.md"
