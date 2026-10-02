#!/usr/bin/env bash
# Decides whether HEAD needs a release and prepares it (used by auto-release.yml).
#
#   scripts/release-prep.sh [auto|patch|minor|major|skip]
#
# - Cargo.toml names a version with no tag (bumped by hand in a PR): release it as is.
# - Else `## [Unreleased]` in CHANGELOG.md has entries: bump Cargo.toml by the level
#   (auto = minor when a `feat` commit landed since the last tag, else patch).
# - Else: nothing to release.
# When releasing, the [Unreleased] entries move under `## [X.Y.Z] - <today>`.
# Prints `tag=vX.Y.Z` (or `tag=`) and appends it to $GITHUB_OUTPUT when set.
set -euo pipefail

level="${1:-auto}"
case "$level" in auto|patch|minor|major|skip) ;; *) echo "unknown level: $level" >&2; exit 2 ;; esac

emit() {
  echo "tag=$1"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then echo "tag=$1" >>"$GITHUB_OUTPUT"; fi
}

current=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -1)
[ -n "$current" ] || { echo "no version in Cargo.toml" >&2; exit 1; }
last_tag=$(git tag --list 'v[0-9]*' --sort=-v:refname | head -1)

# The [Unreleased] entries, without the heading (empty when there are none).
unreleased=$(awk '/^## \[Unreleased\]/ {on=1; next} on && /^## \[/ {exit} on' CHANGELOG.md \
  | sed '/^[[:space:]]*$/d')

if git rev-parse -q --verify "refs/tags/v$current" >/dev/null; then
  if [ "$level" = skip ] || [ -z "$unreleased" ]; then emit ""; exit 0; fi
  if [ "$level" = auto ]; then
    level=patch
    if git log --format=%s "${last_tag:+$last_tag..}HEAD" | grep -Eq '^feat(\([^)]*\))?!?:'; then
      level=minor
    fi
  fi
  IFS=. read -r major minor patch <<<"${current%%[-+]*}"
  case "$level" in
    major) next="$((major + 1)).0.0" ;;
    minor) next="$major.$((minor + 1)).0" ;;
    patch) next="$major.$minor.$((patch + 1))" ;;
  esac
  sed -i "0,/^version = \"$current\"$/s//version = \"$next\"/" Cargo.toml
else
  next="$current"
fi

if [ -n "$unreleased" ]; then
  python3 - "$next" "$(date -u +%Y-%m-%d)" <<'EOF'
import re, sys
version, date = sys.argv[1], sys.argv[2]
text = open("CHANGELOG.md").read()
new, n = re.subn(r"^## \[Unreleased\][ \t]*\n+", f"## [Unreleased]\n\n## [{version}] - {date}\n\n",
                 text, count=1, flags=re.M)
assert n == 1, "no [Unreleased] heading"
open("CHANGELOG.md", "w").write(new)
EOF
fi
emit "v$next"
