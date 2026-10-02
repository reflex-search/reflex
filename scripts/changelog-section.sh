#!/usr/bin/env bash
# Prints the CHANGELOG.md entries of one version (release notes for release.yml).
#
#   scripts/changelog-section.sh 2.2.0   # or v2.2.0
#
# Exits 1 when CHANGELOG.md has no `## [X.Y.Z]` heading.
set -euo pipefail
version="${1#v}"
awk -v h="## [$version]" '
  index($0, h) == 1 {on=1; found=1; next}
  on && /^## \[/ {exit}
  on {print}
  END {exit !found}
' CHANGELOG.md | sed -e '/./,$!d'
