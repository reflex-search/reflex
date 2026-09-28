#!/usr/bin/env bash
# Build the Pulse renderer runtime tarball (node_modules only, pruned, zstd).
#
#   scripts/build-runtime.sh [OUT_DIR] [--verify]
#
# Produces OUT_DIR/pulse-runtime-<platform>-<arch>.tar.zst and prints sizes.
# The tarball is platform-specific: rolldown, lightningcss, satteri, esbuild
# and pagefind all ship native binaries selected by npm at install time.
#
# --verify extracts the tarball into OUT_DIR/verify, stages fixtures/smoke at
# OUT_DIR/verify/sites/smoke/ (so Node's upward resolution finds
# OUT_DIR/verify/node_modules) and builds it with
#   node <runtime>/node_modules/astro/bin/astro.mjs build --root <site>
# from the site directory, as rfx does.
set -euo pipefail

TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$TEMPLATE/.spike/runtime}"
VERIFY=0
for a in "$@"; do [[ "$a" == "--verify" ]] && VERIFY=1; done
[[ "$OUT" == "--verify" ]] && OUT="$TEMPLATE/.spike/runtime"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

PLATFORM="$(node -p 'process.platform + "-" + process.arch')"
TARBALL="$OUT/pulse-runtime-$PLATFORM.tar.zst"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/pulse-runtime.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

stats() { # dir -> "bytes files" (portable: GNU/BSD du and stat disagree)
	node -e '
		const fs=require("fs"),path=require("path");let b=0,n=0;
		(function w(d){for(const e of fs.readdirSync(d,{withFileTypes:true})){const p=path.join(d,e.name);
			if(e.isDirectory())w(p);else if(e.isFile()){n++;b+=fs.statSync(p).size;}}})(process.argv[1]);
		console.log(b+" "+n);' "$1"
}
fsize() { node -p 'require("fs").statSync(process.argv[1]).size' "$1"; }
# Reproducible archive flags exist only in GNU tar (bsdtar on macOS/Windows lacks --sort).
TAR_REPRO=()
if tar --version 2>/dev/null | grep -q GNU; then
	TAR_REPRO=(--sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner)
fi

echo "== npm ci (production, no scripts) in $WORK"
cp "$TEMPLATE/package.json" "$TEMPLATE/package-lock.json" "$WORK/"
(cd "$WORK" && npm ci --ignore-scripts --omit=dev --no-audit --no-fund --loglevel=error)
read -r RAW_BYTES RAW_FILES <<<"$(stats "$WORK/node_modules")"

echo "== prune"
node "$TEMPLATE/scripts/prune.mjs" "$WORK"

echo "== pagefind: plain binary instead of pagefind_extended"
# pagefind's resolver tries pagefind_extended, then pagefind (or $PAGEFIND_BINARY_PATH).
# The path goes in argv, not inside the JS string: Git Bash on Windows converts
# /tmp/... only when it is a separate argument.
PF_VER="$(node -p 'require(process.argv[1]).version' "$WORK/node_modules/pagefind/package.json")"
case "$PLATFORM" in
	linux-x64) PF_TRIPLE=x86_64-unknown-linux-musl PF_DIR=linux-x64 ;;
	linux-arm64) PF_TRIPLE=aarch64-unknown-linux-musl PF_DIR=linux-arm64 ;;
	darwin-arm64) PF_TRIPLE=aarch64-apple-darwin PF_DIR=darwin-arm64 ;;
	darwin-x64) PF_TRIPLE=x86_64-apple-darwin PF_DIR=darwin-x64 ;;
	win32-x64) PF_TRIPLE=x86_64-pc-windows-msvc PF_DIR=windows-x64 ;;
	win32-arm64) PF_TRIPLE=aarch64-pc-windows-msvc PF_DIR=windows-arm64 ;;
	*) echo "unsupported platform $PLATFORM" >&2; exit 1 ;;
esac
PF_TGZ="pagefind-v$PF_VER-$PF_TRIPLE.tar.gz"
curl -fsSL -o "$WORK/$PF_TGZ" "https://github.com/Pagefind/pagefind/releases/download/v$PF_VER/$PF_TGZ"
PF_WANT="$(cut -d' ' -f1 "$WORK/node_modules/pagefind/checksums/$PF_TGZ.sha256")"
PF_GOT="$( (sha256sum "$WORK/$PF_TGZ" 2>/dev/null || shasum -a 256 "$WORK/$PF_TGZ") | cut -d' ' -f1)"
[[ "$PF_WANT" == "$PF_GOT" ]] || { echo "pagefind checksum mismatch: $PF_GOT != $PF_WANT" >&2; exit 1; }
mkdir -p "$WORK/node_modules/@pagefind/$PF_DIR/bin"
tar -C "$WORK/node_modules/@pagefind/$PF_DIR/bin" -xzf "$WORK/$PF_TGZ"
rm -f "$WORK/$PF_TGZ"
ls -la "$WORK/node_modules/@pagefind/$PF_DIR/bin"
read -r PRUNED_BYTES PRUNED_FILES <<<"$(stats "$WORK/node_modules")"

echo "== tar | zstd -19"
T0=$(node -p 'Date.now()/1000')
tar -C "$WORK" ${TAR_REPRO[@]+"${TAR_REPRO[@]}"} -cf - node_modules \
	| zstd -19 -T0 -q -f -o "$TARBALL"
T1=$(node -p 'Date.now()/1000')
ZST_BYTES=$(fsize "$TARBALL")

cat >"$OUT/runtime-stats.json" <<EOF
{
  "platform": "$PLATFORM",
  "tarball": "$(basename "$TARBALL")",
  "raw_bytes": $RAW_BYTES,
  "raw_files": $RAW_FILES,
  "pruned_bytes": $PRUNED_BYTES,
  "pruned_files": $PRUNED_FILES,
  "zst_bytes": $ZST_BYTES,
  "compress_seconds": $(awk "BEGIN{printf \"%.2f\", $T1-$T0}")
}
EOF
cat "$OUT/runtime-stats.json"

if [[ $VERIFY == 1 ]]; then
	echo "== verify: extract + build fixture from the runtime only"
	RT="$OUT/verify"
	rm -rf "$RT"
	mkdir -p "$RT/sites"
	T0=$(node -p 'Date.now()/1000')
	zstd -d -q -c "$TARBALL" | tar -C "$RT" -xf -
	T1=$(node -p 'Date.now()/1000')
	echo "extract_seconds=$(awk "BEGIN{printf \"%.2f\", $T1-$T0}")"
	SITE="$RT/sites/smoke"
	mkdir -p "$SITE"
	cp -r "$TEMPLATE/src" "$TEMPLATE/public" "$TEMPLATE/astro.config.mjs" "$TEMPLATE/package.json" "$SITE/"
	# fixtures/smoke is a real `rfx pulse generate --no-build` bundle (base /reflex/).
	cp -r "$TEMPLATE/fixtures/smoke/." "$SITE/"
	# Run the way rfx does (runtime.rs: cwd = the site): packages resolve upward to
	# <runtime>/node_modules only. Astro writes .astro/ under the cwd, so an unrelated
	# cwd breaks on Windows (EXDEV when it is on another drive).
	(cd "$SITE" && ASTRO_TELEMETRY_DISABLED=1 node "$RT/node_modules/astro/bin/astro.mjs" build --root "$SITE")
	node "$TEMPLATE/scripts/smoke-check.mjs" "$SITE" || { echo "verify: smoke checks FAILED"; exit 1; }
fi
