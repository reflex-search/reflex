#!/usr/bin/env bash
# Cold + warm build benchmark of the Pulse template against an extracted runtime.
#
#   scripts/bench.sh <runtime_dir> <pages> [symbols_per_page=8] [mermaid_every=20]
#
# <runtime_dir> must contain node_modules/ (an extracted pulse-runtime tarball).
# The site is staged at <runtime_dir>/sites/bench-<pages>/ so Node's upward
# resolution finds <runtime_dir>/node_modules. Results go to
# <runtime_dir>/results/bench-<pages>.json. Set TIME_BIN to GNU time
# (default /usr/bin/time) and NODE_OPTIONS as needed.
set -euo pipefail

TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$(cd "$1" && pwd)"
N="$2"
K="${3:-8}"
M="${4:-20}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
# GNU time: -v -o FILE; BSD (macOS) /usr/bin/time: -l -o FILE.
if [[ -z "${TIME_FLAGS:-}" ]]; then
	if [[ "$TIME_BIN" != none ]] && "$TIME_BIN" --version 2>&1 | grep -q GNU; then TIME_FLAGS="-v -o"; else TIME_FLAGS="-l -o"; fi
fi
SITE="$RT/sites/bench-$N"
RES="$RT/results${RESULTS_SUFFIX:-}"
mkdir -p "$RES"
ASTRO="$RT/node_modules/astro/bin/astro.mjs"
export ASTRO_TELEMETRY_DISABLED=1

rm -rf "$SITE"
mkdir -p "$SITE"
cp -r "$TEMPLATE/src" "$TEMPLATE/public" "$TEMPLATE/astro.config.mjs" "$TEMPLATE/package.json" "$SITE/"
T0=$(node -p 'Date.now()/1000')
node "$TEMPLATE/scripts/gen-fixture.mjs" --out "$SITE" --pages "$N" --symbols-per-page "$K" --mermaid-every "$M" --base /reflex/ --site https://example.github.io ${PULSE_BUILD:+--build "$PULSE_BUILD"} ${FIXTURE_ARGS:-}
T1=$(node -p 'Date.now()/1000')
FIXTURE_S=$(awk "BEGIN{printf \"%.2f\", $T1-$T0}")
read -r BUNDLE_BYTES FRAGS <<<"$(node -e '
	const fs=require("fs"),path=require("path");let b=0,n=0;
	(function w(d){for(const e of fs.readdirSync(d,{withFileTypes:true})){const p=path.join(d,e.name);
		if(e.isDirectory())w(p);else{b+=fs.statSync(p).size;if(p.endsWith(".md"))n++;}}})(process.argv[1]);
	console.log(b+" "+n);' "$SITE/bundle")"

run() { # label
	local label="$1" log="$RES/bench-$N-$1.log" tv="$RES/bench-$N-$1.time"
	local t0 t1 wall rss
	t0=$(node -p 'Date.now()/1000')
	if [[ "$TIME_BIN" == none ]]; then
		# Windows runners: no GNU time; wall clock only.
		(cd / && node "$ASTRO" build --root "$SITE") >"$log" 2>&1 || { echo "build $label FAILED"; tail -30 "$log"; return 1; }
	else
		(cd / && "$TIME_BIN" $TIME_FLAGS "$tv" node "$ASTRO" build --root "$SITE") >"$log" 2>&1 || {
			echo "build $label FAILED (see $log)"
			tail -30 "$log"
			return 1
		}
	fi
	t1=$(node -p 'Date.now()/1000')
	wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
	rss=null
	if [[ -f "$tv" ]]; then
		# GNU time -v reports KiB; BSD time -l reports bytes.
		rss=$(awk '/Maximum resident set size/{print $NF} /maximum resident set size/{printf "%d", $1/1024}' "$tv" | head -1)
		[[ -n "$rss" ]] || rss=null
	fi
	# Astro phase lines (timestamps stripped).
	local sync types static_routes pf total
	sync=$(grep -m1 -o 'Synced content' "$log" >/dev/null && echo yes || echo no)
	types=$(grep -m1 -oE '\[types\] Generated ([0-9]+m )?[0-9.]+m?s' "$log" | sed 's/.*Generated //')
	static_routes=$(awk '/generating static routes/{f=1} f && /Completed in/{sub(/.*Completed in /,""); sub(/\.$/,""); print; exit}' "$log")
	pf=$(grep -m1 -oE 'Finished building search index in [0-9.]+m?s' "$log" | awk '{print $NF}')
	total=$(grep -m1 -oE 'page\(s\) built in ([0-9]+m )?[0-9.]+m?s' "$log" | sed 's/.*built in //')
	local content_s
	content_s=$(node -e '
		const l=require("fs").readFileSync(process.argv[1],"utf8").split("\n");
		const ts=(re)=>{const x=l.find(s=>re.test(s)); if(!x) return null; const m=x.match(/^(\d\d):(\d\d):(\d\d)/); return m? (+m[1])*3600+(+m[2])*60+(+m[3]) : null;};
		const a=ts(/Syncing content/), b=ts(/Synced content/); console.log(a!=null&&b!=null? b-a : "null");' "$log")
	echo "{\"label\":\"$label\",\"wall_s\":$wall,\"max_rss_kb\":$rss,\"content_sync_s\":$content_s,\"types\":\"$types\",\"static_routes\":\"$static_routes\",\"pagefind\":\"$pf\",\"astro_total\":\"$total\"}"
}

echo "== bench N=$N K=$K M=$M ($FRAGS fragments, bundle $BUNDLE_BYTES bytes)"
rm -rf "$SITE/dist" "$SITE/.astro" "$SITE/.astro-cache" "$SITE/node_modules"
COLD=$(run cold)
echo "cold: $COLD"
SIZES=$(node "$TEMPLATE/scripts/check-dist.mjs" "$SITE" --sizes-only | node -e '
	let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const r=JSON.parse(s);delete r.checks;console.log(JSON.stringify(r))})')
echo "sizes: $SIZES"

node "$TEMPLATE/scripts/gen-fixture.mjs" --out "$SITE" --touch 0.01 --seed 7
WARM=$(run warm)
echo "warm: $WARM"

cat >"$RES/bench-$N.json" <<EOF
{"pages":$N,"symbols_per_page":$K,"mermaid_every":$M,"fragments":$FRAGS,"bundle_bytes":$BUNDLE_BYTES,"fixture_s":$FIXTURE_S,
 "node_options":"${NODE_OPTIONS:-}","pulse_build":${PULSE_BUILD:-null},"fixture_args":"${FIXTURE_ARGS:-}","cores":$(nproc 2>/dev/null || sysctl -n hw.ncpu),
 "cold":$COLD,
 "warm":$WARM,
 "sizes":$SIZES}
EOF
echo "wrote $RES/bench-$N.json"
