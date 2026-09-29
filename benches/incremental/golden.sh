#!/usr/bin/env bash
# Golden battery for the incremental-index work (see golden.py for the battery itself).
#
#   golden.sh setup                  clone the pinned corpora into $WORK (once)
#   golden.sh capture <rfx> <label>  fresh index + battery on the pristine copies
#   golden.sh updates <rfx> <label>  scripted edits with `rfx index` after each, then
#                                    revert; battery on the updated index and on a fresh
#                                    index of the same directory; both are kept
#   golden.sh compare <labelA> <labelB>
#   golden.sh verify <label>         compare a capture with reference.sha256 (2.0.3)
#
# Pristine copies are never edited, so their readdir order (and therefore walk order)
# stays the one the reference was captured with. `updates` works on separate copies and
# compares against a fresh build made in the same directory.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${GOLDEN_WORK:-/scratch/cache/incremental-work}"
GOLD="${GOLDEN_OUT:-/scratch/cache/incremental-golden}"
CORPORA=(reflex ripgrep tokio corpus)

declare -A SRC=(
  [reflex]="$REPO"
  [ripgrep]="$REPO/benches/efficacy/corpus/ripgrep"
  [tokio]="$REPO/benches/efficacy/corpus/tokio"
)
declare -A PIN=(
  [reflex]=d2935f48f5abea2a76b479040a23478155be9bb0
  [ripgrep]=4649aa9700619f94cf9c66876e9549d83420e16c
  [tokio]=ab3ff69cf2258a8c696b2dca89a2cef4ff114c1c
)

make_copy() { # <name> <dest>
  local name="$1" dest="$2"
  rm -rf "$dest"
  mkdir -p "$(dirname "$dest")"
  if [[ "$name" == corpus ]]; then
    rsync -a --exclude .reflex "$REPO/tests/corpus/" "$dest/"
  else
    git clone --quiet --no-hardlinks "${SRC[$name]}" "$dest"
    git -C "$dest" -c advice.detachedHead=false checkout --quiet "${PIN[$name]}"
  fi
}

cmd_setup() {
  for c in "${CORPORA[@]}"; do
    make_copy "$c" "$WORK/pristine/$c"
    make_copy "$c" "$WORK/upd/$c"
    echo "copied $c"
  done
}

cmd_capture() { # <rfx> <label>
  local bin="$1" label="$2"
  for c in "${CORPORA[@]}"; do
    rm -rf "$WORK/pristine/$c/.reflex"
    python3 "$HERE/golden.py" run --rfx "$bin" --tree "$WORK/pristine/$c" \
      --out "$GOLD/$label/$c" --index
    echo "captured $label/$c"
  done
}

# Code files to edit, chosen by sorted path so every run picks the same ones.
code_files() { # <tree>
  (cd "$1" && find . -path ./.reflex -prune -o -path ./.git -prune -o -type f \
    \( -name '*.rs' -o -name '*.go' -o -name '*.py' -o -name '*.ts' -o -name '*.js' \) \
    -print | sed 's|^\./||' | LC_ALL=C sort)
}

index() { "$1" index --quiet >/dev/null; }

cmd_updates() { # <rfx> <label>
  local bin="$1" label="$2"
  for c in "${CORPORA[@]}"; do
    local t="$WORK/upd/$c" bak
    bak="$(mktemp -d "$WORK/bak.XXXXXX")"
    rm -rf "$t/.reflex" "$t/.reflex-fresh"
    if [[ -d "$t/.git" ]]; then
      # An older tree first, then the pin: a large change set through `rfx index`.
      # Skipped when HEAD~3's blobs are missing (a partial-clone source).
      if git -C "$t" -c advice.detachedHead=false checkout --quiet HEAD~3 2>/dev/null \
        && [[ "$(git -C "$t" rev-parse HEAD)" == "$(git -C "$t" rev-parse "${PIN[$c]}~3")" ]] \
        && [[ -z "$(git -C "$t" status --porcelain --untracked-files=no)" ]]; then
        index "$bin" "$t"
      else
        echo "note: $c: HEAD~3 not available, skipping the older-tree step"
      fi
      git -C "$t" -c advice.detachedHead=false checkout --quiet -f "${PIN[$c]}"
      index "$bin" "$t"
    else
      index "$bin" "$t"
    fi
    mapfile -t files < <(code_files "$t")
    local a="${files[0]}" b="${files[1]}" cfile="${files[2]}" d="${files[3]}"
    for f in "$a" "$b" "$cfile" "$d"; do mkdir -p "$bak/$(dirname "$f")"; cp -p "$t/$f" "$bak/$f"; done
    local ext="${a##*.}" dir
    dir="$(dirname "$a")"
    # 1. append  2. add  3. delete  4. rename  5. atomic-save rewrite  6. add then delete
    echo "// incremental probe" >>"$t/$a"; index "$bin" "$t"
    printf 'zz_probe_added_token\n' >"$t/$dir/zz_probe_new.$ext"; index "$bin" "$t"
    rm "$t/$b"; index "$bin" "$t"
    mv "$t/$cfile" "$t/$cfile.renamed.$ext"; index "$bin" "$t"
    { cat "$t/$d"; echo "// rewritten"; } >"$t/$d.tmp" && mv "$t/$d.tmp" "$t/$d"; index "$bin" "$t"
    printf 'short lived\n' >"$t/$dir/zz_probe_tmp.$ext"; index "$bin" "$t"
    rm "$t/$dir/zz_probe_tmp.$ext"; index "$bin" "$t"
    # Revert every edit, reindexing after each group.
    cp -p "$bak/$a" "$t/$a"; rm "$t/$dir/zz_probe_new.$ext"; index "$bin" "$t"
    cp -p "$bak/$b" "$t/$b"; mv "$t/$cfile.renamed.$ext" "$t/$cfile"; index "$bin" "$t"
    cp -p "$bak/$d" "$t/$d"; index "$bin" "$t"
    rm -rf "$bak"
    python3 "$HERE/golden.py" run --rfx "$bin" --tree "$t" --out "$GOLD/$label-upd/$c"
    mv "$t/.reflex" "$t/.reflex-inc"
    index "$bin" "$t"
    python3 "$HERE/golden.py" run --rfx "$bin" --tree "$t" --out "$GOLD/$label-upd-fresh/$c"
    rm -rf "$t/.reflex"; mv "$t/.reflex-inc" "$t/.reflex"
    echo "updated $label/$c"
  done
  cmd_compare "$label-upd" "$label-upd-fresh"
}

cmd_compare() { # <labelA> <labelB>
  local status=0
  for c in "${CORPORA[@]}"; do
    echo "== $c"
    python3 "$HERE/golden.py" diff "$GOLD/$1/$c" "$GOLD/$2/$c" || status=1
  done
  return $status
}

cmd_verify() { # <label>
  local tmp; tmp="$(mktemp)"
  for c in "${CORPORA[@]}"; do
    python3 "$HERE/golden.py" sums "$GOLD/$1/$c" | sed "s|  |  $c/|"
  done >"$tmp"
  if diff <(grep -v '^#' "$HERE/reference.sha256") "$tmp" >/dev/null; then
    echo "matches reference.sha256"
  else
    diff <(grep -v '^#' "$HERE/reference.sha256") "$tmp" | grep '^[<>]' | awk '{print $1, $3}' | sort -k2 -u
    rm -f "$tmp"; return 1
  fi
  rm -f "$tmp"
}

case "${1:-}" in
  setup) cmd_setup ;;
  capture) cmd_capture "$2" "$3" ;;
  updates) cmd_updates "$2" "$3" ;;
  compare) cmd_compare "$2" "$3" ;;
  verify) cmd_verify "$2" ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
