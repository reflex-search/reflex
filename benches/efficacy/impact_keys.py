"""Answer keys for `impact-wide` questions: "which lines must change if fn X changes?"
for functions picked without bias (any name, shared or not), keyed by the compiler.

The old `impact` sessions (session_bench.py `candidates()`) pick only `pub` items with a
name defined once and 5-30 grep hits. In tokio/src, 71 % of fn definitions share a name
with another definition (2026-11-12), so those sessions measure the easy case only.

Key ("deprecation probe"): put `#[deprecated(note = "REFLEX_PROBE")]` above the fn and
run `cargo check` for every CHECKS entry. Each warning's primary span is a use site.
Unlike adding a parameter, the build keeps going, so test code is checked too. It
matched the parameter key exactly where both ran (tokio set_thread_id, copy_to). Trait
impl methods cannot carry the attribute; they are skipped.

Don't care: a line that names the fn, is not in the key, and sits in code no build
compiled (a `#[cfg]` branch or block, an unexpanded macro, a file outside the build).
The key cannot judge those lines, so a grader drops them from both sides. Found with a
live probe: an undefined name put in the enclosing fn body (or `#[cfg] { .. }` block);
no error means the build never saw it.

  python3 impact_keys.py REPO N OUT.json [SKIP]
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import compiler_keys as ck

MARK = "REFLEX_PROBE"
USE = re.compile(r"^\s*(pub(\([^)]*\))?\s+)?use\b")
FN = re.compile(r"^\s*(pub(\([a-z]+\))?\s+)?(async\s+)?(const\s+)?(unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")


def envs(repo: str) -> list[dict]:
    """The configured build, plus one without RUSTFLAGS cfgs (their `not(..)` branches)."""
    out = [ck.cargo_env(repo)]
    if repo in ck.RUSTFLAGS:
        out.append(dict(os.environ, CARGO_TARGET_DIR=str(ck.KEYS_WORK / f"target-{repo}-nocfg")))
    return out


def check_messages(repo: str, tree: Path):
    for env in envs(repo):
        for args in ck.CHECKS[repo]:
            p = subprocess.run(["cargo", "check", "--message-format=json", "--quiet", *args],
                               cwd=tree, env=env, capture_output=True, text=True)
            for line in p.stdout.splitlines():
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if m.get("reason") == "compiler-message":
                    yield m.get("message") or {}


def deprecation_sites(repo: str, tree: Path, where: str, scope: str) -> tuple[list[str], list[str]]:
    """(use sites under `scope`, error messages). Errors mean the probe is invalid."""
    path, line = where.rsplit(":", 1)
    line = int(line)
    f = tree / path
    src = f.read_text()
    lines = src.split("\n")
    indent = lines[line - 1][: len(lines[line - 1]) - len(lines[line - 1].lstrip())]
    lines.insert(line - 1, f'{indent}#[deprecated(note = "{MARK}")]')
    f.write_text("\n".join(lines))
    out, errs = set(), []
    try:
        for m in check_messages(repo, tree):
            if m.get("level") == "error":
                errs.append(m.get("message"))
            if MARK not in m.get("message", ""):
                continue
            for s in m.get("spans") or []:
                if s.get("is_primary"):
                    fn = s["file_name"]
                    fn = os.path.relpath(fn, tree) if os.path.isabs(fn) else fn
                    ln = s["line_start"]
                    if fn == path and ln >= line:
                        ln -= 1  # undo the inserted line
                    out.add(f"{fn}:{ln}")
    finally:
        f.write_text(src)
    return sorted(s for s in out if s.startswith(scope + "/") and s != where), errs


def in_use(lines: list[str], line: int) -> bool:
    """`line` (1-based) is inside a `use` declaration (a re-export, not a use site)."""
    for i in range(line - 1, max(-1, line - 30), -1):
        if USE.match(lines[i]):
            return True
        t = lines[i].strip()
        if i < line - 1 and t.endswith("{") and re.fullmatch(r"[\w:]*::\{", t):
            continue  # `util::{` opens a group inside a `use` list
        if i < line - 1 and t.endswith((";", "}", "{")):
            return False
    return False


def drop_uses(tree: Path, sites: list[str]) -> list[str]:
    cache: dict[str, list[str]] = {}
    out = []
    for s in sites:
        p, ln = s.rsplit(":", 1)
        if p not in cache:
            cache[p] = (tree / p).read_text(errors="replace").split("\n")
        if not in_use(cache[p], int(ln)):
            out.append(s)
    return out


def cfg_block(src: str, at: int, floor: int) -> int | None:
    """The innermost `{` before byte `at` (after `floor`), still open at `at`, whose
    statement carries a `#[cfg(..)]` attribute."""
    depth, i = 0, at - 1
    while i > floor:
        c = src[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                head = src[src.rfind("\n", 0, src.rfind("\n", 0, i)) + 1 : i]
                if re.search(r"#\[cfg\(", head) and "]" in head and src[i - 1] in " \n\t":
                    return i
            else:
                depth -= 1
        i -= 1
    return None


CFG_ATTR = re.compile(r"^\s*(#\[cfg\(.*\)\])\s*$")


def live(repo: str, tree: Path, sites: list[str]) -> set[str]:
    """The subset of `sites` the build compiles: an undefined name at the start of the
    enclosing fn body (or `#[cfg]` block), on the same line so no line moves. A
    statement under a one-line `#[cfg(..)]` gets its own probe: `let _ = X; #[cfg(..)]`
    put before it, so the old attribute holds the probe and the copy holds the
    statement."""
    by_file = collections.defaultdict(list)
    for s in sites:
        p, ln = s.rsplit(":", 1)
        by_file[p].append(int(ln))
    marks: dict[int, list[str]] = {}
    edits: dict[str, tuple[str, str]] = {}
    k = 0
    for p, lns in by_file.items():
        src = (tree / p).read_text(errors="replace")
        spans = ck.enclosing_fns(src)
        offs = [0]
        for text in src.split("\n"):
            offs.append(offs[-1] + len(text) + 1)
        ins: dict[int, int] = {}
        stmt: dict[int, tuple[int, str]] = {}  # byte -> (mark, attribute)
        lines = src.split("\n")
        for ln in lns:
            a = offs[ln - 1]
            attr = CFG_ATTR.match(lines[ln - 2]) if ln >= 2 else None
            if attr:
                b = a + len(lines[ln - 1]) - len(lines[ln - 1].lstrip())
                if b not in stmt:
                    stmt[b] = (k, attr.group(1))
                    marks[k] = []
                    k += 1
                marks[stmt[b][0]].append(f"{p}:{ln}")
                continue
            inner = [sp for sp in spans if sp[0] <= a <= sp[1]]
            if not inner:
                continue
            sp = max(inner, key=lambda x: x[0])
            parts = ck.fn_parts(src, src.count("\n", 0, sp[0]) + 1, sp[2])
            if not parts:
                continue
            b = parts["b_open"]
            cb = cfg_block(src, a, b)
            if cb is not None:
                b = cb
            if b not in ins:
                ins[b] = k
                marks[k] = []
                k += 1
            marks[ins[b]].append(f"{p}:{ln}")
        edit = {b: (b + 1, f" let _ = __REFLEX_LIVE_{i};") for b, i in ins.items()}
        edit.update({b: (b, f"let _ = __REFLEX_LIVE_{i}; {attr} ") for b, (i, attr) in stmt.items()})
        out = src
        for b in sorted(edit, reverse=True):
            at, text = edit[b]
            out = out[:at] + text + out[at:]
        edits[p] = (src, out)
    groups = collections.defaultdict(list)
    for p in edits:
        groups[ck.package_of(tree, p)].append(p)
    hit = set()
    for files in groups.values():
        for p in files:
            (tree / p).write_text(edits[p][1])
        try:
            for m in check_messages(repo, tree):
                for x in re.finditer(r"__REFLEX_LIVE_(\d+)", m.get("message", "")):
                    hit.add(int(x.group(1)))
        finally:
            for p in files:
                (tree / p).write_text(edits[p][0])
    return {s for i in hit for s in marks.get(i, [])}


def candidates(repo: str, tree: Path, scope: str) -> list[tuple[str, str]]:
    """Every fn definition in a compiled file, in a fixed hash order: (name, path:line)."""
    compiled = ck.compiled_files(repo, tree)
    raw = subprocess.run(["rg", "--no-heading", "-n", "-g", "*.rs", FN.pattern, scope],
                         cwd=tree, capture_output=True, text=True).stdout.splitlines()
    out = []
    for row in raw:
        f, ln, txt = row.split(":", 2)
        if f in compiled:
            name = re.search(r"fn\s+([A-Za-z_][A-Za-z0-9_]*)", txt).group(1)
            out.append((hashlib.sha1(f"{f}:{ln}".encode()).hexdigest(), name, f"{f}:{ln}"))
    return [(n, w) for _, n, w in sorted(out)]


def questions(repo: str, tree: Path, scope: str, n: int, skip: int = 0,
              lo: int = 3, hi: int = 25) -> list[dict]:
    """`n` questions (after `skip`) with lo..hi use sites, keyed and with don't-cares."""
    out = []
    for name, where in candidates(repo, tree, scope):
        if len(out) >= n + skip:
            break
        p, ln = where.rsplit(":", 1)
        src = (tree / p).read_text()
        if ck.in_trait_impl(src, int(ln)):
            continue
        try:
            if not ck.fn_parts(src, int(ln), name):
                continue
        except (ValueError, IndexError):
            continue
        sites, errs = deprecation_sites(repo, tree, where, scope)
        if errs:
            print(f"  skip {where}: {errs[0]}", flush=True)
            continue
        sites = drop_uses(tree, sites)
        if not lo <= len(sites) <= hi:
            continue
        if len(out) < skip:
            out.append({"where": where})  # skipped: only counted
            continue
        text = ck.text_sites(tree, name, scope)
        unreported = sorted(set(text) - set(sites) - {where})
        seen = live(repo, tree, unreported) if unreported else set()
        dontcare = sorted(drop_uses(tree, sorted(set(unreported) - seen)))
        out.append({"repo": repo, "scope": scope, "name": name, "where": where, "sites": sites,
                    "dontcare": dontcare, "grep_lines": len(text)})
        print(f"  {len(out)} {name} {where}: {len(sites)} sites, grep {len(text)}, "
              f"don't care {len(dontcare)}", flush=True)
    return out[skip:]


if __name__ == "__main__":
    import runner
    repo, n, dest = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    skip = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    scope = {"reflex": "src", "ripgrep": "crates", "tokio": "tokio/src"}[repo]
    tree = ck.KEYS_WORK / repo
    if not tree.exists():
        tree = ck.scratch_copy(repo, runner.CORPUS_REPOS[repo])
    if ck.cargo_errors(repo, tree):  # also the build `compiled_files` reads
        raise SystemExit(f"{repo}: the unchanged corpus does not build; no compiler keys")
    json.dump(questions(repo, tree, scope, n, skip), open(dest, "w"), indent=1)
