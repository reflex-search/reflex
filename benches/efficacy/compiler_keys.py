"""Answer keys from the Rust compiler, for the `explain` and `impact` sessions
(tasks 0.3 and 0.4 of .context/AGENT_EDGE_PLAN.md).

The trick: add a required parameter to one function in a scratch copy of the corpus,
run `cargo check`, and read the errors. Every error line is a line that must change
(the `impact` key); the function around each error line is a direct caller (the
`explain` callers key). No arm's own tool (Reflex, rust-analyzer, grep) writes a key.

The callees key (what the function calls) comes from the function's body text: every
`name(`, `.name(` and `path::name(` in code, minus keywords, macros, capitalised names
(constructors, variants) and the body's own locals. `rust_code.code_mask` decides what
is code.

Not seen by the compiler, so such questions are dropped:
- code behind a `cfg` this host does not build (`cfg(windows)`): a function named in
  a file the build did not compile (`compiled_files`, from cargo dep-info);
- code in a unit cargo skips once the probed crate fails: unit tests whose unit needs
  the lib (tokio's dev-dependencies) and other packages that depend on it
  (`unchecked`). Text sites there that the compiler did not report drop the question.
`text_sites` lists the code lines that name the function, for the hand check.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import rust_code

KEYS_WORK = Path(os.environ.get("COMPILER_KEYS_WORK", "/scratch/cache/compiler-keys"))

# cargo check runs per corpus: the normal build and the unit-test build of the code
# under the session scope. Integration tests (tokio/tests) are outside every scope.
CHECKS = {
    "reflex": [["--lib", "--bins"], ["--lib", "--bins", "--profile", "test"]],
    "ripgrep": [["--workspace", "--lib", "--bins"], ["--workspace", "--lib", "--bins", "--profile", "test"]],
    "tokio": [["-p", "tokio", "--lib", "--features", "full"],
              ["-p", "tokio", "--lib", "--features", "full", "--profile", "test"]],
}

# Extra cfgs per corpus, so more of the tree is compiled (tokio's CI builds with
# tokio_unstable; task::Builder and other unstable code sit behind it).
RUSTFLAGS = {"tokio": "--cfg tokio_unstable"}

KEYWORDS = set("""as async await break const continue crate dyn else enum extern false fn for
    if impl in let loop match mod move mut pub ref return self Self static struct super trait true
    type unsafe use where while box yield""".split())


def scratch_copy(repo: str, src: Path) -> Path:
    """A copy of the corpus to edit and build (never the pinned corpus: a Cargo.lock or
    target/ written there would change the grep answer keys)."""
    dest = KEYS_WORK / repo
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    subprocess.run(["rsync", "-a", "--exclude", ".git", "--exclude", ".reflex", "--exclude", "target",
                    f"{src}/", f"{dest}/"], check=True)
    return dest


def cargo_env(repo: str) -> dict:
    env = dict(os.environ, CARGO_TARGET_DIR=str(KEYS_WORK / f"target-{repo}"))
    if repo in RUSTFLAGS:
        env["RUSTFLAGS"] = RUSTFLAGS[repo]
    return env


def compiled_files(repo: str, tree: Path) -> set[str]:
    """Source files the checks compiled (cargo dep-info), relative to the tree. A file
    outside this set (`signal/windows/sys.rs` on Linux) is code the compiler key
    cannot see."""
    out = set()
    for d in (KEYS_WORK / f"target-{repo}" / "debug" / "deps").glob("*.d"):
        for tok in d.read_text(errors="replace").replace("\\\n", " ").split():
            tok = tok.rstrip(":")
            if not tok.endswith(".rs"):
                continue
            if not os.path.isabs(tok):  # workspace members: relative to the root
                out.add(os.path.normpath(tok))
            elif tok.startswith(str(tree) + "/"):
                out.add(os.path.relpath(tok, tree))
    return out


def cargo_errors(repo: str, tree: Path) -> list[dict]:
    """Every compile error's primary span: {code, file (relative), line, message}."""
    env = cargo_env(repo)
    out = []
    for args in CHECKS[repo]:
        proc = subprocess.run(["cargo", "check", "--message-format=json", "--quiet", *args],
                              cwd=tree, env=env, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            m = msg.get("message") or {}
            if msg.get("reason") != "compiler-message" or m.get("level") != "error":
                continue
            for span in m.get("spans") or []:
                if not span.get("is_primary"):
                    continue
                f = span["file_name"]
                if os.path.isabs(f):
                    try:
                        f = os.path.relpath(f, tree)
                    except ValueError:
                        continue
                if f.startswith(".."):
                    continue
                out.append({"code": (m.get("code") or {}).get("code"), "file": f,
                            "line": span["line_start"], "message": m.get("message", "")})
    return out


# ---------------------------------------------------------------------------
# Rust text: a function's signature and body
# ---------------------------------------------------------------------------


def _match(src: str, mask: list[bool], i: int, open_c: str, close_c: str) -> int:
    """Index of the bracket closing the one at `i` (code bytes only)."""
    depth = 0
    for j in range(i, len(src)):
        if not mask[j]:
            continue
        c = src[j]
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced")


def fn_parts(src: str, line: int, name: str) -> dict | None:
    """The function `name` defined on 1-based `line`: offsets of its parameter list
    `(`..`)` and body `{`..`}`. None for a declaration without a body."""
    mask = rust_code.code_mask(src)
    start = sum(len(l) + 1 for l in src.split("\n")[: line - 1])
    m = re.compile(r"\bfn\s+" + re.escape(name) + r"\b").search(src, start)
    if not m or not mask[m.start()]:
        return None
    i = m.end()
    while src[i].isspace():
        i += 1
    if src[i] == "<":  # generics; `->` inside them is not a closing bracket
        depth = 0
        while True:
            if src.startswith("->", i):
                i += 2
                continue
            if src[i] == "<":
                depth += 1
            elif src[i] == ">":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        while src[i].isspace():
            i += 1
    if src[i] != "(":
        return None
    p_close = _match(src, mask, i, "(", ")")
    j = p_close + 1
    while j < len(src) and not (mask[j] and src[j] in "{;"):
        j += 1
    if j >= len(src) or src[j] == ";":
        return None
    b_close = _match(src, mask, j, "{", "}")
    return {"p_open": i, "p_close": p_close, "b_open": j, "b_close": b_close, "mask": mask}


def in_trait_impl(src: str, line: int) -> bool:
    """True when the item on `line` sits in a `trait` or an `impl Trait for Type` block
    (its signature is fixed by the trait: adding a parameter breaks the impl, not the
    callers)."""
    lines = src.split("\n")
    indent = len(lines[line - 1]) - len(lines[line - 1].lstrip())
    if indent == 0:
        return False
    for k in range(line - 2, -1, -1):
        l = lines[k]
        if not l.strip():
            continue
        ind = len(l) - len(l.lstrip())
        if ind < indent and re.match(r"\s*(pub(\([^)]*\))?\s+)?(unsafe\s+)?(impl|trait)\b", l):
            return bool(re.match(r"\s*(pub(\([^)]*\))?\s+)?(unsafe\s+)?trait\b", l)) or bool(
                re.search(r"\bimpl\b.*\bfor\b", l))
        if ind < indent and l.strip() not in ("}",) and not l.lstrip().startswith(("#", "//")):
            # an enclosing block that is not impl/trait (mod, macro): keep looking out
            indent = ind
    return False


def add_parameter(src: str, parts: dict) -> str:
    """The source with one more required parameter at the end of the list."""
    params = src[parts["p_open"] + 1: parts["p_close"]]
    sep = "" if not params.strip() else (" " if params.rstrip().endswith(",") else ", ")
    return src[: parts["p_close"]] + sep + "__impact_probe: u32" + src[parts["p_close"]:]


# `<...>` with up to three levels of nesting: `::<Result<Vec<_>, _>>`, `::<()>`.
TURBOFISH = r"<[^<>;{}]*>"
for _ in range(2):
    TURBOFISH = r"<(?:[^<>;{}]|" + TURBOFISH + r")*>"


def callees(src: str, parts: dict) -> list[str]:
    """Functions and methods the body calls directly, by name."""
    body = src[parts["b_open"] + 1: parts["b_close"]]
    mask = parts["mask"][parts["b_open"] + 1: parts["b_close"]]
    code = "".join(c if mask[k] else " " for k, c in enumerate(body))
    # Attributes are not calls: `#[cfg(all(tokio_unstable, feature = "tracing"))]`.
    for m in reversed(list(re.finditer(r"#!?\[", code))):
        depth, j = 0, m.end() - 1
        while j < len(code):
            depth += {"[": 1, "]": -1}.get(code[j], 0)
            if depth == 0:
                break
            j += 1
        code = code[: m.start()] + " " * (j + 1 - m.start()) + code[j + 1:]
    params = src[parts["p_open"] + 1: parts["p_close"]]
    local = set(re.findall(r"\b([a-z_][A-Za-z0-9_]*)\s*:", params))
    local |= set(re.findall(r"\blet\s+(?:mut\s+)?([a-z_][A-Za-z0-9_]*)", code))
    local |= set(re.findall(r"\|\s*([a-z_][A-Za-z0-9_]*)", code))  # closure params
    out = set()
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:::\s*" + TURBOFISH + r"\s*)?\(", code):
        n = m.group(1)
        before = code[: m.start()].rstrip()
        if n in KEYWORDS or n[0].isupper() or code[m.end(1):].lstrip().startswith("!"):
            continue
        if before.endswith("fn"):
            continue  # a nested fn's own name
        if not (before.endswith(".") or before.endswith("::")) and n in local:
            continue  # a closure or fn-typed local
        out.add(n)
    return sorted(out)


def enclosing_fns(src: str) -> list[tuple[int, int, str]]:
    """(start offset, end offset, name) of every fn with a body."""
    mask = rust_code.code_mask(src)
    spans = []
    for m in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", src):
        if not mask[m.start()]:
            continue
        line = src.count("\n", 0, m.start()) + 1
        try:
            parts = fn_parts(src, line, m.group(1))
        except (ValueError, IndexError):
            continue
        if parts:
            spans.append((m.start(), parts["b_close"], m.group(1)))
    return spans


def caller_of(src: str, line: int, spans) -> str | None:
    """The innermost fn whose text holds 1-based `line`."""
    off = sum(len(l) + 1 for l in src.split("\n")[: line - 1])
    inside = [s for s in spans if s[0] <= off <= s[1]]
    return min(inside, key=lambda s: s[1] - s[0])[2] if inside else None


def package_of(tree: Path, rel: str) -> str:
    """The directory (relative) of the nearest Cargo.toml above `rel`."""
    d = (tree / rel).parent
    while d != tree and not (d / "Cargo.toml").exists():
        d = d.parent
    return os.path.relpath(d, tree)


# A `cfg` that names `test` (`test`, `all(test, not(loom))`, ...), then other attributes.
TEST_CFG = r"#\[cfg\((?:[^\]\"]|\"[^\"]*\")*\btest\b(?:[^\]\"]|\"[^\"]*\")*\)\]\s*(?:#\[[^\]]*\]\s*)*"
VIS = r"(?:pub(?:\([^)]*\))?\s+)?"


def test_lines(src: str) -> set[int]:
    """1-based lines inside test-gated `mod name { ... }` blocks and `macro_rules!`
    bodies (a macro is checked only where it is expanded)."""
    mask = rust_code.code_mask(src)
    out = set()
    for m in re.finditer(TEST_CFG + VIS + r"mod\s+\w+\s*\{|\bmacro_rules!\s*\w+\s*\{", src):
        if not mask[m.start()]:
            continue
        try:
            end = _match(src, mask, m.end() - 1, "{", "}")
        except ValueError:
            continue
        out.update(range(src.count("\n", 0, m.start()) + 1, src.count("\n", 0, end) + 2))
    return out


def test_file(tree: Path, rel: str) -> bool:
    """True when `rel` is a test-gated out-of-line module (`#[cfg(test)] mod testutil;`
    in its parent module file), or under a `tests` directory."""
    f = Path(rel)
    if "tests" in f.parts or f.name in ("tests.rs", "test.rs"):
        return True
    stem = f.parent.name if f.name == "mod.rs" else f.stem
    d = f.parent.parent if f.name == "mod.rs" else f.parent
    parents = [d / n for n in ("lib.rs", "main.rs", "mod.rs")] + [d.parent / f"{d.name}.rs"]
    pat = re.compile(TEST_CFG + VIS + r"mod\s+" + re.escape(stem) + r"\s*;")
    return any((tree / q).is_file() and pat.search((tree / q).read_text(errors="replace"))
               for q in parents)


def unchecked(tree: Path, where: str, extra: list[str]) -> list[str]:
    """Lines grep finds and the compiler did not report that sit where an error in the
    probed crate stops cargo from checking: test code (tokio's `lib test` unit needs
    the lib through its dev-dependencies) or another package (it depends on the
    probed one), or a `macro_rules!` body. A question with such a line has a key that may be short."""
    home = package_of(tree, where.rsplit(":", 1)[0])
    out, cache = [], {}
    for site in extra:
        f, ln = site.rsplit(":", 1)
        if package_of(tree, f) != home or test_file(tree, f):
            out.append(site)
            continue
        if f not in cache:
            cache[f] = test_lines((tree / f).read_text(errors="replace"))
        if int(ln) in cache[f]:
            out.append(site)
    return out


def text_sites(tree: Path, name: str, scope: str) -> list[str]:
    """Code lines (not comments or strings) under `scope` that name `name`."""
    out = subprocess.run(["rg", "-l", "-w", "-g", "*.rs", name, scope], cwd=tree,
                         capture_output=True, text=True).stdout.split()
    return sorted(f"{f}:{n}" for f in out
                  for n in rust_code.code_lines((tree / f).read_text(errors="replace"), name))


def probe(repo: str, tree: Path, scope: str, name: str, where: str,
          compiled: set[str] | None = None) -> dict | None:
    """Add a parameter to `name` at `where` (path:line), build, restore. Returns the
    keys, or None when the function cannot be probed (no body, trait method, no
    error, or an error the parameter cannot explain)."""
    path, line = where.rsplit(":", 1)
    line = int(line)
    f = tree / path
    src = f.read_text()
    if in_trait_impl(src, line):
        return None
    text = text_sites(tree, name, scope)
    if compiled is not None and any(t.rsplit(":", 1)[0] not in compiled for t in text):
        return None  # the name is used in code this build does not compile
    try:
        parts = fn_parts(src, line, name)
    except (ValueError, IndexError):
        return None
    if not parts:
        return None
    f.write_text(add_parameter(src, parts))
    try:
        errors = cargo_errors(repo, tree)
    finally:
        f.write_text(src)
    in_scope = [e for e in errors if e["file"].startswith(scope + "/")]
    sites = sorted({f"{e['file']}:{e['line']}" for e in in_scope})
    callers = set()
    cache: dict[str, tuple[str, list]] = {}
    for s in sites:
        p, ln = s.rsplit(":", 1)
        if p not in cache:
            t = (tree / p).read_text(errors="replace")
            cache[p] = (t, enclosing_fns(t))
        c = caller_of(cache[p][0], int(ln), cache[p][1])
        if c:
            callers.add(c)
    if unchecked(tree, where, sorted(set(text) - set(sites) - {where})):
        return None
    return {
        "name": name, "where": where, "sites": sites, "callers": sorted(callers),
        "calls": callees(src, parts), "codes": sorted({str(e["code"]) for e in in_scope}),
        "errors_outside_scope": len(errors) - len(in_scope),
        "text_sites": text,
    }
