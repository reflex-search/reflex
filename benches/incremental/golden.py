#!/usr/bin/env python3
"""Golden battery for the incremental-index work.

Runs a fixed battery of `rfx` commands against an indexed tree and writes one
normalized output file per command. Two runs are equal when their files are
byte-identical; `diff` reports the files that differ.

Subcommands:
  run   --rfx BIN --tree DIR --out DIR [--index]   run the battery (optionally index first)
  diff  A B                                        compare two output directories
  sums  DIR                                        print "sha256  name" for every output

Only fields that are timestamps, timings or cache-size figures are normalized;
everything else must match byte for byte.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

# Keys whose values are timings, timestamps or cache sizes. Replaced by "<N>".
VOLATILE_KEYS = {
    "timings",
    "timing_ms",
    "last_updated",
    "indexed_at",
    "last_indexed",
    "duration_ms",
    "index_size_bytes",
    "trigram_index_bytes",
    "started_at",
    "updated_at",
    "completed_at",
    "elapsed_ms",
    "took_ms",
}

# Fields serialized from a HashMap: their key order is random in 2.0.3 itself.
HASHMAP_FIELDS = {"files_by_language", "lines_by_language"}

# Commands whose line order is random in 2.0.3 itself (HashMap iteration): lines
# after the first are compared as a sorted set.
UNORDERED_TEXT = {"d_deps_reverse_text"}

# JSON outputs whose list order is random in 2.0.3 itself (transitive deps come from
# a HashMap): every list of objects is compared as a sorted set.
UNORDERED_JSON = {"d_deps_depth", "m_get_transitive_deps"}


def sort_lists(value):
    if isinstance(value, dict):
        return {k: sort_lists(v) for k, v in value.items()}
    if isinstance(value, list):
        items = [sort_lists(v) for v in value]
        if all(isinstance(v, dict) for v in items):
            items.sort(key=lambda v: json.dumps(v, sort_keys=True))
        return items
    return value

# Text-mode lines/fragments that carry timings, timestamps or cache sizes.
TEXT_RULES = [
    (re.compile(r"\b\d+(\.\d+)?\s?(ms|µs|us|s)\b"), "<T>"),
    (re.compile(r"(Last updated:\s*).*"), r"\1<TS>"),
    (re.compile(r"(Cache size:\s*).*"), r"\1<SIZE>"),
    (re.compile(r"(Index size:\s*).*"), r"\1<SIZE>"),
    (re.compile(r"(Index/corpus ratio:\s*).*"), r"\1<RATIO>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?"), "<TS>"),
]


def normalize_json(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in VOLATILE_KEYS:
                out[k] = "<N>"
            elif k in HASHMAP_FIELDS and isinstance(v, dict):
                out[k] = {kk: normalize_json(v[kk]) for kk in sorted(v)}
            else:
                out[k] = normalize_json(v)
        return out
    if isinstance(value, list):
        return [normalize_json(v) for v in value]
    return value


def normalize_text(text):
    for pattern, repl in TEXT_RULES:
        text = pattern.sub(repl, text)
    return text


def normalize(stdout):
    """JSON (one document or one per line) is normalized structurally; text by rules."""
    stripped = stdout.strip()
    if not stripped:
        return ""
    try:
        return json.dumps(normalize_json(json.loads(stripped)), indent=1, sort_keys=False) + "\n"
    except ValueError:
        pass
    lines = stripped.splitlines()
    out = []
    all_json = True
    for line in lines:
        try:
            out.append(json.dumps(normalize_json(json.loads(line)), sort_keys=False))
        except ValueError:
            all_json = False
            break
    if all_json:
        return "\n".join(out) + "\n"
    return normalize_text(stdout)


def rfx(bin_path, tree, args, timeout=900):
    env = dict(os.environ)
    env.pop("RUST_LOG", None)
    env["NO_COLOR"] = "1"
    proc = subprocess.run(
        [bin_path] + args,
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def wait_for_symbol_pass(bin_path, tree, limit_s=300):
    deadline = time.time() + limit_s
    while time.time() < deadline:
        _, out, _ = rfx(bin_path, tree, ["index", "status"])
        if "Running" not in out:
            return
        time.sleep(0.5)
    print("warning: symbol pass still running after %ss" % limit_s, file=sys.stderr)


# The query battery: (name, args). Patterns are generic so they hit every corpus.
QUERIES = [
    ("q_literal", ["query", "new", "--json"]),
    ("q_whole_word", ["query", "Result", "--json"]),
    ("q_contains", ["query", "unwr", "--contains", "--json"]),
    ("q_ignore_case", ["query", "result", "-i", "--json"]),
    ("q_ignore_case_contains", ["query", "resul", "-i", "--contains", "--json"]),
    ("q_regex_literal", ["query", r"fn \w+_new", "--regex", "--json"]),
    ("q_regex_no_literal", ["query", r"\w+_?id\b", "--regex", "--json", "--limit", "20"]),
    ("q_regex_icase", ["query", "(?i)config", "--regex", "--json"]),
    ("q_symbols", ["query", "new", "--symbols", "--json"]),
    ("q_kind", ["query", "new", "--kind", "function", "--json"]),
    (
        "q_ast",
        ["query", "(function_item) @fn", "--ast", "--lang", "rust", "--glob", "**/*.rs",
         "--json", "--limit", "30"],
    ),
    ("q_count", ["query", "self", "--count", "--json"]),
    ("q_paths", ["query", "self", "--paths", "--json"]),
    ("q_limit", ["query", "self", "--limit", "5", "--json"]),
    ("q_limit_offset", ["query", "self", "--limit", "5", "--offset", "5", "--json"]),
    ("q_limit_common", ["query", "the", "--limit", "3", "--json"]),
    (
        "q_glob_exclude",
        ["query", "use", "--glob", "**/src/**", "--exclude", "**/tests/**", "--json",
         "--limit", "50"],
    ),
    ("q_lang_text", ["query", "the", "--lang", "text", "--json", "--limit", "50"]),
    ("q_include_locks", ["query", "version", "--include-locks", "--json", "--limit", "50"]),
    ("q_zero", ["query", "zzqqxxnonexistentzz", "--json"]),
    ("q_lock_only", ["query", "checksum", "--json"]),
    ("q_bracket", ["query", "Vec<String>", "--json"]),
    ("q_two_char", ["query", "fn", "--json", "--limit", "20"]),
    ("q_expand", ["query", "new", "--symbols", "--expand", "--json", "--limit", "10"]),
    ("q_context", ["query", "config", "-C", "2", "--json", "--limit", "10"]),
    ("q_dependencies", ["query", "Result", "--dependencies", "--json", "--limit", "10"]),
    ("q_file_filter", ["query", "Default", "--file", "src", "--json"]),
    ("q_text_plain", ["query", "Result", "--limit", "10"]),
    ("q_text_count", ["query", "self", "--count"]),
    ("q_text_symbols", ["query", "new", "--symbols", "--limit", "10"]),
]

ANALYZE = [
    ("a_summary", ["analyze", "--json"]),
    ("a_circular", ["analyze", "--circular", "--json"]),
    ("a_hotspots", ["analyze", "--hotspots", "--json"]),
    ("a_unused", ["analyze", "--unused", "--json"]),
    ("a_islands", ["analyze", "--islands", "--json"]),
    ("a_circular_text", ["analyze", "--circular"]),
    ("a_hotspots_text", ["analyze", "--hotspots"]),
    ("a_summary_text", ["analyze"]),
]

OTHER = [
    ("s_stats", ["stats"]),
    ("s_stats_json", ["stats", "--json"]),
    ("s_list_files", ["list-files", "--json"]),
    ("s_context", ["context", "--json"]),
]


class Mcp:
    def __init__(self, bin_path, tree):
        env = dict(os.environ)
        env.pop("RUST_LOG", None)
        self.proc = subprocess.Popen(
            [bin_path, "mcp"],
            cwd=tree,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self.next_id = 1

    def send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params):
        rid = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("rfx mcp closed stdout")
            msg = json.loads(line)
            if msg.get("id") == rid:
                return msg

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def mcp_text(msg):
    """A tools/call result: parse the text content as JSON where possible."""
    result = msg.get("result")
    if result is None:
        return normalize(json.dumps(msg))
    parts = []
    for item in result.get("content", []):
        text = item.get("text", "")
        try:
            parts.append(json.dumps(normalize_json(json.loads(text)), indent=1))
        except ValueError:
            parts.append(normalize_text(text))
    extra = {k: v for k, v in result.items() if k != "content"}
    if extra:
        parts.append(json.dumps(normalize_json(extra), indent=1, sort_keys=True))
    return "\n".join(parts) + "\n"


def pick_targets(bin_path, tree):
    """Files to use for deps queries: the top hotspot and the first file that imports."""
    _, out, _ = rfx(bin_path, tree, ["analyze", "--hotspots", "--json"])
    hot = None
    try:
        data = json.loads(out)
        items = data if isinstance(data, list) else data.get("results") or data.get("hotspots") or []
        if items:
            first = items[0]
            hot = first.get("path") if isinstance(first, dict) else None
    except ValueError:
        pass
    _, out, _ = rfx(bin_path, tree, ["list-files", "--json"])
    files = []
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            data = data.get("files", [])
        files = [f.get("path") for f in data if isinstance(f, dict)]
    except ValueError:
        pass
    code = [f for f in files if f and f.endswith((".rs", ".ts", ".go", ".py", ".js"))]
    first = code[0] if code else (files[0] if files else "README.md")
    return hot or first, first


def run_battery(bin_path, tree, out_dir, do_index):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    def record(name, rc, stdout, stderr):
        body = normalize(stdout)
        if name in UNORDERED_JSON and body.strip():
            body = json.dumps(sort_lists(json.loads(body)), indent=1) + "\n"
        err = normalize_text(stderr.strip())
        # Drop progress-bar and log noise from stderr; keep warnings and errors.
        err_lines = [
            l for l in err.splitlines()
            if l.strip() and ("warn" in l.lower() or "error" in l.lower())
        ]
        if name in UNORDERED_TEXT:
            lines = body.splitlines()
            body = "\n".join(lines[:1] + sorted(lines[1:])) + "\n"
        text = "exit=%d\n%s" % (rc, body)
        if err_lines:
            text += "--- stderr ---\n" + "\n".join(err_lines) + "\n"
        with open(os.path.join(out_dir, name + ".out"), "w") as f:
            f.write(text)
        results.append(name)

    if do_index:
        rc, out, err = rfx(bin_path, tree, ["index"])
        record("i_index", rc, out, err)
    wait_for_symbol_pass(bin_path, tree)

    for name, args in QUERIES + ANALYZE + OTHER:
        rc, out, err = rfx(bin_path, tree, args)
        record(name, rc, out, err)

    hot, first = pick_targets(bin_path, tree)
    for name, args in [
        ("d_deps", ["deps", first, "--json"]),
        ("d_deps_reverse", ["deps", hot, "--reverse", "--json"]),
        ("d_deps_depth", ["deps", first, "--depth", "2", "--json"]),
        ("d_deps_text", ["deps", first]),
        ("d_deps_reverse_text", ["deps", hot, "--reverse"]),
    ]:
        rc, out, err = rfx(bin_path, tree, args)
        record(name, rc, out, err)

    mcp = Mcp(bin_path, tree)
    try:
        init = mcp.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "golden", "version": "1"},
            },
        )
        mcp.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        with open(os.path.join(out_dir, "m_initialize.out"), "w") as f:
            f.write(json.dumps(init, indent=1, sort_keys=True) + "\n")
        tools = mcp.request("tools/list", {})
        with open(os.path.join(out_dir, "m_tools_list.out"), "w") as f:
            f.write(json.dumps(tools, indent=1, sort_keys=True) + "\n")
        calls = [
            ("search_code", {"pattern": "new"}),
            ("search_code_limit", {"pattern": "self", "limit": 5}),
            ("search_regex", {"pattern": r"fn \w+_new"}),
            ("list_locations", {"pattern": "Result"}),
            ("count_occurrences", {"pattern": "self"}),
            ("find_references", {"pattern": "new"}),
            ("search_ast", {"pattern": "(function_item) @fn", "lang": "rust", "glob": ["**/*.rs"]}),
            ("get_dependencies", {"path": first}),
            ("get_dependents", {"path": hot}),
            ("get_transitive_deps", {"path": first}),
            ("find_hotspots", {}),
            ("find_circular", {}),
            ("find_unused", {}),
            ("find_islands", {}),
            ("analyze_summary", {}),
            ("gather_context", {}),
            ("check_index_status", {}),
            ("index_project", {}),
        ]
        for label, args in calls:
            tool = label if label != "search_code_limit" else "search_code"
            msg = mcp.request("tools/call", {"name": tool, "arguments": args})
            text = mcp_text(msg)
            if "m_" + label in UNORDERED_JSON:
                try:
                    text = json.dumps(sort_lists(json.loads(text)), indent=1) + "\n"
                except ValueError:
                    pass
            with open(os.path.join(out_dir, "m_" + label + ".out"), "w") as f:
                f.write(text)
    finally:
        mcp.close()
    return results


def cmd_run(args):
    run_battery(args.rfx, os.path.abspath(args.tree), os.path.abspath(args.out), args.index)


def cmd_diff(args):
    a, b = args.a, args.b
    names = sorted(set(os.listdir(a)) | set(os.listdir(b)))
    bad = 0
    for n in names:
        pa, pb = os.path.join(a, n), os.path.join(b, n)
        if not os.path.exists(pa) or not os.path.exists(pb):
            print("MISSING %s" % n)
            bad += 1
            continue
        if open(pa, "rb").read() != open(pb, "rb").read():
            print("DIFF    %s" % n)
            bad += 1
    print("%d file(s) differ out of %d" % (bad, len(names)))
    sys.exit(1 if bad else 0)


def cmd_sums(args):
    for n in sorted(os.listdir(args.dir)):
        h = hashlib.sha256(open(os.path.join(args.dir, n), "rb").read()).hexdigest()
        print("%s  %s" % (h, n))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rfx", required=True)
    r.add_argument("--tree", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--index", action="store_true")
    r.set_defaults(func=cmd_run)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    d.set_defaults(func=cmd_diff)
    s = sub.add_parser("sums")
    s.add_argument("dir")
    s.set_defaults(func=cmd_sums)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
