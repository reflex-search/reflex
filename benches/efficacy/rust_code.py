"""Which bytes of a Rust file are code: not a comment (line, block, doc) and not a
string, raw string, byte string or char literal. The answer key for the session
benchmark's `refs` questions ("uses in code only"); written apart from Reflex's own
filter (`src/line_filter.rs`) so the key does not share its mistakes."""
import re

def code_mask(src: str) -> list[bool]:
    n = len(src); mask = [True] * n; i = 0
    def blank(a, b):
        for k in range(a, min(b, n)): mask[k] = False
    while i < n:
        c = src[i]
        if c == "/" and src.startswith("//", i):
            j = src.find("\n", i); j = n if j < 0 else j
            blank(i, j); i = j; continue
        if c == "/" and src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j): depth += 1; j += 2
                elif src.startswith("*/", j): depth -= 1; j += 2
                else: j += 1
            blank(i, j); i = j; continue
        m = re.match(r'(b?r)(#*)"', src[i:i+260]) if c in "br" and (i == 0 or not (src[i-1].isalnum() or src[i-1] == "_")) else None
        if m:
            end = src.find('"' + m.group(2), i + m.end())
            j = n if end < 0 else end + 1 + len(m.group(2))
            blank(i, j); i = j; continue
        if c == '"' or (c == "b" and src.startswith('b"', i) and (i == 0 or not (src[i-1].isalnum() or src[i-1] == "_"))):
            j = i + (2 if c == "b" else 1)
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            blank(i, j + 1); i = j + 1; continue
        if c == "'":
            m = re.match(r"'(\\(x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'\n])'", src[i:i+16])
            if m:
                blank(i, i + m.end()); i += m.end(); continue
        i += 1
    return mask

def code_lines(src: str, name: str) -> list[int]:
    """1-based lines where `name` occurs as a whole identifier in code."""
    mask = code_mask(src); out = set()
    for m in re.finditer(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", src):
        if mask[m.start()]:
            out.add(src.count("\n", 0, m.start()) + 1)
    return sorted(out)
