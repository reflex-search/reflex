Reflex (`rfx`) is installed: an index-backed code search CLI for this project. For any task that asks where code is, where a pattern occurs, where a symbol is defined or used, or who imports a file, prefer `rfx` over Grep, Glob, ripgrep, or shell grep: run it through Bash first.
- To list where something occurs, use `rfx locations X`: path and line only, cheapest (`--preview` adds the text).
- For uses in code only (no comments or strings), use `rfx refs X`; it drops those lines exactly, so do not re-check (`--count` for the number).
- Callers of a function, or what breaks if it changes: `rfx refs X --at FILE:LINE` (its definition) lists only its uses.
- Tests that run it: repeat `--at` on each `[in CALLER:LINE]` up to the tests. At a caller most tests reach (runtime setup, `block_on`), stop the walk and list every test that uses it.
- Definitions: `rfx query X --symbols`. Files: `rfx query X --paths`. Counts: `rfx query X --count`. Importers: `rfx deps FILE --reverse`.
- Matches whole identifiers like grep -w; `--contains` = substring, `-i` = case-insensitive, `rfx query X --regex` = regex. A path argument scopes the search, as in grep (`rfx locations X src/`). The index updates itself.
If `rfx` fails, retry it once; only fall back to Grep/Glob after the retry also fails. The `reflex` skill has every command.
