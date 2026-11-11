---
name: reflex
description: Search code with the Reflex `rfx` CLI instead of Grep, Glob, ripgrep or shell grep. Use it to find where an identifier, string or pattern occurs, every use of a name in code only (not in comments or strings), where a symbol is defined, which files contain a name, how many matches there are, or which files import a file.
---

# Reflex code search (`rfx`)

`rfx` answers grep questions from a trigram index, with symbol kinds and import links.
For any task that asks where code is, where a pattern occurs, where a symbol is defined
or used, or who imports a file, prefer `rfx` over Grep, Glob, ripgrep, or shell grep:
call `rfx` first.

## Setup

- Check: `rfx --version`. If it is missing: `npm install -g reflex-search`
  (or `cargo install reflex-search`). The native binary is faster per call (the npm
  `rfx` is a Node wrapper):
  `curl --proto '=https' --tlsv1.2 -LsSf https://github.com/reflex-search/reflex/releases/latest/download/reflex-search-installer.sh | sh`
  installs it into `~/.cargo/bin`.
- Do not run `rfx index` first. Every command builds a missing index and updates a
  stale one before it answers. Notes go to stderr; stdout is only the answer.
- In a subdirectory, `rfx` uses the index of the enclosing project. Paths and globs
  are relative to the project root.

## Which command

To list where something occurs, use `rfx locations X`: path and line only, the
cheapest answer (`src/a.rs:4,128,193`, one line per file). `--preview` gives
`path:line:text` for each matching line.

For uses in code only (no comments or strings), use `rfx refs X`; it drops those lines
exactly, so do not re-check. It prints the definition, then `path:line:text  [in caller]`
per use. When several functions share the name, it lists them all; `--at PATH:LINE`
keeps the uses of one.

| Question | Command |
| --- | --- |
| Where does X occur | `rfx locations X` (add `--preview` for the line text) |
| Every use of X in code, not comments or strings | `rfx refs X` (`--count` for the number) |
| Every use of the X defined at one line (Rust; other items share the name) | `rfx refs X --at src/a.rs:35` (each use names its caller) |
| Where is X defined | `rfx query X --symbols --json` (narrow with `--kind function`) |
| The full body of a definition | `rfx query X --symbols --expand --json` |
| Lines around each match | `rfx query X -C 3 --json` |
| Which files contain X | `rfx query X --paths --json` |
| How many matches | `rfx query X --count` |
| What a file imports | `rfx deps src/a.rs --json` (`--depth N` = transitive) |
| What imports a file | `rfx deps src/a.rs --reverse --json` (direct importers) |
| Import graph | `rfx analyze --hotspots --json` (or `--circular`, `--unused`, `--islands`) |

Filters (all search commands): a path argument as in grep (`rfx locations X src/`), `--lang rust`, `--glob 'src/**/*.rs'`,
`--exclude 'tests/**'`, `--file <path substring>`. Quote globs. `rfx refs` and
`rfx locations` take `--json` for the full answer.

## Matching

- Default: WHOLE identifiers, like `grep -w`. `rfx locations verify_csrf` does not
  match `verify_csrf_form_field`.
- `--contains`: substrings (`grep -F`). `-i`: case-insensitive (`rg -i`).
  `rfx query X --regex`: a regular expression (`grep -E`).
- A pattern with brackets runs as an escaped regex.
- A pattern that starts with `-`: `--pattern '-> Result<'`.
- Search for code, not natural language. A zero result carries a hint saying why.

## Reading `rfx query --json`

- `results[]`: `{path, language, matches: [{span: {start_line, end_line}, preview}]}`.
  A symbol match adds `kind` and `symbol`; `-C` adds `context_before` / `context_after`.
- The default page is 100 matches: `--limit N`, `--offset N`, or `--all`.
- `pagination.total_is_exact: false`: the page filled before every candidate was
  checked. `total` is null and `approx_total` is an estimate. Use `--count` for the
  exact number.
- `can_trust_results: false`: the automatic update could not run; `warnings` says why.
  Run `rfx index`, then search again.

## Coverage

- Like ripgrep's defaults: not gitignored, not binary, not under a dot-directory
  (`.github/`, `.githooks/`). Use grep for hidden paths.
- Docs and config files are searched too; `rfx refs` reads code files only.
- Lock and generated files are left out unless asked for: `--include-locks`,
  `--include-generated`.
- Globs follow gitignore rules: `src/**/*.rs` is anchored at the root, `*.rs` matches
  at any depth.

If `rfx` fails, retry it once; only fall back to Grep/Glob after the retry also fails.
