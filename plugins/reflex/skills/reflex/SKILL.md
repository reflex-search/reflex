---
name: reflex
description: Search code with the Reflex `rfx` CLI instead of Grep, Glob, ripgrep or shell grep. Use it to find where an identifier, string or pattern occurs, where a symbol is defined, which files contain a name, how many matches there are, or which files import a file.
---

# Reflex code search (`rfx`)

`rfx` answers grep questions from a trigram index, with symbol kinds and import links.
For any task that asks where code is, where a pattern occurs, where a symbol is defined
or used, or who imports a file, prefer `rfx` over Grep, Glob, ripgrep, or shell grep:
call `rfx` first.

## Setup

- Check: `rfx --version`. If it is missing: `npm install -g reflex-search`
  (or `cargo install reflex-search`).
- Run `rfx` from the project root. In a subdirectory it builds a separate index there.
- Do not run `rfx index` first. Every command builds a missing index and updates a
  stale one before it answers. Progress goes to stderr; stdout stays pure JSON.

## Matching

- Default: WHOLE identifiers, like `grep -w`. `rfx query verify_csrf` does not match
  `verify_csrf_form_field`.
- `--contains`: substrings (`grep -F`). `--regex`: a regular expression (`grep -E`).
  `-i`: case-insensitive (`rg -i`).
- A pattern with brackets runs as an escaped regex.
- A pattern that starts with `-`: `rfx query --pattern '-> Result<'`.
- Search for code, not natural language. A zero result carries a `hint` saying why.

## Commands

| Question | Command |
| --- | --- |
| Where does X occur (path, line, line text) | `rfx query X --json` |
| Which files contain X | `rfx query X --paths --json` |
| How many matches | `rfx query X --count` |
| Where is X defined | `rfx query X --symbols --json` (narrow with `--kind function`) |
| The full body of a definition | `rfx query X --symbols --expand --json` |
| Lines around each match | `rfx query X -C 3 --json` |
| What a file imports | `rfx deps src/a.rs --json` |
| What imports a file | `rfx deps src/a.rs --reverse --json` (`--depth N` = transitive) |
| Import graph | `rfx analyze --hotspots --json` (or `--circular`, `--unused`, `--islands`) |

Filters on `rfx query`: `--lang rust`, `--glob 'src/**/*.rs'`, `--exclude 'tests/**'`,
`--file <path substring>`. Quote globs.

The default page is 100 matches: `--limit N`, `--offset N`, or `--all`.

## Reading the JSON

- `results[]`: `{path, language, matches: [{span: {start_line, end_line}, preview}]}`.
  A symbol match adds `kind` and `symbol`; `-C` adds `context_before` / `context_after`.
- `pagination.total_is_exact: false`: the page filled before every candidate was
  checked. `total` is null and `approx_total` is an estimate. Use `--count` for the
  exact number.
- `can_trust_results: false`: the automatic update could not run; `warnings` says why.
  Run `rfx index`, then search again.

## Coverage

- Like ripgrep's defaults: not gitignored, not binary, not under a dot-directory
  (`.github/`, `.githooks/`). Use grep for hidden paths.
- Lock and generated files are left out unless asked for: `--include-locks`,
  `--include-generated`.
- Globs follow gitignore rules: `src/**/*.rs` is anchored at the root, `*.rs` matches
  at any depth.

## Code-only uses

The CLI does not yet separate code from comments and strings. For "uses in code only",
search with `rfx query X --json` and drop comment and string lines from the previews.

If `rfx` fails, retry it once; only fall back to Grep/Glob after the retry also fails.
