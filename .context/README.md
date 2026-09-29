# `.context/` Directory

Planning documents, research notes and decision records that keep context across
development sessions (human and AI). Committed to git.

## Files

| File | What it holds |
| --- | --- |
| `TODO.md` | **Required reading at session start.** Live work only: in-progress projects, open bugs, current policy, open follow-ups, backlog. |
| `BINARY_FORMAT_RESEARCH.md` | Reference for the on-disk formats: `trigrams.bin` V4, `content.bin` V2, the `meta.db` symbol cache. |
| `TRIGRAM_RESEARCH.md` | Trigram indexing design and the resolved design questions. |
| `RUNTIME_SYMBOL_DETECTION.md` | 2025-11 decision to parse symbols at query time; now superseded in part by the background symbol cache. |
| `PERFORMANCE_RESEARCH.md` | Dated benchmark and latency rounds. Only the 2026-09 sections describe the 2.0.x format. |
| `PULSE_RENDERER_SPIKE.md` | Pulse M0a renderer spike (2026-09-26). |
| `INCREMENTAL_INDEX_RESEARCH.md` | How the index is rebuilt today (facts, file:line) and the staged design for incremental updates. |
| `EFFICACY-2.0.3.md` | Efficacy A/B on 2.0.3 (Opus 5.5, Sonnet 5): method, all endpoints, per-task tables, limits. |

## Rules

- **Keep `TODO.md` live.** When work finishes, move its record to CHANGELOG.md (user-visible)
  or delete it (git keeps history). Do not leave finished work as "COMPLETED" sections.
- **Date every measurement.** A number carries the Reflex version and the date it was
  measured. A number from an older format or release is labelled historical, or removed.
- **Research files** (`{TOPIC}_RESEARCH.md`, uppercase): create one for a focused
  investigation. Include version numbers, what was tried, and what did not work.
  When the code changes what a research file describes, update the file or add a dated
  banner that says which parts no longer hold.
- **Code wins.** When a file here and the code disagree, fix the file.

## Workflow for AI assistants

See "Context Management & AI Workflow" in `CLAUDE.md`.

1. Start: read `CLAUDE.md`, then `TODO.md`.
2. During work: update task status in `TODO.md`; write findings into a research file.
3. End: make every status accurate; record blockers and open questions.
