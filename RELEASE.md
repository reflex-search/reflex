# Release Management

Reflex follows **semantic versioning** (SemVer) with a simple manual release workflow powered by cargo-dist.

## Semantic Versioning

Version format: `MAJOR.MINOR.PATCH` (e.g., `0.2.7`)

- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward-compatible functionality)
- **PATCH**: Bug fixes (backward-compatible bug fixes)

**Examples:**
- `0.2.6 → 0.2.7`: Bug fix (PATCH bump)
- `0.2.7 → 0.3.0`: New feature like `--timeout` flag (MINOR bump)
- `0.3.0 → 1.0.0`: Breaking change or stable release (MAJOR bump)

## Creating a Release

**Merging a PR is the release.** `.github/workflows/auto-release.yml` runs on every
push to `main`:

1. If `## [Unreleased]` in `CHANGELOG.md` has entries, it bumps `Cargo.toml`:
   - **minor** when a `feat` commit landed since the last tag,
   - **patch** otherwise.
2. It moves the entries under `## [X.Y.Z] - <date>`, commits
   `chore: release vX.Y.Z` to `main` and pushes the `vX.Y.Z` tag.
3. If `[Unreleased]` is empty, nothing is released. A docs or CI PR adds no entry.

**So, in every PR with a user-visible change:** add its entries under `[Unreleased]`.

**Choose the level yourself** with a label on the PR:
`release:major`, `release:minor`, `release:patch` or `release:skip`.
A major release happens only through `release:major` (or a hand bump).

**Hand bump:** if a PR sets a `Cargo.toml` version that has no tag, that version is
tagged as it is.

**Release without a PR:** Actions → **Auto Release** → **Run workflow**, choose a level.

**Never push tags by hand with `git push --tags`.** It pushes every local tag.

The workflow needs the `RELEASE_TOKEN` secret: a fine-grained token with
`contents: write` on this repository. A tag pushed with the default `GITHUB_TOKEN`
starts no workflow, so the release build would never run.

**That's it!** When the tag is pushed, GitHub Actions automatically:
- Builds binaries for all platforms (Linux, macOS, Windows, ARM, x86_64)
- Extracts raw executables from cargo-dist archives
- Creates a GitHub Release with:
  - Raw binaries (e.g., `rfx-x86_64-unknown-linux-gnu`, `rfx-x86_64-pc-windows-msvc.exe`)
  - Shell and PowerShell installer scripts
  - Release notes: the version's `CHANGELOG.md` section

## What Gets Released

The GitHub Release will contain:

**Binaries (raw executables, no archives):**
- `rfx-aarch64-apple-darwin` - macOS ARM (Apple Silicon)
- `rfx-aarch64-unknown-linux-gnu` - Linux ARM64
- `rfx-x86_64-apple-darwin` - macOS Intel
- `rfx-x86_64-unknown-linux-gnu` - Linux x64 (glibc)
- `rfx-x86_64-unknown-linux-musl` - Linux x64 (static, no libc)
- `rfx-x86_64-pc-windows-msvc.exe` - Windows x64

**Installers:**
- `reflex-installer.sh` - Shell install script (`curl | sh`)
- `reflex-installer.ps1` - PowerShell install script

## Workflow Configuration

Releases are configured in:
- **`.github/workflows/auto-release.yml`** - bumps, dates the changelog and tags on merge (`scripts/release-prep.sh`)
- **`dist-workspace.toml`** - cargo-dist configuration (platforms, installers)
- **`.github/workflows/release.yml`** - GitHub Actions workflow (builds binaries, extracts archives)

**Key settings:**
```toml
# dist-workspace.toml
[dist]
targets = ["aarch64-apple-darwin", "aarch64-unknown-linux-gnu",
           "x86_64-apple-darwin", "x86_64-unknown-linux-gnu",
           "x86_64-unknown-linux-musl", "x86_64-pc-windows-msvc"]
installers = ["shell", "powershell"]
auto-includes = false  # Don't bundle README/CHANGELOG in archives
allow-dirty = ["ci"]   # Allow custom workflow modifications
```

## CHANGELOG.md Format

```markdown
# Changelog

All notable changes to Reflex will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] - 2025-11-03

### Added
- Query timeout support with `--timeout` flag
- HTTP API timeout parameter

### Fixed
- Handle empty files without panicking

## [1.0.0] - 2025-11-01

### Added
- Initial release
- Trigram-based full-text search
- Symbol-aware filtering
- Multi-language support
```
