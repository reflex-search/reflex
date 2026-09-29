//! Build-time schema hash computation for automatic cache invalidation
//!
//! This build script computes a hash of all cache-critical source files at compile time.
//! If any of these files change (schema modifications, data format changes), the hash
//! will change, triggering automatic cache invalidation on next startup.
//!
//! ## How it works:
//! 1. At build time: Hash all cache-critical files and store as CACHE_SCHEMA_HASH env var
//! 2. At runtime: Compare stored hash in meta.db with current CACHE_SCHEMA_HASH
//! 3. On mismatch: Warn user and suggest `rfx index` to rebuild cache
//!
//! ## Cache-critical files:
//! - src/cache.rs: SQLite schema definitions (files, statistics, config tables)
//! - src/content_store.rs: Binary format for content.bin (magic bytes, offsets)
//! - src/trigram.rs: Inverted index format for trigrams.bin (posting lists)
//! - src/indexer.rs: Data extraction and serialization logic
//! - src/symbol_cache.rs: Symbol storage format
//! - src/models.rs: Core data structures (Span, SymbolKind, SearchResult)
//! - src/dependency.rs: Dependency extraction and storage
//! - src/trigram_build.rs: The trigrams.bin writer used by `rfx index`
//! - src/snapshot.rs: manifest.json, planning-size and tombstone files, delta tiers
//! - src/meta_update.rs: how an index run writes `files` rows (ids, walk order)
//!
//! Changes to these files may break compatibility with existing cache files.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Cache-critical source files that affect binary format compatibility
const CACHE_CRITICAL_FILES: &[&str] = &[
    "src/cache.rs",
    "src/content_store.rs",
    "src/trigram.rs",
    "src/trigram_build.rs",
    "src/indexer.rs",
    "src/symbol_cache.rs",
    "src/models.rs",
    "src/dependency.rs",
    "src/snapshot.rs",
    "src/meta_update.rs",
];

/// Code that decides what goes into the dependency, export and symbol rows. A
/// change here does not touch the stores, but every stored row may be stale:
/// `rfx index` then re-extracts every file's imports and clears the symbol cache.
/// Before stable file ids, every rebuild did that anyway.
const EXTRACTION_FILES: &[&str] = &["src/line_filter.rs", "src/dependency_resolve.rs"];
const EXTRACTION_DIRS: &[&str] = &["src/parsers"];

fn main() {
    // Compute schema hash from all cache-critical files
    let schema_hash = compute_schema_hash();

    // Export as environment variable for runtime access
    println!("cargo:rustc-env=CACHE_SCHEMA_HASH={}", schema_hash);

    // Tell cargo to rerun this build script if any cache-critical file changes
    for file in CACHE_CRITICAL_FILES {
        println!("cargo:rerun-if-changed={}", file);
    }

    let extraction_files = extraction_files();
    println!(
        "cargo:rustc-env=EXTRACTION_HASH={}",
        hash_files(&extraction_files)
    );
    for file in &extraction_files {
        println!("cargo:rerun-if-changed={}", file);
    }
    for dir in EXTRACTION_DIRS {
        println!("cargo:rerun-if-changed={}", dir);
    }

    // REF-212: emit the short git SHA so the running binary can report its build
    // provenance at startup. This makes a *stale binary* — a benchmark run against
    // an rfx built before a flag/behaviour change — detectable at a glance, which
    // is the exact failure mode behind B_sc2 silently running Stage 1. Falls back
    // to "unknown" when git is unavailable (e.g. source-tarball builds) so the
    // build never fails on it.
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=REFLEX_GIT_SHA={}", git_sha);
    // Re-run when HEAD or its reflog moves so the SHA tracks new commits/checkouts.
    for p in [".git/HEAD", ".git/logs/HEAD"] {
        if Path::new(p).exists() {
            println!("cargo:rerun-if-changed={}", p);
        }
    }

    embed_pulse_template();

    println!("cargo:warning=Cache schema hash: {}", schema_hash);
}

/// Embed `pulse-template/` (the Astro/Starlight site template `rfx pulse` builds with)
/// as `OUT_DIR/pulse_template.rs`: a file table plus two hashes. `DEPS_HASH` covers
/// package.json + package-lock.json and names the shared `node_modules` runtime;
/// `TEMPLATE_HASH` covers every embedded file.
fn embed_pulse_template() {
    const SKIP_DIRS: &[&str] = &[
        "node_modules",
        "dist",
        ".astro",
        ".astro-cache",
        "bundle",
        "scripts",
        "runtime",
        ".spike",
        "results",
        "fixtures",
    ];
    const SKIP_FILES: &[&str] = &[
        "pulse.config.json",
        ".pulse-files.json",
        "pulse-highlight.css",
        "runtime.lock.json",
    ];
    let root = Path::new("pulse-template");
    let mut files: Vec<(String, std::path::PathBuf)> = Vec::new();
    fn walk(
        dir: &Path,
        root: &Path,
        skip_dirs: &[&str],
        skip_files: &[&str],
        out: &mut Vec<(String, std::path::PathBuf)>,
    ) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                if !skip_dirs.contains(&name.as_str()) {
                    println!("cargo:rerun-if-changed={}", path.display());
                    walk(&path, root, skip_dirs, skip_files, out);
                }
            } else if !skip_files.contains(&name.as_str())
                && !name.ends_with(".tar.zst")
                && !name.ends_with(".tar")
            {
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                println!("cargo:rerun-if-changed={}", path.display());
                out.push((rel, path));
            }
        }
    }
    println!("cargo:rerun-if-changed=pulse-template");
    walk(root, root, SKIP_DIRS, SKIP_FILES, &mut files);
    files.sort();

    let mut all = blake3::Hasher::new();
    // SHA-256, not blake3: `scripts/runtime-key.mjs` recomputes it in CI with Node alone.
    let mut deps = <sha2::Sha256 as sha2::Digest>::new();
    let mut table = String::from("pub static FILES: &[(&str, &[u8])] = &[\n");
    for (rel, path) in &files {
        let bytes = fs::read(path).unwrap_or_default();
        all.update(rel.as_bytes());
        all.update(&bytes);
        if rel == "package.json" || rel == "package-lock.json" {
            sha2::Digest::update(&mut deps, rel.as_bytes());
            sha2::Digest::update(&mut deps, &bytes);
        }
        let abs = fs::canonicalize(path).unwrap_or_else(|_| path.clone());
        table.push_str(&format!(
            "    ({:?}, include_bytes!({:?})),\n",
            rel,
            abs.display().to_string()
        ));
    }
    table.push_str("];\n");
    table.push_str(&format!(
        "pub const TEMPLATE_HASH: &str = {:?};\npub const DEPS_HASH: &str = {:?};\n",
        &all.finalize().to_hex()[..16],
        &sha2::Digest::finalize(deps)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()[..12]
    ));
    // The prebuilt-runtime manifest (published tarballs and their SHA-256s).
    let lock = root.join("runtime.lock.json");
    println!("cargo:rerun-if-changed={}", lock.display());
    let lock_json = fs::read_to_string(&lock).unwrap_or_else(|_| "{}".to_string());
    table.push_str(&format!(
        "pub const RUNTIME_LOCK: &str = {:?};\n",
        lock_json
    ));
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("pulse_template.rs");
    fs::write(out, table).expect("write pulse_template.rs");
}

/// Every `.rs` file of [`EXTRACTION_FILES`] and under [`EXTRACTION_DIRS`], sorted.
fn extraction_files() -> BTreeSet<String> {
    let mut files: BTreeSet<String> = EXTRACTION_FILES.iter().map(|s| s.to_string()).collect();
    let mut dirs: Vec<std::path::PathBuf> = EXTRACTION_DIRS.iter().map(Into::into).collect();
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {}", dir.display(), e))
        {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.insert(path.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    files
}

/// Compute a deterministic hash of all cache-critical source files
fn compute_schema_hash() -> String {
    // Use BTreeSet to ensure deterministic ordering (sorted by file path)
    let files: BTreeSet<String> = CACHE_CRITICAL_FILES.iter().map(|s| s.to_string()).collect();
    hash_files(&files)
}

/// blake3 over (path, content) of each file in sorted order, first 16 hex chars.
fn hash_files(files: &BTreeSet<String>) -> String {
    let mut hasher = blake3::Hasher::new();

    // Hash each file's content in sorted order
    for file_path in files {
        let path = Path::new(file_path);

        if !path.exists() {
            panic!("Cache-critical file not found: {}", file_path);
        }

        // Read file content
        let content =
            fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", file_path, e));

        // Hash: file path (for identity) + file content (for changes)
        hasher.update(file_path.as_bytes());
        hasher.update(&content);
    }

    // Return first 16 hex chars (64 bits) for compactness
    // Full blake3 hash is 256 bits, but 64 bits gives ~0% collision probability
    let hash = hasher.finalize();

    // Convert first 8 bytes to hex string manually (no external hex crate needed)
    hash.as_bytes()[..8]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}
