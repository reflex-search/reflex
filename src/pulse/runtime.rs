//! The Node runtime that builds Pulse sites, managed outside the rfx binary.
//!
//! ```text
//! ~/.reflex/pulse/                      ($REFLEX_PULSE_HOME)
//!   runtime/<deps-hash>/
//!     package.json, package-lock.json   the template's pinned dependencies
//!     node_modules/                     installed once per deps hash, shared by all projects
//!     sites/<workspace-key>/            staged Astro project per workspace (template + bundle)
//! ```
//! The staged project sits under the runtime so Node's upward module resolution finds
//! `node_modules` without links. The rfx binary embeds only the template sources
//! (tens of KB); Node and `node_modules` never ship inside it.

use crate::pulse::render::project::ProjectWriter;
use anyhow::{Context, Result, bail};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The embedded site template.
pub mod template {
    include!(concat!(env!("OUT_DIR"), "/pulse_template.rs"));
}

/// Minimum Node version (Astro 7).
pub const MIN_NODE: (u32, u32, u32) = (22, 12, 0);

pub fn pulse_home() -> PathBuf {
    if let Ok(h) = std::env::var("REFLEX_PULSE_HOME") {
        return PathBuf::from(h);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".reflex")
        .join("pulse")
}

#[derive(Debug, Clone)]
pub struct Node {
    pub path: PathBuf,
    pub version: (u32, u32, u32),
}

pub fn parse_node_version(s: &str) -> Option<(u32, u32, u32)> {
    let v = s.trim().trim_start_matches('v');
    let mut it = v.split('.').map(|p| p.parse::<u32>().ok());
    Some((it.next()??, it.next()??, it.next().flatten().unwrap_or(0)))
}

/// `$REFLEX_PULSE_NODE`, else `node` on `PATH`, at [`MIN_NODE`] or newer.
pub fn find_node() -> Result<Node> {
    let path = std::env::var("REFLEX_PULSE_NODE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("node"));
    let out = Command::new(&path)
        .arg("--version")
        .output()
        .with_context(|| {
            format!(
                "Node.js {}.{}+ is needed to build the site and `{}` was not found.\n  \
             Install Node (https://nodejs.org), or set REFLEX_PULSE_NODE, or run with --no-build \
             to write the site project without building it.",
                MIN_NODE.0,
                MIN_NODE.1,
                path.display()
            )
        })?;
    let version = parse_node_version(&String::from_utf8_lossy(&out.stdout))
        .context("could not read `node --version`")?;
    if version < MIN_NODE {
        bail!(
            "Node {}.{}.{} is too old to build the site; {}.{}+ is needed (set REFLEX_PULSE_NODE to another node).",
            version.0,
            version.1,
            version.2,
            MIN_NODE.0,
            MIN_NODE.1
        );
    }
    Ok(Node { path, version })
}

/// An installed runtime: `node_modules` for the embedded template.
#[derive(Debug, Clone)]
pub struct Runtime {
    pub root: PathBuf,
    pub node: Node,
}

impl Runtime {
    pub fn astro_entry(&self) -> PathBuf {
        self.root.join("node_modules/astro/bin/astro.mjs")
    }

    /// Staging directory for one workspace.
    pub fn site_dir(&self, workspace: &Path) -> PathBuf {
        let canon = std::fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
        let key = blake3::hash(canon.to_string_lossy().as_bytes()).to_hex()[..12].to_string();
        let name = canon
            .file_name()
            .map(|n| crate::pulse::model::ids::slugify(&n.to_string_lossy()))
            .unwrap_or_else(|| "site".into());
        self.root.join("sites").join(format!("{name}-{key}"))
    }
}

const COMPLETE: &str = ".pulse-runtime-complete";

/// Find or install the runtime for the embedded template.
///
/// `$REFLEX_PULSE_RUNTIME` points at a directory with `node_modules` (CI caches,
/// air-gapped mirrors). Otherwise the runtime lives under [`pulse_home`], keyed by the
/// template's dependency hash, and is installed once with `npm ci --ignore-scripts`.
pub fn ensure(offline: bool) -> Result<Runtime> {
    let node = find_node()?;
    if let Ok(dir) = std::env::var("REFLEX_PULSE_RUNTIME") {
        let root = PathBuf::from(dir);
        if !root.join("node_modules/astro").exists() {
            bail!(
                "REFLEX_PULSE_RUNTIME={} has no node_modules/astro",
                root.display()
            );
        }
        return Ok(Runtime { root, node });
    }
    let root = pulse_home().join("runtime").join(template::DEPS_HASH);
    if root.join(COMPLETE).exists() {
        return Ok(Runtime { root, node });
    }
    if offline {
        bail!(
            "The site runtime is not installed ({}) and --offline forbids installing it.\n  \
             Run once without --offline, or use --no-build.",
            root.display()
        );
    }
    install(&root, &node)?;
    Ok(Runtime { root, node })
}

fn npm_for(node: &Node) -> PathBuf {
    // Prefer the npm that ships next to this node.
    if let Some(dir) = node.path.parent().filter(|d| !d.as_os_str().is_empty()) {
        for name in ["npm", "npm.cmd"] {
            let p = dir.join(name);
            if p.exists() {
                return p;
            }
        }
    }
    PathBuf::from(if cfg!(windows) { "npm.cmd" } else { "npm" })
}

fn install(root: &Path, node: &Node) -> Result<()> {
    std::fs::create_dir_all(root)?;
    // Another rfx may be installing the same runtime; wait for it.
    let lock = crate::atomic_write::IndexLock::acquire_with_timeout(root, Duration::from_secs(900))
        .context("waiting for another rfx process that is installing the site runtime")?;
    if root.join(COMPLETE).exists() {
        drop(lock);
        return Ok(());
    }
    for (rel, bytes) in template::FILES {
        if *rel == "package.json" || *rel == "package-lock.json" {
            std::fs::write(root.join(rel), bytes)?;
        }
    }
    eprintln!(
        "Installing the Pulse site runtime (Astro + Starlight) into {} — once per template version…",
        root.display()
    );
    let start = Instant::now();
    let status = Command::new(npm_for(node))
        .args([
            "ci",
            "--ignore-scripts",
            "--omit=dev",
            "--no-audit",
            "--no-fund",
            "--loglevel=error",
        ])
        .current_dir(root)
        .status()
        .context("running `npm ci` (npm must be installed alongside node)")?;
    if !status.success() {
        bail!("`npm ci` failed in {} ({status})", root.display());
    }
    std::fs::write(root.join(COMPLETE), template::DEPS_HASH)?;
    eprintln!("  runtime ready in {:.0}s", start.elapsed().as_secs_f64());
    drop(lock);
    Ok(())
}

/// Write the embedded template into `site_dir`; unchanged files keep their mtime.
pub fn stage_template(site_dir: &Path) -> Result<()> {
    let mut w = ProjectWriter::open(site_dir)?;
    for (rel, bytes) in template::FILES {
        if *rel == "package-lock.json" {
            continue; // the runtime owns dependencies; a site has no node_modules
        }
        w.put(rel, bytes)?;
    }
    // Template files are never swept: only bundle files come and go.
    w.finish("\0")?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct BuildOutcome {
    pub dist: PathBuf,
    pub elapsed: Duration,
}

/// Run `astro build` in `site_dir`. Output goes to `build.log`; a failure returns the
/// last lines of it.
pub fn astro_build(rt: &Runtime, site_dir: &Path, verbose: bool) -> Result<BuildOutcome> {
    let start = Instant::now();
    let log_path = site_dir.join("build.log");
    let mut child = Command::new(&rt.node.path)
        .arg(rt.astro_entry())
        .args(["build", "--root"])
        .arg(site_dir)
        .current_dir(site_dir)
        .env("ASTRO_TELEMETRY_DISABLED", "1")
        .env("NODE_ENV", "production")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting astro build")?;
    let stderr = child.stderr.take().expect("piped stderr");
    let err_thread = std::thread::spawn(move || {
        BufReader::new(stderr)
            .lines()
            .map_while(Result::ok)
            .collect::<Vec<_>>()
    });
    let mut log: Vec<String> = Vec::new();
    let stdout = child.stdout.take().expect("piped stdout");
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if verbose {
            eprintln!("  {line}");
        } else if line.contains("Complete!")
            || line.contains("pages built in")
            || line.contains("[build] ") && line.contains("page(s) built")
        {
            eprintln!("  {}", strip_ansi(&line));
        }
        log.push(line);
    }
    let status = child.wait()?;
    log.extend(err_thread.join().unwrap_or_default());
    let _ = std::fs::write(&log_path, log.join("\n"));
    if !status.success() {
        let tail: Vec<String> = log
            .iter()
            .rev()
            .take(40)
            .rev()
            .map(|l| strip_ansi(l))
            .collect();
        bail!(
            "astro build failed ({status}). Last lines of {}:\n{}",
            log_path.display(),
            tail.join("\n")
        );
    }
    Ok(BuildOutcome {
        dist: site_dir.join("dist"),
        elapsed: start.elapsed(),
    })
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for n in chars.by_ref() {
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_versions() {
        assert_eq!(parse_node_version("v24.15.0\n"), Some((24, 15, 0)));
        assert_eq!(parse_node_version("22.12"), Some((22, 12, 0)));
        assert!(parse_node_version("v22.11.0").unwrap() < MIN_NODE);
        assert!(parse_node_version("v22.12.0").unwrap() >= MIN_NODE);
        assert_eq!(parse_node_version("nope"), None);
    }

    #[test]
    fn template_is_embedded() {
        let names: Vec<&str> = template::FILES.iter().map(|(n, _)| *n).collect();
        for want in [
            "astro.config.mjs",
            "package.json",
            "package-lock.json",
            "src/layouts/PulsePage.astro",
        ] {
            assert!(names.contains(&want), "{want} missing from {names:?}");
        }
        assert!(
            !names
                .iter()
                .any(|n| n.starts_with("node_modules/") || n.starts_with("scripts/"))
        );
        assert_eq!(template::DEPS_HASH.len(), 12);
    }

    #[test]
    fn staging_writes_template_without_dependencies() {
        let dir = tempfile::TempDir::new().unwrap();
        stage_template(dir.path()).unwrap();
        assert!(dir.path().join("astro.config.mjs").exists());
        assert!(!dir.path().join("package-lock.json").exists());
        assert!(
            dir.path().join("package.json").exists(),
            "declares \"type\": \"module\""
        );
    }

    #[test]
    fn ansi_is_stripped() {
        assert_eq!(strip_ansi("\u{1b}[32m✓ Complete!\u{1b}[39m"), "✓ Complete!");
    }
}
