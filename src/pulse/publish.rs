//! Copy a built site into the output directory, safely.
//!
//! The output directory is replaced wholesale, so it must be one Pulse owns: empty,
//! absent, or marked by `.pulse-site.json` from an earlier run (or `--clean`). The
//! workspace root, the home directory, `/`, and anything holding `.git` or `.reflex`
//! are refused outright.

use anyhow::{Context, Result, bail};
use std::path::Path;

pub const MARKER: &str = ".pulse-site.json";

/// Refuse output directories that must never be replaced.
pub fn check_target(out: &Path, workspace: &Path, clean: bool) -> Result<()> {
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let target = canon(out);
    let danger = [
        Some(canon(workspace)),
        dirs::home_dir().map(|h| canon(&h)),
        Some(std::path::PathBuf::from("/")),
    ];
    if danger.iter().flatten().any(|d| *d == target) {
        bail!(
            "refusing to write the site into {}: pick a subdirectory such as pulse-site/",
            out.display()
        );
    }
    if out.join(".git").exists() || out.join(".reflex").exists() {
        bail!(
            "refusing to replace {}: it contains .git or .reflex",
            out.display()
        );
    }
    let non_empty = std::fs::read_dir(out)
        .map(|mut d| d.next().is_some())
        .unwrap_or(false);
    if non_empty && !out.join(MARKER).exists() && !clean {
        if out.join("config.toml").exists() && out.join("content").exists() {
            bail!(
                "{} holds a site from the old Zola-based Pulse. Pass --clean to replace it \
                 (the new output is plain HTML directly in this directory).",
                out.display()
            );
        }
        bail!(
            "{} is not empty and was not written by rfx pulse; pass --clean to replace it",
            out.display()
        );
    }
    Ok(())
}

fn copy_dir(from: &Path, to: &Path) -> Result<u64> {
    std::fs::create_dir_all(to)?;
    let mut n = 0;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let dest = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            n += copy_dir(&e.path(), &dest)?;
        } else {
            std::fs::copy(e.path(), &dest)
                .with_context(|| format!("copying {}", e.path().display()))?;
            n += 1;
        }
    }
    Ok(n)
}

/// Replace `out` with the contents of `dist`. Returns the number of files.
pub fn publish(
    dist: &Path,
    out: &Path,
    workspace: &Path,
    clean: bool,
    meta: &serde_json::Value,
) -> Result<u64> {
    check_target(out, workspace, clean)?;
    if out.exists() {
        std::fs::remove_dir_all(out).with_context(|| format!("clearing {}", out.display()))?;
    }
    let n = copy_dir(dist, out)?;
    std::fs::write(out.join(".nojekyll"), b"")?;
    std::fs::write(out.join(MARKER), serde_json::to_vec_pretty(meta)?)?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_workspace_and_foreign_dirs() {
        let ws = tempfile::TempDir::new().unwrap();
        assert!(check_target(ws.path(), ws.path(), true).is_err());

        let foreign = ws.path().join("site");
        std::fs::create_dir_all(&foreign).unwrap();
        std::fs::write(foreign.join("index.html"), "x").unwrap();
        assert!(check_target(&foreign, ws.path(), false).is_err());
        assert!(check_target(&foreign, ws.path(), true).is_ok());

        let old = ws.path().join("old");
        std::fs::create_dir_all(old.join("content")).unwrap();
        std::fs::write(old.join("config.toml"), "").unwrap();
        let err = check_target(&old, ws.path(), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Zola"), "{err}");

        let git = ws.path().join("repo");
        std::fs::create_dir_all(git.join(".git")).unwrap();
        assert!(check_target(&git, ws.path(), true).is_err());
    }

    #[test]
    fn publishes_and_marks() {
        let ws = tempfile::TempDir::new().unwrap();
        let dist = ws.path().join("dist");
        std::fs::create_dir_all(dist.join("docs")).unwrap();
        std::fs::write(dist.join("index.html"), "home").unwrap();
        std::fs::write(dist.join("docs/index.html"), "docs").unwrap();
        let out = ws.path().join("pulse-site");
        let n = publish(
            &dist,
            &out,
            ws.path(),
            false,
            &serde_json::json!({"base": "/"}),
        )
        .unwrap();
        assert_eq!(n, 2);
        assert!(out.join(MARKER).exists());
        assert!(out.join(".nojekyll").exists());
        // A second publish replaces a marked directory without --clean.
        std::fs::remove_file(dist.join("docs/index.html")).unwrap();
        publish(&dist, &out, ws.path(), false, &serde_json::json!({})).unwrap();
        assert!(!out.join("docs/index.html").exists());
    }
}
