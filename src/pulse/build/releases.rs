//! The changelog: one page per release, built from git tags, CHANGELOG.md and the API.
//!
//! A release is a semver tag (`v2.0.1`, `2.0.1`); "Unreleased" is everything after the
//! newest tag. Each release page shows, in order:
//! 1. its `CHANGELOG.md` section, verbatim, when one names the version;
//! 2. **API changes**: public items added, removed or with a changed signature between
//!    the previous tag and this one, found by extracting both versions of every changed
//!    source file (read from git in one `git cat-file --batch` process);
//! 3. the commits, grouped by conventional-commit type.

use crate::models::Language;
use crate::parsers::api::{self, ApiItem, Visibility};
use crate::pulse::changelog::ChangelogCommit;
use crate::pulse::extract::roles::{self, FileRole};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

/// Releases shown, newest first (plus Unreleased).
pub const MAX_RELEASES: usize = 12;
/// API changes listed per kind (added, removed, changed) before "and N more".
pub const MAX_API_CHANGES: usize = 100;

#[derive(Debug, Clone)]
pub struct Release {
    /// `None` for Unreleased.
    pub tag: Option<String>,
    /// `2.0.1` or `Unreleased`.
    pub version: String,
    pub date: String,
    pub commits: Vec<ChangelogCommit>,
    /// CHANGELOG.md section: (markdown, first line, last line).
    pub notes: Option<(String, u32, u32)>,
    pub api: ApiDelta,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ApiDelta {
    pub added: Vec<ApiChange>,
    pub removed: Vec<ApiChange>,
    pub changed: Vec<(ApiChange, String)>,
    /// True counts; the lists hold at most [`MAX_API_CHANGES`] each.
    pub added_total: usize,
    pub removed_total: usize,
    pub changed_total: usize,
}

impl ApiDelta {
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    pub fn total(&self) -> usize {
        self.added_total + self.removed_total + self.changed_total
    }

    /// Changes counted but not listed.
    pub fn omitted(&self) -> usize {
        self.total() - self.added.len() - self.removed.len() - self.changed.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApiChange {
    /// `Type::method` or `function`.
    pub name: String,
    pub kind: &'static str,
    pub file: String,
    pub signature: String,
}

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `v1.2.3` / `1.2.3` / `v1.2.3-rc.1` → comparable version, else `None`.
pub fn semver(tag: &str) -> Option<(u64, u64, u64, String)> {
    let t = tag.strip_prefix('v').unwrap_or(tag);
    let (core, pre) = t.split_once('-').unwrap_or((t, ""));
    let mut it = core.split('.');
    let v = (
        it.next()?.parse().ok()?,
        it.next()?.parse().ok()?,
        it.next().unwrap_or("0").parse().ok()?,
    );
    if it.next().is_some() {
        return None;
    }
    Some((v.0, v.1, v.2, pre.to_string()))
}

/// Semver tags with their dates, newest version first.
pub fn tags(root: &Path) -> Vec<(String, String)> {
    let Some(out) = git(
        root,
        &[
            "for-each-ref",
            "--format=%(refname:short)|%(creatordate:short)",
            "refs/tags",
        ],
    ) else {
        return Vec::new();
    };
    let mut tags: Vec<(String, String)> = out
        .lines()
        .filter_map(|l| l.split_once('|'))
        .filter(|(t, _)| semver(t).is_some())
        .map(|(t, d)| (t.to_string(), d.to_string()))
        .collect();
    tags.sort_by(|a, b| {
        let (x, y) = (semver(&a.0).unwrap(), semver(&b.0).unwrap());
        // Newest first; a pre-release sorts before its release.
        (y.0, y.1, y.2, y.3.is_empty()).cmp(&(x.0, x.1, x.2, x.3.is_empty()))
    });
    tags
}

/// Commits in `from..to` (all of `to` when `from` is `None`), newest first.
pub fn commits(root: &Path, from: Option<&str>, to: &str, limit: usize) -> Vec<ChangelogCommit> {
    let range = match from {
        Some(f) => format!("{f}..{to}"),
        None => to.to_string(),
    };
    let fmt = "--format=%H%x1f%an%x1f%at%x1f%s";
    let n = format!("-{limit}");
    let Some(out) = git(root, &["log", &n, fmt, &range]) else {
        return Vec::new();
    };
    out.lines()
        .filter_map(|l| {
            let mut p = l.split('\u{1f}');
            let (hash, author, ts, subject) = (p.next()?, p.next()?, p.next()?, p.next()?);
            let timestamp: i64 = ts.parse().ok()?;
            Some(ChangelogCommit {
                hash: hash.to_string(),
                author: author.to_string(),
                timestamp,
                date: crate::pulse::git_intel::epoch_to_date_string(timestamp),
                subject: subject.to_string(),
                files_changed: Vec::new(),
            })
        })
        .collect()
}

/// CHANGELOG.md sections: (heading, markdown body, first line, last line).
pub fn changelog_sections(content: &str) -> Vec<(String, String, u32, u32)> {
    let mut out = Vec::new();
    let mut cur: Option<(String, Vec<&str>, u32)> = None;
    let lines: Vec<&str> = content.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if let Some(h) = line.strip_prefix("## ") {
            if let Some((head, body, start)) = cur.take() {
                out.push((head, body.join("\n").trim().to_string(), start, i as u32));
            }
            cur = Some((h.trim().to_string(), Vec::new(), i as u32 + 2));
        } else if let Some((_, body, _)) = cur.as_mut() {
            body.push(line);
        }
    }
    if let Some((head, body, start)) = cur {
        out.push((
            head,
            body.join("\n").trim().to_string(),
            start,
            lines.len() as u32,
        ));
    }
    out
}

/// The section whose heading names `version` as a whole token, or the bare
/// `Unreleased` section for `None`.
pub fn section_for(
    sections: &[(String, String, u32, u32)],
    version: Option<&str>,
) -> Option<(String, u32, u32)> {
    sections
        .iter()
        .find(|(head, body, _, _)| {
            !body.is_empty()
                && match version {
                    Some(v) => head
                        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '-'))
                        .any(|tok| tok.trim_start_matches('v') == v),
                    None => {
                        let h = head.to_ascii_lowercase();
                        h.contains("unreleased") && !h.chars().any(|c| c.is_ascii_digit())
                    }
                }
        })
        .map(|(_, body, a, b)| (body.clone(), *a, *b))
}

/// Read many `rev:path` blobs through one `git cat-file --batch` process.
struct BlobReader {
    child: std::process::Child,
}

impl BlobReader {
    fn open(root: &Path) -> Option<Self> {
        let child = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["cat-file", "--batch"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        Some(Self { child })
    }

    fn read(&mut self, rev: &str, path: &str) -> Option<String> {
        let stdin = self.child.stdin.as_mut()?;
        writeln!(stdin, "{rev}:{path}").ok()?;
        stdin.flush().ok()?;
        let stdout = self.child.stdout.as_mut()?;
        let mut reader = BufReader::new(stdout);
        let mut header = String::new();
        reader.read_line(&mut header).ok()?;
        if header.trim_end().ends_with("missing") {
            return None;
        }
        let size: usize = header.split_whitespace().nth(2)?.parse().ok()?;
        let mut buf = vec![0u8; size + 1]; // content + trailing newline
        reader.read_exact(&mut buf).ok()?;
        buf.pop();
        Some(String::from_utf8_lossy(&buf).into_owned())
    }
}

impl Drop for BlobReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Public items of one file version, keyed by `Type::member` / `name`.
fn public_items(lang: Language, source: &str) -> BTreeMap<String, (String, &'static str)> {
    let mut out = BTreeMap::new();
    let Some(file) = api::extract(lang, source) else {
        return out;
    };
    fn walk(
        items: &[ApiItem],
        owner: Option<&str>,
        out: &mut BTreeMap<String, (String, &'static str)>,
    ) {
        for it in items {
            if it.test_only || it.hidden || !matches!(it.visibility, Visibility::Public) {
                continue;
            }
            let owner_name = it
                .self_type
                .as_deref()
                .map(|t| t.split('<').next().unwrap_or(t).trim());
            let name = match owner.or(owner_name) {
                Some(o) => format!("{o}::{}", it.name),
                None => it.name.clone(),
            };
            out.insert(
                name.clone(),
                (
                    crate::parsers::api::collapse_ws(&it.signature),
                    it.kind.label(),
                ),
            );
            if it.kind.is_type() || it.kind == api::ApiKind::Module {
                walk(&it.members, Some(&name), out);
            }
        }
    }
    walk(&file.items, None, &mut out);
    out
}

/// Public API differences between two revisions (`to` may be `HEAD`).
pub fn api_delta(root: &Path, from: &str, to: &str) -> ApiDelta {
    let mut delta = ApiDelta::default();
    let Some(names) = git(root, &["diff", "--name-only", "--no-renames", from, to]) else {
        return delta;
    };
    let Some(mut blobs) = BlobReader::open(root) else {
        return delta;
    };
    for path in names.lines() {
        let lang = Language::from_path(Path::new(path));
        if !api::has_extractor(lang) || roles::classify(path) != FileRole::Source {
            continue;
        }
        let old = blobs
            .read(from, path)
            .map(|s| public_items(lang, &s))
            .unwrap_or_default();
        let new = blobs
            .read(to, path)
            .map(|s| public_items(lang, &s))
            .unwrap_or_default();
        let change = |name: &str, sig: &str, kind| ApiChange {
            name: name.to_string(),
            kind,
            file: path.to_string(),
            signature: sig.to_string(),
        };
        for (name, (sig, kind)) in &new {
            match old.get(name) {
                None => {
                    delta.added_total += 1;
                    if delta.added.len() < MAX_API_CHANGES {
                        delta.added.push(change(name, sig, kind));
                    }
                }
                Some((old_sig, _)) if old_sig != sig => {
                    delta.changed_total += 1;
                    if delta.changed.len() < MAX_API_CHANGES {
                        delta
                            .changed
                            .push((change(name, sig, kind), old_sig.clone()));
                    }
                }
                _ => {}
            }
        }
        for (name, (sig, kind)) in &old {
            if !new.contains_key(name) {
                delta.removed_total += 1;
                if delta.removed.len() < MAX_API_CHANGES {
                    delta.removed.push(change(name, sig, kind));
                }
            }
        }
    }
    delta
}

/// Releases, newest first: Unreleased (if it has commits), then up to
/// [`MAX_RELEASES`] tags.
pub fn collect(root: &Path, changelog_md: Option<&str>) -> Vec<Release> {
    let tags = tags(root);
    let sections = changelog_md.map(changelog_sections).unwrap_or_default();
    let mut out = Vec::new();
    let newest = tags.first().map(|(t, _)| t.clone());
    let unreleased = commits(root, newest.as_deref(), "HEAD", 500);
    if !unreleased.is_empty() || newest.is_none() {
        let api = newest
            .as_deref()
            .map(|t| api_delta(root, t, "HEAD"))
            .unwrap_or_default();
        out.push(Release {
            tag: None,
            version: "Unreleased".into(),
            date: unreleased
                .first()
                .map(|c| c.date.clone())
                .unwrap_or_default(),
            commits: unreleased,
            notes: section_for(&sections, None),
            api,
        });
    }
    for (i, (tag, date)) in tags.iter().take(MAX_RELEASES).enumerate() {
        let prev = tags.get(i + 1).map(|(t, _)| t.as_str());
        let version = tag.strip_prefix('v').unwrap_or(tag).to_string();
        out.push(Release {
            tag: Some(tag.clone()),
            date: date.clone(),
            commits: commits(root, prev, tag, 500),
            notes: section_for(&sections, Some(&version)),
            api: prev.map(|p| api_delta(root, p, tag)).unwrap_or_default(),
            version,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver_parsing_and_order() {
        assert_eq!(semver("v2.0.1"), Some((2, 0, 1, String::new())));
        assert_eq!(semver("1.4"), Some((1, 4, 0, String::new())));
        assert_eq!(semver("v1.2.3-rc.1"), Some((1, 2, 3, "rc.1".into())));
        assert_eq!(semver("latest"), None);
        assert_eq!(semver("pulse-runtime-abc"), None);
    }

    #[test]
    fn changelog_section_matching() {
        let md = "# Changelog\n\n## [Unreleased]\n\n- next\n\n## [Unreleased] - 2.0.0\n\n- two\n\n## [1.7.2] - 2026-09-22\n\n- one-seven-two\n\n## [1.7.20] - x\n\n- not it\n";
        let s = changelog_sections(md);
        assert_eq!(s.len(), 4);
        assert_eq!(section_for(&s, None).unwrap().0, "- next");
        assert_eq!(section_for(&s, Some("2.0.0")).unwrap().0, "- two");
        assert_eq!(section_for(&s, Some("1.7.2")).unwrap().0, "- one-seven-two");
        assert!(section_for(&s, Some("1.7")).is_none());
    }

    #[test]
    fn public_item_keys() {
        let src = "pub struct S;\nimpl S { pub fn a(&self) {} fn hidden(&self) {} }\npub fn f(x: u8) {}\nfn private() {}\n";
        let items = public_items(Language::Rust, src);
        let keys: Vec<&str> = items.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["S", "S::a", "f"]);
        assert_eq!(items["f"].0, "pub fn f(x: u8)");
    }

    #[test]
    fn api_delta_between_commits() {
        let t = tempfile::TempDir::new().unwrap();
        let r = t.path();
        let run = |args: &[&str]| {
            assert!(
                Command::new("git")
                    .arg("-C")
                    .arg(r)
                    .args(args)
                    .output()
                    .unwrap()
                    .status
                    .success(),
                "{args:?}"
            );
        };
        run(&["init", "-q"]);
        run(&["config", "user.email", "t@t"]);
        run(&["config", "user.name", "t"]);
        std::fs::create_dir_all(r.join("src")).unwrap();
        std::fs::write(
            r.join("src/lib.rs"),
            "pub fn keep() {}\npub fn gone() {}\npub fn sig(a: u8) {}\n",
        )
        .unwrap();
        run(&["add", "."]);
        run(&["commit", "-qm", "feat: one"]);
        run(&["tag", "v1.0.0"]);
        std::fs::write(
            r.join("src/lib.rs"),
            "pub fn keep() {}\npub fn sig(a: u16) {}\npub fn new_one() {}\n",
        )
        .unwrap();
        run(&["commit", "-qam", "feat(api)!: two"]);
        run(&["tag", "v1.1.0"]);

        let d = api_delta(r, "v1.0.0", "v1.1.0");
        assert_eq!(
            d.added.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            vec!["new_one"]
        );
        assert_eq!(
            d.removed
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["gone"]
        );
        assert_eq!(d.changed[0].0.signature, "pub fn sig(a: u16)");
        assert_eq!(d.changed[0].1, "pub fn sig(a: u8)");

        let rel = collect(r, None);
        assert_eq!(
            rel.iter().map(|x| x.version.as_str()).collect::<Vec<_>>(),
            vec!["1.1.0", "1.0.0"]
        );
        assert_eq!(rel[0].commits.len(), 1);
        assert_eq!(rel[0].commits[0].subject, "feat(api)!: two");
    }
}
