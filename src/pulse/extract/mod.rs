//! Extraction: read what the docs builders need from the Reflex index, once.
//!
//! Everything comes from the index (`meta.db`, `content.bin`), never from the working
//! tree, so the site describes exactly what was indexed and every number agrees with
//! `rfx query`.

pub mod roles;

use crate::cache::CacheManager;
use crate::content_store::ContentReader;
use crate::models::Language;
use anyhow::{Context, Result};
pub use roles::FileRole;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Display name of a language: `Rust`, `TypeScript`, `C#`.
pub fn language_name(lang: Language) -> String {
    let id = serde_json::to_value(lang)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".into());
    match id.as_str() {
        "typescript" => "TypeScript".into(),
        "javascript" => "JavaScript".into(),
        "php" => "PHP".into(),
        "csharp" => "C#".into(),
        "cpp" => "C++".into(),
        other => {
            let mut c = other.chars();
            match c.next() {
                Some(f) => f.to_uppercase().chain(c).collect(),
                None => other.to_string(),
            }
        }
    }
}

/// One indexed file.
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: String,
    pub language: Language,
    pub lines: u64,
    pub role: FileRole,
}

/// A repository document read from the index.
#[derive(Debug, Clone)]
pub struct DocFile {
    pub path: String,
    pub content: String,
}

/// The index, as the docs builders see it.
#[derive(Debug, Clone)]
pub struct Corpus {
    pub root: PathBuf,
    /// Sorted by path.
    pub files: Vec<FileInfo>,
    /// Resolved file-level imports as indices into `files` (importer, imported), deduped.
    pub edges: Vec<(usize, usize)>,
    pub readme: Option<DocFile>,
}

impl Corpus {
    pub fn load(cache: &CacheManager) -> Result<Self> {
        let root = cache
            .path()
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let conn = crate::cache::open_meta_db(cache.path().join("meta.db"))
            .context("opening meta.db (run `rfx index` first)")?;

        let mut rows: Vec<(i64, FileInfo)> = conn
            .prepare("SELECT id, path, line_count FROM files")?
            .query_map([], |r| {
                let path: String = r.get(1)?;
                Ok((
                    r.get::<_, i64>(0)?,
                    FileInfo {
                        language: Language::from_path(Path::new(&path)),
                        role: roles::classify(&path),
                        lines: r.get::<_, i64>(2)?.max(0) as u64,
                        path,
                    },
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;
        rows.sort_by(|a, b| a.1.path.cmp(&b.1.path));
        let index_of: std::collections::HashMap<i64, usize> = rows
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i))
            .collect();

        let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut stmt = conn.prepare(
            "SELECT file_id, resolved_file_id FROM file_dependencies
             WHERE resolved_file_id IS NOT NULL",
        )?;
        let pairs = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?;
        for pair in pairs {
            let (from, to) = pair?;
            if let (Some(&a), Some(&b)) = (index_of.get(&from), index_of.get(&to))
                && a != b
            {
                edges.insert((a, b));
            }
        }

        let files: Vec<FileInfo> = rows.into_iter().map(|(_, f)| f).collect();
        let readme = Self::read_readme(cache, &files);
        Ok(Self {
            root,
            files,
            edges: edges.into_iter().collect(),
            readme,
        })
    }

    /// The root README, preferring `README.md`.
    fn read_readme(cache: &CacheManager, files: &[FileInfo]) -> Option<DocFile> {
        let candidates = [
            "README.md",
            "README.mdx",
            "README.markdown",
            "README.rst",
            "README.txt",
            "README",
        ];
        let path = candidates.iter().find_map(|c| {
            files
                .iter()
                .find(|f| f.path.eq_ignore_ascii_case(c))
                .map(|f| f.path.clone())
        })?;
        let reader = ContentReader::open(cache.path().join("content.bin")).ok()?;
        let id = reader.get_file_id_by_path(&path)?;
        let content = reader.get_file_content(id).ok()?.to_string();
        Some(DocFile { path, content })
    }

    /// Files with a given role.
    pub fn with_role(&self, role: FileRole) -> impl Iterator<Item = (usize, &FileInfo)> {
        self.files
            .iter()
            .enumerate()
            .filter(move |(_, f)| f.role == role)
    }

    /// Count of indexed files per role.
    pub fn role_counts(&self) -> std::collections::BTreeMap<String, usize> {
        let mut out = std::collections::BTreeMap::new();
        for f in &self.files {
            *out.entry(f.role.as_str().to_string()).or_insert(0) += 1;
        }
        out
    }
}
