//! Facts: every number and claim the site states, computed once, with provenance.
//!
//! Pages reference facts by id instead of formatting their own numbers. That is what
//! keeps the home page, a module page and the glossary from disagreeing about how many
//! files `src/` has, and it is the evidence the grounded writer will cite.

pub use super::ids::FactId;
use super::ids::ModuleId;
use super::xref::SourceLoc;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "type", content = "id", rename_all = "kebab-case")]
pub enum Subject {
    Site,
    Module(ModuleId),
    File(String),
}

impl Subject {
    fn key_prefix(&self) -> String {
        match self {
            Subject::Site => "site".into(),
            Subject::Module(m) => format!("module:{m}"),
            Subject::File(p) => format!("file:{p}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "kebab-case")]
pub enum FactValue {
    Count(u64),
    Text(String),
    List(Vec<String>),
}

impl FactValue {
    /// Human rendering: counts get thousands separators.
    pub fn display(&self) -> String {
        match self {
            FactValue::Count(n) => group_thousands(*n),
            FactValue::Text(s) => s.clone(),
            FactValue::List(v) => v.join(", "),
        }
    }
}

pub fn group_thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Where a fact comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Provenance {
    /// Lines of a file.
    Source { loc: SourceLoc },
    /// A query over the Reflex index.
    Index { query: String },
    /// Git history.
    Git { detail: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub id: FactId,
    pub subject: Subject,
    pub key: String,
    pub value: FactValue,
    pub provenance: Vec<Provenance>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FactStore {
    facts: BTreeMap<FactId, Fact>,
}

impl FactStore {
    /// The id a fact about `subject` with `key` has.
    pub fn id_for(subject: &Subject, key: &str) -> FactId {
        FactId(format!("{}:{key}", subject.key_prefix()))
    }

    /// Insert or replace a fact; returns its id.
    pub fn put(
        &mut self,
        subject: Subject,
        key: &str,
        value: FactValue,
        provenance: Provenance,
    ) -> FactId {
        let id = Self::id_for(&subject, key);
        self.facts.insert(
            id.clone(),
            Fact {
                id: id.clone(),
                subject,
                key: key.to_string(),
                value,
                provenance: vec![provenance],
            },
        );
        id
    }

    pub fn count(&mut self, subject: Subject, key: &str, n: u64, query: &str) -> FactId {
        self.put(
            subject,
            key,
            FactValue::Count(n),
            Provenance::Index {
                query: query.to_string(),
            },
        )
    }

    pub fn get(&self, id: &FactId) -> Option<&Fact> {
        self.facts.get(id)
    }

    pub fn contains(&self, id: &FactId) -> bool {
        self.facts.contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Fact> {
        self.facts.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_semantic() {
        let s = Subject::Module(ModuleId::new("src/pulse"));
        assert_eq!(
            FactStore::id_for(&s, "files").as_str(),
            "module:src/pulse:files"
        );
        assert_eq!(
            FactStore::id_for(&Subject::Site, "lines").as_str(),
            "site:lines"
        );
    }

    #[test]
    fn put_replaces_and_display_groups() {
        let mut f = FactStore::default();
        let id = f.count(Subject::Site, "lines", 1200, "q");
        f.count(Subject::Site, "lines", 1234567, "q");
        assert_eq!(f.len(), 1);
        assert_eq!(f.get(&id).unwrap().value.display(), "1,234,567");
        assert_eq!(group_thousands(999), "999");
        assert_eq!(group_thousands(1000), "1,000");
    }
}
