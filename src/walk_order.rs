//! Walk order without a walk.
//!
//! The indexer's directory walk visits files in pre-order, each directory's entries
//! in the order `read_dir` returns them (see `Indexer::walk_builder`). Two paths
//! therefore compare by their first differing component: its position in the
//! listing of the directory the two share. The library update path uses this to
//! place a new or rewritten file among the indexed ones by listing only its
//! ancestor directories.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;

/// Compares `/`-separated paths relative to `root` in walk order, listing each
/// directory it needs once.
pub struct WalkOrder<'a> {
    root: &'a Path,
    /// Directory (relative, `""` = root) → position of each entry name; `None`
    /// when the directory cannot be listed.
    listings: HashMap<String, Option<HashMap<String, usize>>>,
}

impl<'a> WalkOrder<'a> {
    pub fn new(root: &'a Path) -> Self {
        Self {
            root,
            listings: HashMap::new(),
        }
    }

    fn position(&mut self, dir: &str, name: &str) -> Option<usize> {
        let root = self.root;
        let listing = self.listings.entry(dir.to_string()).or_insert_with(|| {
            let path = if dir.is_empty() {
                root.to_path_buf()
            } else {
                root.join(dir)
            };
            let entries = std::fs::read_dir(path).ok()?;
            Some(
                entries
                    .filter_map(|e| e.ok())
                    .enumerate()
                    .map(|(i, e)| (e.file_name().to_string_lossy().into_owned(), i))
                    .collect(),
            )
        });
        listing.as_ref()?.get(name).copied()
    }

    /// The walk order of `a` and `b`; `None` when a listing does not hold one of
    /// them (it is gone, or was renamed since the caller saw it).
    pub fn cmp(&mut self, a: &str, b: &str) -> Option<Ordering> {
        let mut dir = String::new();
        let (mut ai, mut bi) = (a.split('/'), b.split('/'));
        loop {
            match (ai.next(), bi.next()) {
                (Some(x), Some(y)) if x == y => {
                    if !dir.is_empty() {
                        dir.push('/');
                    }
                    dir.push_str(x);
                }
                (Some(x), Some(y)) => {
                    let px = self.position(&dir, x)?;
                    let py = self.position(&dir, y)?;
                    return Some(px.cmp(&py));
                }
                // A directory comes before everything under it.
                (None, Some(_)) => return Some(Ordering::Less),
                (Some(_), None) => return Some(Ordering::Greater),
                (None, None) => return Some(Ordering::Equal),
            }
        }
    }

    /// Where `path` goes in `ordered` (walk-ordered): the index of the first entry
    /// after it. `None` when a comparison cannot be made.
    pub fn insert_position<S: AsRef<str>>(&mut self, ordered: &[S], path: &str) -> Option<usize> {
        let (mut lo, mut hi) = (0, ordered.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.cmp(ordered[mid].as_ref(), path)? {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(mid),
            }
        }
        Some(lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Files under `root` in the order a default `ignore` walk visits them.
    fn walked(root: &Path) -> Vec<String> {
        ignore::WalkBuilder::new(root)
            .hidden(false)
            .build()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
            .map(|e| {
                e.path()
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect()
    }

    fn tree() -> tempfile::TempDir {
        let temp = tempfile::TempDir::new().unwrap();
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let dirs = ["", "a", "a/b", "a/b/c", "d", "d/e", "zz", "m/n/o/p"];
        for i in 0..300 {
            let dir = dirs[next() % dirs.len()];
            let name = format!("f{}_{}.rs", next() % 1000, i);
            let path = temp.path().join(dir).join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "x").unwrap();
        }
        temp
    }

    #[test]
    fn sorting_by_the_comparator_gives_the_walk_order() {
        let temp = tree();
        let root = temp.path();
        let expected = walked(root);
        let mut shuffled = expected.clone();
        shuffled.sort(); // any order that is not the walk's
        shuffled.reverse();
        let mut order = WalkOrder::new(root);
        shuffled.sort_by(|a, b| order.cmp(a, b).unwrap());
        assert_eq!(shuffled, expected);
    }

    #[test]
    fn inserting_each_file_finds_its_walk_position() {
        let temp = tree();
        let root = temp.path();
        let expected = walked(root);
        let mut order = WalkOrder::new(root);
        for (k, path) in expected.iter().enumerate().step_by(7) {
            let mut rest = expected.clone();
            rest.remove(k);
            assert_eq!(order.insert_position(&rest, path), Some(k), "{path}");
        }
    }

    #[test]
    fn a_missing_entry_cannot_be_compared() {
        let temp = tree();
        let mut order = WalkOrder::new(temp.path());
        assert_eq!(order.cmp("a/none.rs", "a/other.rs"), None);
        assert_eq!(order.cmp("a/x", "a/x"), Some(Ordering::Equal));
    }
}
