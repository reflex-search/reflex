//! Directory structure generation for context

use anyhow::Result;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

/// Common directories to exclude from structure
const EXCLUDED_DIRS: &[&str] = &[
    "target",
    "node_modules",
    "dist",
    "build",
    ".git",
    ".reflex",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "vendor",
    ".next",
    ".nuxt",
    "coverage",
];

/// Generate ASCII tree structure
pub fn generate_tree(root: &Path, max_depth: usize) -> Result<String> {
    let mut output = Vec::new();

    // Show root directory name
    let root_name = root.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(".");
    output.push(format!("{}/", root_name));

    generate_tree_recursive(root, "", max_depth, 0, &mut output)?;

    Ok(output.join("\n"))
}

/// Recursive tree generation
fn generate_tree_recursive(
    dir: &Path,
    prefix: &str,
    max_depth: usize,
    current_depth: usize,
    output: &mut Vec<String>,
) -> Result<()> {
    if current_depth >= max_depth {
        return Ok(());
    }

    // Read directory entries
    let mut entries: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| !should_exclude(e.path().as_path()))
        .collect();

    // Sort: directories first, then files, alphabetically
    entries.sort_by(|a, b| {
        let a_is_dir = a.path().is_dir();
        let b_is_dir = b.path().is_dir();

        match (a_is_dir, b_is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.file_name().cmp(&b.file_name()),
        }
    });

    let entry_count = entries.len();

    for (idx, entry) in entries.iter().enumerate() {
        let is_last = idx == entry_count - 1;
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Determine tree characters
        let connector = if is_last { "└──" } else { "├──" };
        let extension = if is_last { "    " } else { "│   " };

        // Use symlink_metadata to avoid following symlinks when checking is_dir.
        let is_real_dir = fs::symlink_metadata(&path)
            .map(|m| m.is_dir())
            .unwrap_or(false);

        if is_real_dir {
            // Directory: show name with slash and possibly recurse
            let dir_info = get_dir_info(&path);
            output.push(format!("{}{} {}/ {}", prefix, connector, name_str, dir_info));

            // Recurse if not at max depth
            if current_depth + 1 < max_depth {
                let new_prefix = format!("{}{}", prefix, extension);
                generate_tree_recursive(&path, &new_prefix, max_depth, current_depth + 1, output)?;
            }
        } else {
            // File or symlink: show name with metadata
            let file_info = get_file_info(&path);
            output.push(format!("{}{} {} {}", prefix, connector, name_str, file_info));
        }
    }

    Ok(())
}

/// Get directory information (file count, description)
fn get_dir_info(dir: &Path) -> String {
    // Count direct children
    if let Ok(entries) = fs::read_dir(dir) {
        let count = entries
            .filter_map(|e| e.ok())
            .filter(|e| !should_exclude(&e.path()))
            .count();

        if count == 0 {
            return "(empty)".to_string();
        } else if count == 1 {
            return "(1 file)".to_string();
        } else {
            return format!("({} files)", count);
        }
    }

    String::new()
}

/// Get file information (size, line count)
fn get_file_info(file: &Path) -> String {
    // Use symlink_metadata so we see the symlink itself, not its target.
    let Ok(meta) = fs::symlink_metadata(file) else {
        return String::new();
    };

    if meta.file_type().is_symlink() {
        // Show the link target instead of resolving the target's size.
        if let Ok(target) = fs::read_link(file) {
            return format!("→ {}", target.display());
        }
        return "(symlink)".to_string();
    }

    let size = meta.len();

    // Try to count lines for text files (use the real file content).
    if let Ok(content) = fs::read_to_string(file) {
        let lines = content.lines().count();
        if lines > 0 {
            return format!("({} lines)", lines);
        }
    }

    // Fallback to size
    if size < 1024 {
        format!("({} bytes)", size)
    } else if size < 1024 * 1024 {
        format!("({} KB)", size / 1024)
    } else {
        format!("({} MB)", size / (1024 * 1024))
    }
}

/// Check if path should be excluded
fn should_exclude(path: &Path) -> bool {
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        // Check against exclusion list
        if EXCLUDED_DIRS.contains(&name) {
            return true;
        }

        // Exclude hidden files/directories (except .gitignore, etc.)
        if name.starts_with('.') && name.len() > 1 {
            let keep_files = ["gitignore", "gitattributes", "dockerignore", "editorconfig"];
            if !keep_files.iter().any(|f| name == &format!(".{}", f)) {
                return true;
            }
        }
    }

    false
}

/// Generate JSON tree structure
pub fn generate_tree_json(root: &Path, max_depth: usize) -> Result<Value> {
    let root_name = root.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(".");

    Ok(json!({
        "root": root_name,
        "tree": generate_tree_json_recursive(root, max_depth, 0)?
    }))
}

/// Recursive JSON tree generation
fn generate_tree_json_recursive(
    dir: &Path,
    max_depth: usize,
    current_depth: usize,
) -> Result<Value> {
    if current_depth >= max_depth {
        return Ok(json!({}));
    }

    let mut entries: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| !should_exclude(&e.path()))
        .collect();

    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let mut tree = serde_json::Map::new();
    let mut files = Vec::new();
    let mut subdirs = Vec::new();

    for entry in entries {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        let is_real_dir = fs::symlink_metadata(&path)
            .map(|m| m.is_dir())
            .unwrap_or(false);

        if is_real_dir {
            if current_depth + 1 < max_depth {
                let subtree = generate_tree_json_recursive(&path, max_depth, current_depth + 1)?;
                tree.insert(name.clone(), subtree);
            }
            subdirs.push(name);
        } else {
            let is_symlink = fs::symlink_metadata(&path)
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false);
            files.push(json!({
                "name": name,
                "size": if is_symlink { None } else { fs::metadata(&path).ok().map(|m| m.len()) },
                "lines": if is_symlink { None } else { count_lines(&path).ok() },
                "symlink_target": if is_symlink { fs::read_link(&path).ok().map(|t| t.display().to_string()) } else { None },
            }));
        }
    }

    Ok(json!({
        "type": "directory",
        "files": files,
        "subdirectories": subdirs,
        "children": tree,
    }))
}

/// Count lines in a text file
fn count_lines(path: &Path) -> Result<usize> {
    let content = fs::read_to_string(path)?;
    Ok(content.lines().count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_generate_tree_empty_dir() {
        let temp = TempDir::new().unwrap();
        let result = generate_tree(temp.path(), 3).unwrap();

        // Should show directory name
        assert!(result.contains(temp.path().file_name().unwrap().to_str().unwrap()));
    }

    #[test]
    fn test_generate_tree_with_files() {
        let temp = TempDir::new().unwrap();

        // Create some files
        File::create(temp.path().join("file1.txt")).unwrap()
            .write_all(b"line1\nline2\nline3").unwrap();
        File::create(temp.path().join("file2.rs")).unwrap()
            .write_all(b"fn main() {}").unwrap();

        let result = generate_tree(temp.path(), 3).unwrap();

        assert!(result.contains("file1.txt"));
        assert!(result.contains("file2.rs"));
        assert!(result.contains("lines"));
    }

    #[test]
    fn test_generate_tree_with_nested_dirs() {
        let temp = TempDir::new().unwrap();

        // Create nested structure
        fs::create_dir(temp.path().join("src")).unwrap();
        fs::create_dir(temp.path().join("src/api")).unwrap();
        File::create(temp.path().join("src/main.rs")).unwrap();
        File::create(temp.path().join("src/api/routes.rs")).unwrap();

        let result = generate_tree(temp.path(), 3).unwrap();

        assert!(result.contains("src/"));
        assert!(result.contains("main.rs"));
        assert!(result.contains("api/"));
        assert!(result.contains("routes.rs"));
    }

    #[test]
    fn test_exclude_build_dirs() {
        let temp = TempDir::new().unwrap();

        // Create build directories that should be excluded
        fs::create_dir(temp.path().join("target")).unwrap();
        fs::create_dir(temp.path().join("node_modules")).unwrap();
        File::create(temp.path().join("target/debug.txt")).unwrap();
        File::create(temp.path().join("file.txt")).unwrap();

        let result = generate_tree(temp.path(), 3).unwrap();

        assert!(!result.contains("target"));
        assert!(!result.contains("node_modules"));
        assert!(!result.contains("debug.txt"));
        assert!(result.contains("file.txt"));
    }

    #[test]
    fn test_depth_limiting() {
        let temp = TempDir::new().unwrap();

        // Create deep nested structure
        fs::create_dir_all(temp.path().join("a/b/c/d")).unwrap();
        File::create(temp.path().join("a/b/c/d/deep.txt")).unwrap();

        // Depth 2 should not show d/
        let result = generate_tree(temp.path(), 2).unwrap();
        assert!(result.contains("a/"));
        assert!(result.contains("b/"));
        assert!(!result.contains("c/"));
        assert!(!result.contains("deep.txt"));
    }

    #[test]
    fn test_generate_tree_json() {
        let temp = TempDir::new().unwrap();

        File::create(temp.path().join("test.txt")).unwrap()
            .write_all(b"hello\nworld").unwrap();
        fs::create_dir(temp.path().join("subdir")).unwrap();

        let result = generate_tree_json(temp.path(), 3).unwrap();

        assert!(result["tree"]["files"].is_array());
        assert!(result["tree"]["subdirectories"].is_array());
    }

    #[test]
    fn test_should_exclude_hidden_files() {
        let temp = TempDir::new().unwrap();
        let hidden = temp.path().join(".hidden");
        let gitignore = temp.path().join(".gitignore");

        assert!(should_exclude(&hidden));
        assert!(!should_exclude(&gitignore)); // Keep .gitignore
    }
}
