//! Explorer: Interactive treemap visualization
//!
//! Generates a nested treemap showing the entire codebase as rectangles
//! proportional to line count, colored by language. Uses D3.js for
//! client-side rendering. No LLM needed.

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::Serialize;
use std::collections::HashMap;

use crate::cache::CacheManager;

/// A node in the treemap hierarchy
#[derive(Debug, Clone, Serialize)]
pub struct TreemapNode {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<TreemapNode>,
}

/// Full explorer data
#[derive(Debug, Clone)]
pub struct ExplorerData {
    pub root: TreemapNode,
    pub language_colors: HashMap<String, String>,
    pub total_files: usize,
    pub total_lines: usize,
}

/// Generate treemap data from the index
pub fn generate_explorer(cache: &CacheManager) -> Result<ExplorerData> {
    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)
        .context("Failed to open meta.db")?;

    // Query all files with line counts and languages
    let mut stmt = conn.prepare(
        "SELECT path, line_count, COALESCE(language, 'other') FROM files ORDER BY path"
    )?;

    let files: Vec<(String, usize, String)> = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, usize>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?.filter_map(|r| r.ok()).collect();

    let total_files = files.len();
    let total_lines: usize = files.iter().map(|(_, lines, _)| lines).sum();

    // Build tree hierarchy from file paths
    let mut root = TreemapNode {
        name: "root".to_string(),
        value: None,
        language: None,
        path: None,
        children: vec![],
    };

    for (path, lines, language) in &files {
        let parts: Vec<&str> = path.split('/').collect();
        insert_into_tree(&mut root, &parts, *lines, language);
    }

    // Collapse single-child directories for cleaner display
    collapse_single_children(&mut root);

    // Build language color map
    let language_colors = build_language_colors(&files);

    Ok(ExplorerData {
        root,
        language_colors,
        total_files,
        total_lines,
    })
}

/// Insert a file into the tree hierarchy
fn insert_into_tree(node: &mut TreemapNode, parts: &[&str], lines: usize, language: &str) {
    if parts.is_empty() { return; }

    if parts.len() == 1 {
        // Leaf node (file)
        node.children.push(TreemapNode {
            name: parts[0].to_string(),
            value: Some(lines),
            language: Some(language.to_string()),
            path: None, // Will be set during serialization
            children: vec![],
        });
        return;
    }

    // Find or create directory node
    let dir_name = parts[0];
    let child = node.children.iter_mut().find(|c| c.name == dir_name && c.value.is_none());

    if let Some(child) = child {
        insert_into_tree(child, &parts[1..], lines, language);
    } else {
        let mut new_dir = TreemapNode {
            name: dir_name.to_string(),
            value: None,
            language: None,
            path: None,
            children: vec![],
        };
        insert_into_tree(&mut new_dir, &parts[1..], lines, language);
        node.children.push(new_dir);
    }
}

/// Collapse directory nodes that have only one child directory
fn collapse_single_children(node: &mut TreemapNode) {
    // Recurse first
    for child in &mut node.children {
        collapse_single_children(child);
    }

    // If this directory has exactly one child that is also a directory, merge them
    if node.children.len() == 1 && node.children[0].value.is_none() && node.name != "root" {
        let child = node.children.remove(0);
        node.name = format!("{}/{}", node.name, child.name);
        node.children = child.children;
    }
}

/// Assign colors to languages (Synthwave palette)
fn build_language_colors(files: &[(String, usize, String)]) -> HashMap<String, String> {
    let palette = [
        "#a78bfa", // soft violet
        "#4ade80", // soft green
        "#f472b6", // soft pink
        "#fbbf24", // warm amber
        "#67e8f9", // soft cyan
        "#fb923c", // soft orange
        "#818cf8", // indigo
        "#f9a8d4", // light pink
        "#86efac", // mint green
        "#c4b5fd", // light violet
    ];

    let mut lang_counts: HashMap<String, usize> = HashMap::new();
    for (_, _, lang) in files {
        *lang_counts.entry(lang.clone()).or_default() += 1;
    }

    let mut sorted: Vec<(String, usize)> = lang_counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    sorted.into_iter()
        .enumerate()
        .map(|(i, (lang, _))| (lang, palette[i % palette.len()].to_string()))
        .collect()
}

/// Generate treemap JSON for the D3.js visualization
pub fn treemap_json(data: &ExplorerData) -> Result<String> {
    serde_json::to_string(&data.root)
        .context("Failed to serialize treemap data")
}

/// Render explorer page markdown with embedded D3.js treemap
pub fn render_explorer_markdown(data: &ExplorerData) -> Result<String> {
    let mut md = String::new();

    md.push_str(&format!(
        "Visual overview of the codebase: **{}** files, **{}** lines of code.\n\n",
        data.total_files, data.total_lines
    ));

    md.push_str("Rectangles are proportional to line count. Colors represent languages. Click to zoom into a directory.\n\n");

    // Language legend
    md.push_str("### Languages\n\n");
    let mut sorted_colors: Vec<(&String, &String)> = data.language_colors.iter().collect();
    sorted_colors.sort_by_key(|(lang, _)| lang.to_lowercase());
    for (lang, color) in &sorted_colors {
        md.push_str(&format!(
            "<span style=\"display:inline-block;width:12px;height:12px;background:{};border-radius:2px;margin-right:4px;\"></span> {}  \n",
            color, lang
        ));
    }
    md.push('\n');

    // Treemap container
    md.push_str("<div id=\"treemap-container\" style=\"width:100%;height:600px;background:var(--bg-surface);border-radius:8px;overflow:hidden;position:relative;\"></div>\n\n");

    // Breadcrumb for navigation
    md.push_str("<div id=\"treemap-breadcrumb\" style=\"padding:8px 0;color:var(--fg-muted);font-size:0.9em;\"></div>\n\n");

    // Embed the treemap JSON and D3.js script
    let json = treemap_json(data)?;
    let colors_json = serde_json::to_string(&data.language_colors).unwrap_or_default();

    md.push_str("<script type=\"module\">\n");
    md.push_str("import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';\n\n");
    md.push_str(&format!("const data = {};\n", json));
    md.push_str(&format!("const colors = {};\n\n", colors_json));
    md.push_str(r#"const container = document.getElementById('treemap-container');
const breadcrumb = document.getElementById('treemap-breadcrumb');
const width = container.clientWidth;
const height = container.clientHeight;

const root = d3.hierarchy(data)
    .sum(d => d.value || 0)
    .sort((a, b) => b.value - a.value);

// Compute layout ONCE on the full tree
d3.treemap()
    .size([width, height])
    .paddingOuter(3)
    .paddingTop(19)
    .paddingInner(1)
    .round(true)(root);

const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', width)
    .attr('height', height)
    .style('font', '10px sans-serif');

let currentRoot = root;

function render(focus) {
    svg.selectAll('*').remove();
    currentRoot = focus;

    // Coordinate-transform zoom: map focus bounds to fill viewport
    const x = d3.scaleLinear().domain([focus.x0, focus.x1]).rangeRound([0, width]);
    const y = d3.scaleLinear().domain([focus.y0, focus.y1]).rangeRound([0, height]);

    // Get all descendants of focus that are directories (have children)
    const groups = focus.descendants().filter(d => d.children && d !== focus);

    // Draw directory group headers
    groups.forEach(group => {
        const gx = x(group.x0), gy = y(group.y0);
        const gw = x(group.x1) - gx;
        const gh = 18;
        if (gw < 20) return;

        svg.append('rect')
            .attr('x', gx).attr('y', gy)
            .attr('width', Math.max(0, gw))
            .attr('height', gh)
            .attr('fill', '#1a1a2e')
            .style('cursor', 'pointer')
            .on('click', () => render(group));

        svg.append('text')
            .attr('x', gx + 4).attr('y', gy + 13)
            .attr('fill', '#a78bfa')
            .attr('font-weight', 700)
            .attr('font-size', '11px')
            .style('cursor', 'pointer')
            .text(() => {
                const maxChars = Math.floor((gw - 8) / 6.5);
                const name = group.data.name;
                return name.length > maxChars ? name.slice(0, maxChars) : name;
            })
            .on('click', () => render(group));
    });

    // Draw leaf file cells
    const leaves = focus.leaves();
    const cell = svg.selectAll('g.leaf')
        .data(leaves)
        .join('g')
        .attr('class', 'leaf')
        .attr('transform', d => `translate(${x(d.x0)},${y(d.y0)})`);

    const cellW = d => Math.max(0, x(d.x1) - x(d.x0));
    const cellH = d => Math.max(0, y(d.y1) - y(d.y0));

    cell.append('rect')
        .attr('width', cellW)
        .attr('height', cellH)
        .attr('fill', d => colors[d.data.language] || '#2a2a4a')
        .attr('opacity', 0.85)
        .attr('rx', 2)
        .style('cursor', 'pointer')
        .on('click', (event, d) => {
            // Click a file: zoom into its parent directory (if not already the focus)
            if (d.parent && d.parent !== focus) {
                render(d.parent);
            } else if (focus.parent) {
                // Already at this level — zoom back out
                render(focus.parent);
            }
        });

    cell.append('title')
        .text(d => `${d.ancestors().reverse().map(d => d.data.name).join('/')}\n${(d.value || 0).toLocaleString()} lines`);

    cell.filter(d => cellW(d) > 40 && cellH(d) > 14)
        .append('text')
        .attr('x', 3)
        .attr('y', 12)
        .attr('fill', '#0d0d0d')
        .attr('font-weight', 600)
        .text(d => {
            const w = cellW(d) - 6;
            const name = d.data.name;
            return name.length * 6 > w ? name.slice(0, Math.floor(w / 6)) : name;
        });

    // Update breadcrumb — all ancestors are clickable to zoom out
    const pathArr = [];
    let node = focus;
    while (node) {
        pathArr.unshift(node);
        node = node.parent;
    }
    breadcrumb.innerHTML = pathArr.map((n, i) => {
        if (i < pathArr.length - 1) {
            return '<a href="javascript:void(0)" style="color:var(--fg-accent);text-decoration:none;">' + n.data.name + '</a>';
        }
        return '<span style="color:var(--fg);font-weight:600;">' + n.data.name + '</span>';
    }).join(' / ');

    const links = breadcrumb.querySelectorAll('a');
    links.forEach((link, i) => {
        link.onclick = (e) => {
            e.preventDefault();
            render(pathArr[i]);
        };
    });
}

render(root);

// Double-click resets to root
container.addEventListener('dblclick', () => render(root));
</script>
"#);

    Ok(md)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_into_tree() {
        let mut root = TreemapNode {
            name: "root".to_string(),
            value: None,
            language: None,
            path: None,
            children: vec![],
        };
        insert_into_tree(&mut root, &["src", "main.rs"], 100, "Rust");
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].name, "src");
        assert_eq!(root.children[0].children.len(), 1);
        assert_eq!(root.children[0].children[0].name, "main.rs");
        assert_eq!(root.children[0].children[0].value, Some(100));
    }

    #[test]
    fn test_collapse_single_children() {
        let mut root = TreemapNode {
            name: "root".to_string(),
            value: None,
            language: None,
            path: None,
            children: vec![TreemapNode {
                name: "src".to_string(),
                value: None,
                language: None,
                path: None,
                children: vec![TreemapNode {
                    name: "lib".to_string(),
                    value: None,
                    language: None,
                    path: None,
                    children: vec![TreemapNode {
                        name: "main.rs".to_string(),
                        value: Some(100),
                        language: Some("Rust".to_string()),
                        path: None,
                        children: vec![],
                    }],
                }],
            }],
        };
        collapse_single_children(&mut root);
        // src -> lib should be collapsed to src/lib
        assert_eq!(root.children[0].name, "src/lib");
    }

    #[test]
    fn test_build_language_colors() {
        let files = vec![
            ("a.rs".to_string(), 100, "Rust".to_string()),
            ("b.py".to_string(), 50, "Python".to_string()),
        ];
        let colors = build_language_colors(&files);
        assert!(colors.contains_key("Rust"));
        assert!(colors.contains_key("Python"));
    }
}
