//! Go API extraction by walking the syntax tree.
//!
//! Top-level declarations are read in source order. A doc comment is the comment group
//! (`//` lines or a `/* */` block) that ends on the line right above a declaration;
//! a blank line breaks the group. Function bodies are never entered.
//!
//! Go doc comments (go.dev/doc/comment) are not Markdown, so they are converted:
//! indented spans become code blocks (or lists, when they start with a list marker),
//! `# Heading` lines become headings, `[Name]` / `[pkg.Name]` doc links become
//! intra-doc links (`` [`Name`] ``), and a `Deprecated:` paragraph becomes
//! [`ApiItem::deprecated`].

use super::{
    ApiFile, ApiItem, ApiKind, Deprecation, DocComment, Param, SigParts, Visibility, collapse_ws,
    doc,
};
use crate::models::Language;
use anyhow::{Context, Result};
use tree_sitter::Node;

/// Longest signature kept, in characters.
const MAX_SIGNATURE: usize = 600;
/// Longest `const`/`var` value shown in a signature; longer ones become `…`.
const MAX_VALUE: usize = 80;

fn parse(source: &str) -> Result<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&crate::parsers::ParserFactory::get_language_grammar(
            Language::Go,
        )?)
        .context("loading the Go grammar")?;
    parser.parse(source, None).context("parsing Go")
}

pub fn extract(source: &str) -> Result<ApiFile> {
    let tree = parse(source)?;
    let root = tree.root_node();
    let mut file = ApiFile::default();
    let mut group = CommentGroup::default();
    let mut prev_end: Option<usize> = None;
    let mut c = root.walk();
    for child in root.named_children(&mut c) {
        if child.kind() == "comment" {
            // A comment on the line a declaration ends on belongs to that declaration.
            if prev_end != Some(child.start_position().row) {
                group.push(&child, source);
            }
            continue;
        }
        let doc = group.take_for(&child);
        prev_end = Some(child.end_position().row);
        match child.kind() {
            "package_clause" => {
                let mut pc = child.walk();
                file.package = child
                    .named_children(&mut pc)
                    .find(|n| n.kind() == "package_identifier")
                    .map(|n| text(&n, source).to_string());
                file.module_doc = doc.and_then(|d| d.doc);
            }
            "function_declaration" | "method_declaration" => {
                file.items.push(function(&child, source, doc));
            }
            "type_declaration" => type_decl(&child, source, doc, &mut file.items),
            "const_declaration" | "var_declaration" => {
                value_decl(&child, source, doc, &mut file.items)
            }
            _ => {}
        }
    }
    Ok(file)
}

/// An `Example…` function from a `_test.go` file.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    /// The name without the `Example` prefix: `""`, `Foo`, `T_M`, `T_M_suffix`.
    pub name: String,
    /// The function body, unindented, including its `// Output:` comment.
    pub code: String,
    pub doc: Option<String>,
}

/// `Example…` functions of a test file (testable examples, go.dev/blog/examples).
pub fn examples(source: &str) -> Result<Vec<Example>> {
    let tree = parse(source)?;
    let root = tree.root_node();
    let mut out = Vec::new();
    let mut group = CommentGroup::default();
    let mut c = root.walk();
    for child in root.named_children(&mut c) {
        if child.kind() == "comment" {
            group.push(&child, source);
            continue;
        }
        let doc = group.take_for(&child);
        if child.kind() != "function_declaration" {
            continue;
        }
        let name = child
            .child_by_field_name("name")
            .map(|n| text(&n, source))
            .unwrap_or("");
        let Some(rest) = name.strip_prefix("Example") else {
            continue;
        };
        // `Examplefoo` is not an example; `Example_suffix` is the package example.
        if rest.starts_with(|c: char| c.is_lowercase()) {
            continue;
        }
        let no_params = child
            .child_by_field_name("parameters")
            .is_some_and(|p| p.named_child_count() == 0);
        let Some(body) = child.child_by_field_name("body") else {
            continue;
        };
        if !no_params || child.child_by_field_name("result").is_some() {
            continue;
        }
        let inner = text(&body, source);
        let inner = inner
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .unwrap_or(inner);
        let lines: Vec<String> = inner.lines().map(|l| l.trim_end().to_string()).collect();
        let code = unindent(trim_blank(lines)).join("\n");
        out.push(Example {
            name: rest.to_string(),
            code,
            doc: doc.and_then(|d| d.doc).map(|d| d.markdown),
        });
    }
    Ok(out)
}

fn text<'s>(node: &Node, src: &'s str) -> &'s str {
    node.utf8_text(src.as_bytes()).unwrap_or("")
}

fn line(node: &Node) -> u32 {
    node.start_position().row as u32 + 1
}

fn end_line(node: &Node) -> u32 {
    node.end_position().row as u32 + 1
}

/// Exported names start with an uppercase letter.
pub fn is_exported(name: &str) -> bool {
    name.chars().next().is_some_and(char::is_uppercase)
}

fn visibility(name: &str) -> Visibility {
    if is_exported(name) {
        Visibility::Public
    } else {
        Visibility::Private
    }
}

/// Consecutive comments, reset by a blank line.
#[derive(Default)]
struct CommentGroup {
    /// (start row, end row, text lines).
    comments: Vec<(usize, usize, Vec<String>)>,
}

/// A converted doc comment and the deprecation notice found in it.
#[derive(Default)]
struct Doc {
    doc: Option<DocComment>,
    deprecated: Option<Deprecation>,
}

impl CommentGroup {
    fn push(&mut self, node: &Node, src: &str) {
        let (start, end) = (node.start_position().row, node.end_position().row);
        if let Some(&(_, last_end, _)) = self.comments.last()
            && start > last_end + 1
        {
            self.comments.clear();
        }
        self.comments.push((start, end, comment_lines(node, src)));
    }

    /// The group, as the doc of `node`, if it ends on the line right above it.
    fn take_for(&mut self, node: &Node) -> Option<Doc> {
        let comments = std::mem::take(&mut self.comments);
        let &(_, last_end, _) = comments.last()?;
        let row = node.start_position().row;
        if last_end + 1 != row && last_end != row {
            return None;
        }
        let start = comments[0].0;
        let lines: Vec<String> = comments.into_iter().flat_map(|(_, _, l)| l).collect();
        Some(convert(&lines, start as u32 + 1, last_end as u32 + 1))
    }
}

/// The text of one comment with its markers removed; directives are dropped.
fn comment_lines(node: &Node, src: &str) -> Vec<String> {
    let t = text(node, src);
    if let Some(rest) = t.strip_prefix("//") {
        if is_directive(rest) {
            return Vec::new();
        }
        return vec![
            rest.strip_prefix(' ')
                .unwrap_or(rest)
                .trim_end()
                .to_string(),
        ];
    }
    let inner = t
        .strip_prefix("/*")
        .and_then(|s| s.strip_suffix("*/"))
        .unwrap_or(t);
    let mut lines: Vec<String> = inner.lines().map(|l| l.trim_end().to_string()).collect();
    if let Some(first) = lines.first_mut() {
        *first = first.trim_start().to_string();
    }
    strip_block_stars(&mut lines);
    lines
}

/// `/*\n * a\n * b\n */` → `a`, `b`.
fn strip_block_stars(lines: &mut [String]) {
    let nonblank: Vec<&String> = lines.iter().filter(|l| !l.trim().is_empty()).collect();
    if nonblank.len() > 1 && nonblank.iter().all(|l| l.trim_start().starts_with('*')) {
        for l in lines.iter_mut() {
            let t = l.trim_start();
            *l = t
                .strip_prefix("* ")
                .or_else(|| t.strip_prefix('*'))
                .unwrap_or(t)
                .to_string();
        }
    }
}

/// `//go:generate`, `//nolint:x`, `//line`, `//export`: not documentation.
fn is_directive(after_slashes: &str) -> bool {
    let s = after_slashes;
    if s.starts_with("line ") || s.starts_with("extern ") || s.starts_with("export ") {
        return true;
    }
    if s.starts_with(" +build") || s.starts_with("+build") {
        return true;
    }
    let word: String = s
        .chars()
        .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        .collect();
    !word.is_empty()
        && s[word.len()..].starts_with(':')
        && s[word.len() + 1..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
}

fn trim_blank(mut lines: Vec<String>) -> Vec<String> {
    while lines.first().is_some_and(|l| l.trim().is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    lines
}

/// Remove the longest common space/tab prefix of the non-blank lines.
fn unindent(lines: Vec<String>) -> Vec<String> {
    let mut prefix: Option<&str> = None;
    for l in lines.iter().filter(|l| !l.trim().is_empty()) {
        let ws = &l[..l.len() - l.trim_start_matches([' ', '\t']).len()];
        prefix = Some(match prefix {
            None => ws,
            Some(p) => {
                let n = p
                    .bytes()
                    .zip(ws.bytes())
                    .take_while(|(a, b)| a == b)
                    .count();
                &p[..n]
            }
        });
    }
    let n = prefix.map(str::len).unwrap_or(0);
    lines
        .iter()
        .map(|l| {
            if l.trim().is_empty() {
                String::new()
            } else {
                l[n..].to_string()
            }
        })
        .collect()
}

fn is_indented(line: &str) -> bool {
    line.starts_with([' ', '\t'])
}

/// A list marker at the start of `t` (already trimmed): `-`, `*`, `+`, `•`, `1.`, `1)`.
fn list_marker(t: &str) -> Option<(bool, &str)> {
    for m in ["- ", "* ", "+ ", "• "] {
        if let Some(rest) = t.strip_prefix(m) {
            return Some((false, rest));
        }
    }
    let digits = t.chars().take_while(char::is_ascii_digit).count();
    if digits > 0 {
        let rest = &t[digits..];
        if let Some(r) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
            return Some((true, r));
        }
    }
    None
}

/// Convert Go doc comment lines to a [`Doc`].
fn convert(raw: &[String], start_line: u32, end_line: u32) -> Doc {
    let lines = unindent(trim_blank(raw.to_vec()));
    let mut md: Vec<String> = Vec::new();
    let mut links: Vec<String> = Vec::new();
    let mut summary: Option<String> = None;
    let mut deprecated: Option<Deprecation> = None;
    let mut i = 0;
    let blank_before = |md: &mut Vec<String>| {
        if md.last().is_some_and(|l| !l.is_empty()) {
            md.push(String::new());
        }
    };
    while i < lines.len() {
        let l = &lines[i];
        if l.trim().is_empty() {
            i += 1;
            continue;
        }
        if is_indented(l) {
            // A span of indented (or blank) lines: a list or a code block.
            let mut span = Vec::new();
            while i < lines.len() && (lines[i].is_empty() || is_indented(&lines[i])) {
                span.push(lines[i].clone());
                i += 1;
            }
            let span = trim_blank(span);
            blank_before(&mut md);
            if list_marker(span[0].trim_start()).is_some() {
                let mut items: Vec<(bool, String)> = Vec::new();
                for s in &span {
                    let t = s.trim();
                    if t.is_empty() {
                        continue;
                    }
                    match list_marker(t) {
                        Some((numbered, rest)) => items.push((numbered, rest.to_string())),
                        None => {
                            if let Some((_, last)) = items.last_mut() {
                                last.push(' ');
                                last.push_str(t);
                            }
                        }
                    }
                }
                let mut n = 0;
                for (numbered, item) in items {
                    let body = inline(&item, &mut links, true);
                    if numbered {
                        n += 1;
                        md.push(format!("{n}. {body}"));
                    } else {
                        md.push(format!("- {body}"));
                    }
                }
            } else {
                md.push("```go".into());
                md.extend(unindent(span));
                md.push("```".into());
            }
            md.push(String::new());
            continue;
        }
        // A paragraph: unindented lines up to a blank or indented line.
        let mut para = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() && !is_indented(&lines[i]) {
            para.push(lines[i].as_str());
            i += 1;
        }
        if para.len() == 1
            && let Some(h) = para[0].strip_prefix("# ")
            && !h.trim().is_empty()
        {
            blank_before(&mut md);
            md.push(format!("# {}", inline(h.trim(), &mut links, true)));
            md.push(String::new());
            continue;
        }
        if deprecated.is_none()
            && let Some(rest) = para[0].strip_prefix("Deprecated:")
        {
            let mut note = vec![rest.trim()];
            note.extend(para[1..].iter().map(|l| l.trim()));
            let note = note.join(" ").trim().to_string();
            deprecated = Some(Deprecation {
                since: None,
                note: (!note.is_empty()).then_some(note),
            });
            continue;
        }
        if summary.is_none() {
            let joined = para.join(" ");
            summary = Some(inline(first_sentence(&joined), &mut Vec::new(), false));
        }
        blank_before(&mut md);
        for p in &para {
            md.push(escape_line_start(&inline(p, &mut links, true)));
        }
        md.push(String::new());
    }
    let markdown = md.join("\n").trim_matches('\n').to_string();
    let doc = (!markdown.trim().is_empty()).then(|| DocComment {
        summary: summary.unwrap_or_default(),
        sections: doc::sections(&markdown),
        examples: doc::examples(&markdown),
        links,
        markdown,
        start_line,
        end_line,
    });
    Doc { doc, deprecated }
}

/// Go's synopsis rule: the text up to the first period followed by a space that does
/// not end a single capital letter (`U.S. `), or the whole text.
fn first_sentence(s: &str) -> &str {
    let (mut ppp, mut pp, mut p) = (' ', ' ', ' ');
    for (i, q) in s.char_indices() {
        let q = if q.is_whitespace() { ' ' } else { q };
        if q == ' ' && p == '.' && (!pp.is_uppercase() || ppp.is_uppercase()) {
            return &s[..i];
        }
        if p == '。' || p == '．' {
            return &s[..i];
        }
        (ppp, pp, p) = (pp, p, q);
    }
    s
}

/// Keep a paragraph line from turning into a Markdown heading, quote or rule.
fn escape_line_start(line: &str) -> String {
    let t = line.trim_start();
    if t.starts_with('#') || t.starts_with('>') {
        return format!("\\{t}");
    }
    if !t.is_empty() && t.chars().all(|c| c == '=' || c == '-') {
        return format!("\\{t}");
    }
    line.to_string()
}

/// A Go doc link target: `Name`, `Name.Method`, `pkg.Name`, `pkg.Name.Method`.
fn doc_link_target(inner: &str) -> bool {
    let segs: Vec<&str> = inner.split('.').collect();
    let ident = |s: &str| {
        s.chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_')
            && s.chars().all(|c| c.is_alphanumeric() || c == '_')
    };
    if !segs.iter().all(|s| ident(s)) {
        return false;
    }
    match segs.as_slice() {
        [name] => is_exported(name),
        [a, b] => is_exported(a) || is_exported(b),
        [pkg, ty, _] => !is_exported(pkg) && is_exported(ty),
        _ => false,
    }
}

/// Inline text: doc links become `` [`Name`] `` (recorded in `links`) and, when
/// `escape` is set, Markdown emphasis and HTML characters are escaped outside
/// backtick code spans.
fn inline(s: &str, links: &mut Vec<String>, escape: bool) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len() + 8);
    let mut in_code = false;
    let mut i = 0;
    let word = |c: char| c.is_alphanumeric() || c == '_';
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            // Toggle only when a closing backtick exists.
            if in_code || chars[i + 1..].contains(&'`') {
                in_code = !in_code;
            }
            out.push(c);
            i += 1;
            continue;
        }
        if in_code {
            out.push(c);
            i += 1;
            continue;
        }
        if c == '['
            && (i == 0 || !word(chars[i - 1]))
            && let Some(len) = chars[i + 1..].iter().position(|&c| c == ']')
        {
            let inner: String = chars[i + 1..i + 1 + len].iter().collect();
            let after = chars.get(i + 2 + len).copied();
            let ok_after = after.is_none_or(|a| !word(a) && a != '(' && a != '[')
                && !(i == 0 && after == Some(':'));
            let (star, target) = match inner.strip_prefix('*') {
                Some(t) => (true, t.to_string()),
                None => (false, inner.clone()),
            };
            if ok_after {
                if target.contains('/') && doc_link_target(target.rsplit('/').next().unwrap_or(""))
                {
                    // An import-path link (`[net/http.Client]`): shown as code.
                    out.push('`');
                    out.push_str(&inner);
                    out.push('`');
                    i += len + 2;
                    continue;
                }
                if doc_link_target(&target) {
                    if star {
                        out.push_str(if escape { "\\*" } else { "*" });
                    }
                    out.push_str(&format!("[`{target}`]"));
                    if !links.contains(&target) {
                        links.push(target);
                    }
                    i += len + 2;
                    continue;
                }
            }
        }
        if escape {
            match c {
                '*' | '<' => out.push('\\'),
                '\\' if chars.get(i + 1).is_some_and(|n| n.is_ascii_punctuation()) => {
                    out.push('\\');
                }
                '_' => {
                    let prev = i.checked_sub(1).map(|p| chars[p]);
                    let next = chars.get(i + 1).copied();
                    if !prev.is_some_and(char::is_alphanumeric)
                        || !next.is_some_and(char::is_alphanumeric)
                    {
                        out.push('\\');
                    }
                }
                _ => {}
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Declaration text from `node` up to `body` (or the whole node), whitespace-collapsed.
fn signature(node: &Node, src: &str) -> String {
    let end = node
        .child_by_field_name("body")
        .map(|b| b.start_byte())
        .unwrap_or(node.end_byte());
    cap(clean_until(node, src, end))
}

/// `node`'s text, comments removed and whitespace collapsed.
fn clean(node: &Node, src: &str) -> String {
    clean_until(node, src, node.end_byte())
}

/// `node`'s text up to byte `end`, comments removed and whitespace collapsed, so a
/// parameter list annotated line by line reads as a clean one-line signature. Comments
/// are found in the tree, so `//` inside a string or struct tag is kept.
fn clean_until(node: &Node, src: &str, end: usize) -> String {
    let mut comments = Vec::new();
    let mut stack = vec![*node];
    while let Some(n) = stack.pop() {
        let mut c = n.walk();
        for ch in n.children(&mut c) {
            if ch.start_byte() >= end {
                break;
            }
            if ch.kind() == "comment" {
                comments.push((ch.start_byte(), ch.end_byte().min(end)));
            } else if ch.child_count() > 0 {
                stack.push(ch);
            }
        }
    }
    comments.sort_unstable();
    let mut out = String::new();
    let mut pos = node.start_byte();
    for (a, b) in comments {
        if a >= pos {
            out.push_str(&src[pos..a]);
            out.push(' ');
            pos = b;
        }
    }
    out.push_str(&src[pos..end.max(pos)]);
    collapse_ws(out.trim())
}

fn cap(mut s: String) -> String {
    if s.chars().count() > MAX_SIGNATURE {
        s = s.chars().take(MAX_SIGNATURE).collect::<String>() + " …";
    }
    s
}

/// `*List[T]`, `pkg.T`, `T[K, V]` → `List`, `T`, `T`.
pub fn base_type_name(t: &str) -> &str {
    let t = t.trim().trim_start_matches('*').trim();
    let t = t.split('[').next().unwrap_or(t);
    t.rsplit('.').next().unwrap_or(t).trim()
}

fn params(list: &Node, src: &str) -> Vec<Param> {
    let mut out = Vec::new();
    let mut c = list.walk();
    for p in list.named_children(&mut c) {
        let variadic = p.kind() == "variadic_parameter_declaration";
        if p.kind() != "parameter_declaration" && !variadic {
            continue;
        }
        let mut ty = p
            .child_by_field_name("type")
            .map(|n| clean(&n, src))
            .unwrap_or_default();
        if variadic {
            ty = format!("...{ty}");
        }
        let mut nc = p.walk();
        let names: Vec<String> = p
            .children_by_field_name("name", &mut nc)
            .filter(|n| n.kind() == "identifier")
            .map(|n| text(&n, src).to_string())
            .collect();
        if names.is_empty() {
            out.push(Param {
                name: String::new(),
                ty,
            });
            continue;
        }
        for name in names {
            out.push(Param {
                name,
                ty: ty.clone(),
            });
        }
    }
    out
}

fn sig_parts(node: &Node, src: &str) -> SigParts {
    let mut sig = SigParts::default();
    if let Some(tp) = node.child_by_field_name("type_parameters") {
        sig.generics = Some(clean(&tp, src));
    }
    if let Some(r) = node.child_by_field_name("receiver") {
        let t = clean(&r, src);
        sig.receiver = Some(
            t.trim_start_matches('(')
                .trim_end_matches(')')
                .trim()
                .to_string(),
        );
    }
    if let Some(p) = node.child_by_field_name("parameters") {
        sig.params = params(&p, src);
    }
    if let Some(r) = node.child_by_field_name("result") {
        sig.returns = Some(clean(&r, src));
    }
    sig
}

fn new_item(name: String, kind: ApiKind, node: &Node, doc: Option<Doc>) -> ApiItem {
    let doc = doc.unwrap_or_default();
    ApiItem {
        visibility: visibility(&name),
        name,
        kind,
        signature: String::new(),
        sig: None,
        doc: doc.doc,
        attrs: Vec::new(),
        deprecated: doc.deprecated,
        hidden: false,
        test_only: false,
        start_line: line(node),
        end_line: end_line(node),
        trait_impl: None,
        self_type: None,
        members: Vec::new(),
    }
}

fn function(node: &Node, src: &str, doc: Option<Doc>) -> ApiItem {
    let name = node
        .child_by_field_name("name")
        .map(|n| text(&n, src).to_string())
        .unwrap_or_default();
    let is_method = node.kind() == "method_declaration";
    let kind = if is_method {
        ApiKind::Method
    } else {
        ApiKind::Function
    };
    let mut it = new_item(name, kind, node, doc);
    it.signature = signature(node, src);
    let sig = sig_parts(node, src);
    if is_method {
        // The receiver's type: `(c *Client)` → `Client`, `(l *List[T])` → `List`.
        it.self_type = node.child_by_field_name("receiver").and_then(|r| {
            let mut c = r.walk();
            let p = r
                .named_children(&mut c)
                .find(|p| p.kind() == "parameter_declaration")?;
            let ty = p.child_by_field_name("type")?;
            Some(base_type_name(text(&ty, src)).to_string())
        });
    }
    it.sig = Some(sig);
    it
}

/// Whether a `type`/`const`/`var` declaration is parenthesized.
fn is_grouped(node: &Node) -> bool {
    let mut c = node.walk();
    node.children(&mut c)
        .any(|ch| ch.kind() == "(" || ch.kind() == "var_spec_list")
}

/// Specs of a declaration, each with its own doc comment (and trailing comment).
fn specs<'t>(node: &Node<'t>, src: &str, kinds: &[&str]) -> Vec<(Node<'t>, Option<Doc>)> {
    let mut out: Vec<(Node<'t>, Option<Doc>)> = Vec::new();
    let mut group = CommentGroup::default();
    let mut prev_end: Option<usize> = None;
    let mut stack = vec![*node];
    while let Some(n) = stack.pop() {
        let mut c = n.walk();
        let children: Vec<Node<'t>> = n.named_children(&mut c).collect();
        for ch in children {
            if ch.kind() == "var_spec_list" {
                stack.push(ch);
                continue;
            }
            if ch.kind() == "comment" {
                if prev_end == Some(ch.start_position().row) {
                    // Trailing comment: the previous spec's doc when it has none.
                    if let Some((_, d)) = out.last_mut()
                        && d.as_ref().is_none_or(|d| d.doc.is_none())
                    {
                        let lines = comment_lines(&ch, src);
                        let deprecated = d.take().and_then(|d| d.deprecated);
                        let mut converted = convert(&lines, line(&ch), end_line(&ch));
                        converted.deprecated = converted.deprecated.or(deprecated);
                        *d = Some(converted);
                    }
                } else {
                    group.push(&ch, src);
                }
                continue;
            }
            if kinds.contains(&ch.kind()) {
                let d = group.take_for(&ch);
                prev_end = Some(ch.end_position().row);
                out.push((ch, d));
            }
        }
    }
    out
}

fn type_decl(node: &Node, src: &str, decl_doc: Option<Doc>, out: &mut Vec<ApiItem>) {
    let grouped = is_grouped(node);
    let specs = specs(node, src, &["type_spec", "type_alias"]);
    let mut decl_doc = decl_doc;
    for (spec, doc) in &specs {
        let has_own = doc.as_ref().is_some_and(|d| d.doc.is_some());
        // Like go/doc, a type without its own doc uses the declaration's.
        let doc = if has_own || decl_doc.is_none() {
            doc.as_ref().map(clone_doc)
        } else if grouped {
            decl_doc.as_ref().map(clone_doc)
        } else {
            decl_doc.take()
        };
        let span = if grouped { spec } else { node };
        out.push(type_spec(spec, span, src, doc));
    }
}

fn clone_doc(d: &Doc) -> Doc {
    Doc {
        doc: d.doc.clone(),
        deprecated: d.deprecated.clone(),
    }
}

fn type_spec(spec: &Node, span: &Node, src: &str, doc: Option<Doc>) -> ApiItem {
    let name = spec
        .child_by_field_name("name")
        .map(|n| text(&n, src).to_string())
        .unwrap_or_default();
    let tparams = spec
        .child_by_field_name("type_parameters")
        .map(|t| clean(&t, src));
    let ty = spec.child_by_field_name("type");
    let kind = match (spec.kind(), ty.map(|t| t.kind())) {
        ("type_alias", _) => ApiKind::TypeAlias,
        (_, Some("struct_type")) => ApiKind::Struct,
        (_, Some("interface_type")) => ApiKind::Interface,
        // A defined type (`type Kind int`) documents like an alias; its methods attach
        // in the surface.
        _ => ApiKind::TypeAlias,
    };
    let mut it = new_item(name.clone(), kind, span, doc);
    let tp = tparams.clone().unwrap_or_default();
    it.signature = match kind {
        ApiKind::Struct => format!("type {name}{tp} struct"),
        ApiKind::Interface => format!("type {name}{tp} interface"),
        _ => cap(format!("type {}", clean(spec, src))),
    };
    if tparams.is_some() {
        it.sig = Some(SigParts {
            generics: tparams,
            ..SigParts::default()
        });
    }
    if let Some(t) = ty {
        match kind {
            ApiKind::Struct => {
                let mut c = t.walk();
                if let Some(list) = t
                    .named_children(&mut c)
                    .find(|n| n.kind() == "field_declaration_list")
                {
                    it.members = fields(&list, src);
                }
            }
            ApiKind::Interface => it.members = interface_members(&t, src),
            _ => {}
        }
    }
    it
}

fn fields(list: &Node, src: &str) -> Vec<ApiItem> {
    let mut out = Vec::new();
    for (f, doc) in specs(list, src, &["field_declaration"]) {
        let ty = f
            .child_by_field_name("type")
            .map(|n| clean(&n, src))
            .unwrap_or_default();
        let tag = f
            .child_by_field_name("tag")
            .map(|n| format!(" {}", text(&n, src)))
            .unwrap_or_default();
        let mut c = f.walk();
        let names: Vec<String> = f
            .children_by_field_name("name", &mut c)
            .map(|n| text(&n, src).to_string())
            .collect();
        if names.is_empty() {
            // Embedded field: named by its type.
            let mut it = new_item(
                base_type_name(&ty).to_string(),
                ApiKind::Field,
                &f,
                doc.as_ref().map(clone_doc),
            );
            it.signature = clean(&f, src);
            it.attrs.push("embedded".into());
            out.push(it);
            continue;
        }
        for name in names {
            let mut it = new_item(
                name.clone(),
                ApiKind::Field,
                &f,
                doc.as_ref().map(clone_doc),
            );
            it.signature = format!("{name} {ty}{tag}");
            out.push(it);
        }
    }
    out
}

fn interface_members(iface: &Node, src: &str) -> Vec<ApiItem> {
    let mut out = Vec::new();
    for (m, doc) in specs(iface, src, &["method_elem", "type_elem"]) {
        if m.kind() == "method_elem" {
            let name = m
                .child_by_field_name("name")
                .map(|n| text(&n, src).to_string())
                .unwrap_or_default();
            let mut it = new_item(name, ApiKind::Method, &m, doc);
            it.signature = clean(&m, src);
            it.sig = Some(sig_parts(&m, src));
            out.push(it);
        } else {
            // An embedded interface or a type-set term: `io.Reader`, `~int | ~string`.
            let t = clean(&m, src);
            let simple = !t.contains(['|', '~']);
            let name = if simple {
                base_type_name(&t).to_string()
            } else {
                t.clone()
            };
            let mut it = new_item(name, ApiKind::Field, &m, doc);
            if !simple {
                it.visibility = Visibility::Inherited;
            }
            it.signature = t;
            it.attrs.push("embedded".into());
            out.push(it);
        }
    }
    out
}

fn value_decl(node: &Node, src: &str, decl_doc: Option<Doc>, out: &mut Vec<ApiItem>) {
    let is_const = node.kind() == "const_declaration";
    let (keyword, kind, spec_kind) = if is_const {
        ("const", ApiKind::Const, "const_spec")
    } else {
        ("var", ApiKind::Static, "var_spec")
    };
    let grouped = is_grouped(node);
    let mut decl_doc = decl_doc;
    // Implicit repetition in const groups: `B` after `A Kind = iota` has type `Kind`.
    let mut last_type: Option<String> = None;
    let mut first_done = false;
    for (spec, doc) in specs(node, src, &[spec_kind]) {
        let ty = spec.child_by_field_name("type").map(|n| clean(&n, src));
        let value = spec.child_by_field_name("value");
        let ty = match (&ty, value) {
            (Some(_), _) => {
                last_type = ty.clone();
                ty
            }
            (None, Some(_)) => {
                last_type = None;
                None
            }
            (None, None) if is_const => last_type.clone(),
            (None, None) => None,
        };
        let values: Vec<String> = value
            .map(|v| {
                let mut c = v.walk();
                v.named_children(&mut c)
                    .filter(|n| n.kind() != "comment")
                    .map(|n| clean(&n, src))
                    .collect()
            })
            .unwrap_or_default();
        let mut c = spec.walk();
        let names: Vec<String> = spec
            .children_by_field_name("name", &mut c)
            .filter(|n| n.kind() == "identifier")
            .map(|n| text(&n, src).to_string())
            .collect();
        let has_own = doc.as_ref().is_some_and(|d| d.doc.is_some());
        // The declaration's doc goes to its first spec, unless that has its own.
        let group_doc = decl_doc.take();
        let mut doc = if has_own || first_done {
            doc
        } else {
            group_doc.or(doc)
        };
        first_done = true;
        let span = if grouped { spec } else { *node };
        for (i, name) in names.iter().enumerate() {
            if name == "_" {
                continue;
            }
            let mut it = new_item(name.clone(), kind, &span, doc.take());
            let mut sig = format!("{keyword} {name}");
            if let Some(t) = &ty {
                sig.push(' ');
                sig.push_str(t);
            }
            let v = if values.len() == names.len() {
                values.get(i).cloned()
            } else if !values.is_empty() {
                Some(values.join(", "))
            } else {
                None
            };
            if let Some(v) = v {
                if v.chars().count() <= MAX_VALUE {
                    sig.push_str(&format!(" = {v}"));
                } else {
                    sig.push_str(" = …");
                }
            }
            it.signature = sig;
            out.push(it);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find<'a>(items: &'a [ApiItem], name: &str) -> &'a ApiItem {
        items.iter().find(|i| i.name == name).unwrap_or_else(|| {
            panic!(
                "{name} not found in {:?}",
                items.iter().map(|i| &i.name).collect::<Vec<_>>()
            )
        })
    }

    const SAMPLE: &str = r#"// Copyright 2024 The Authors.

//go:build linux

// Package client talks to the server.
//
// Start with [NewClient] and [Client.Do].
package client

import (
	"context"
	"io"
)

// Client sends requests.
//
// # Retries
//
// A client retries idempotent requests:
//
//	c := client.NewClient(opts...)
//	c.Do(ctx, req)
//
// Options:
//   - [WithTimeout] sets a timeout
//   - [io.Reader] bodies are streamed,
//     not buffered
//
// See [Option] and [*http.Client]; x[i] is not a link.
type Client struct {
	// BaseURL is the root of every request.
	BaseURL string `json:"base_url" yaml:"baseURL"`
	io.Reader
	*Pool
	a, B int // trailing field comment
	inner bool
}

// Do sends req.
//
// Deprecated: use [Client.Send] instead.
func (c *Client) Do(ctx context.Context, req *Request) (*Response, error) {
	return nil, nil
}

func (l *List[K, V]) Len() int { return 0 }

// Map applies f to every element.
func Map[T, U any](xs []T, f func(T) U, extra ...string) []U {
	return nil
}

// Kind is a request kind.
type Kind int

// Kinds of request.
const (
	// Get reads.
	Get Kind = iota
	Put
	post
	Delete // removes
)

const Single = "one"

var (
	ErrClosed = errors.New("closed")
	Table     = map[string]int{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9}
)

// ReadCloser reads and closes.
type ReadCloser interface {
	io.Reader
	// Close closes.
	Close() error
	close2()
}

// Number is a constraint.
type Number interface {
	~int | ~float64
}

type (
	// ID identifies.
	ID string
	Alias = ID
)

/*
Handler handles.
It is a block comment.
*/
type Handler func(w io.Writer) error
"#;

    #[test]
    fn package_doc_and_name() {
        let f = extract(SAMPLE).unwrap();
        assert_eq!(f.package.as_deref(), Some("client"));
        let d = f.module_doc.as_ref().unwrap();
        assert_eq!(d.summary, "Package client talks to the server.");
        assert!(
            d.markdown
                .contains("Start with [`NewClient`] and [`Client.Do`]."),
            "{}",
            d.markdown
        );
        assert_eq!(d.links, vec!["NewClient", "Client.Do"]);
        assert!(!d.markdown.contains("Copyright"));
        assert!(!d.markdown.contains("go:build"));
    }

    #[test]
    fn doc_comment_conversion() {
        let f = extract(SAMPLE).unwrap();
        let c = find(&f.items, "Client");
        let d = c.doc.as_ref().unwrap();
        assert_eq!(d.summary, "Client sends requests.");
        let md = &d.markdown;
        assert!(md.contains("\n# Retries\n"), "{md}");
        assert!(
            md.contains("```go\nc := client.NewClient(opts...)\nc.Do(ctx, req)\n```"),
            "{md}"
        );
        assert!(
            md.contains("- [`WithTimeout`] sets a timeout\n- [`io.Reader`] bodies are streamed, not buffered"),
            "{md}"
        );
        assert!(
            md.contains("See [`Option`] and \\*[`http.Client`]; x[i] is not a link."),
            "{md}"
        );
        assert_eq!(d.sections[0].0, "retries");
        assert_eq!(d.examples[0].lang, "go");
        assert!(d.links.contains(&"http.Client".to_string()));
    }

    #[test]
    fn struct_fields_tags_and_embedding() {
        let f = extract(SAMPLE).unwrap();
        let c = find(&f.items, "Client");
        assert_eq!(c.kind, ApiKind::Struct);
        assert_eq!(c.signature, "type Client struct");
        assert_eq!(c.visibility, Visibility::Public);
        let names: Vec<&str> = c.members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["BaseURL", "Reader", "Pool", "a", "B", "inner"]);
        let base = &c.members[0];
        assert_eq!(
            base.signature,
            r#"BaseURL string `json:"base_url" yaml:"baseURL"`"#
        );
        assert_eq!(
            base.doc.as_ref().unwrap().summary,
            "BaseURL is the root of every request."
        );
        assert_eq!(c.members[1].signature, "io.Reader");
        assert_eq!(c.members[2].signature, "*Pool");
        assert_eq!(c.members[3].visibility, Visibility::Private);
        assert_eq!(c.members[4].signature, "B int");
        assert_eq!(
            c.members[4].doc.as_ref().unwrap().summary,
            "trailing field comment"
        );
    }

    #[test]
    fn methods_and_generics() {
        let f = extract(SAMPLE).unwrap();
        let d = find(&f.items, "Do");
        assert_eq!(d.kind, ApiKind::Method);
        assert_eq!(d.self_type.as_deref(), Some("Client"));
        assert_eq!(
            d.signature,
            "func (c *Client) Do(ctx context.Context, req *Request) (*Response, error)"
        );
        let sig = d.sig.as_ref().unwrap();
        assert_eq!(sig.receiver.as_deref(), Some("c *Client"));
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[1].name, "req");
        assert_eq!(sig.params[1].ty, "*Request");
        assert_eq!(sig.returns.as_deref(), Some("(*Response, error)"));
        let dep = d.deprecated.as_ref().unwrap();
        assert_eq!(dep.note.as_deref(), Some("use [Client.Send] instead."));
        assert!(!d.doc.as_ref().unwrap().markdown.contains("Deprecated"));

        let len = find(&f.items, "Len");
        assert_eq!(len.self_type.as_deref(), Some("List"));
        assert_eq!(len.signature, "func (l *List[K, V]) Len() int");

        let m = find(&f.items, "Map");
        assert_eq!(m.kind, ApiKind::Function);
        let sig = m.sig.as_ref().unwrap();
        assert_eq!(sig.generics.as_deref(), Some("[T, U any]"));
        let p: Vec<(&str, &str)> = sig
            .params
            .iter()
            .map(|p| (p.name.as_str(), p.ty.as_str()))
            .collect();
        assert_eq!(
            p,
            vec![("xs", "[]T"), ("f", "func(T) U"), ("extra", "...string")]
        );
        assert_eq!(sig.returns.as_deref(), Some("[]U"));
    }

    #[test]
    fn grouped_consts_with_iota_and_vars() {
        let f = extract(SAMPLE).unwrap();
        let get = find(&f.items, "Get");
        assert_eq!(get.kind, ApiKind::Const);
        assert_eq!(get.signature, "const Get Kind = iota");
        assert_eq!(get.doc.as_ref().unwrap().summary, "Get reads.");
        let put = find(&f.items, "Put");
        assert_eq!(put.signature, "const Put Kind");
        assert!(put.doc.is_none(), "the group doc goes to one spec only");
        assert_eq!(find(&f.items, "post").visibility, Visibility::Private);
        let del = find(&f.items, "Delete");
        assert_eq!(del.doc.as_ref().unwrap().summary, "removes");
        assert_eq!(
            find(&f.items, "Single").signature,
            r#"const Single = "one""#
        );

        let err = find(&f.items, "ErrClosed");
        assert_eq!(err.kind, ApiKind::Static);
        assert_eq!(err.signature, r#"var ErrClosed = errors.New("closed")"#);
        assert_eq!(find(&f.items, "Table").signature, "var Table = …");
    }

    #[test]
    fn interfaces_and_named_types() {
        let f = extract(SAMPLE).unwrap();
        let rc = find(&f.items, "ReadCloser");
        assert_eq!(rc.kind, ApiKind::Interface);
        assert_eq!(rc.signature, "type ReadCloser interface");
        let names: Vec<(&str, ApiKind)> = rc
            .members
            .iter()
            .map(|m| (m.name.as_str(), m.kind))
            .collect();
        assert_eq!(
            names,
            vec![
                ("Reader", ApiKind::Field),
                ("Close", ApiKind::Method),
                ("close2", ApiKind::Method)
            ]
        );
        assert_eq!(rc.members[0].signature, "io.Reader");
        assert_eq!(rc.members[1].signature, "Close() error");
        assert_eq!(rc.members[1].doc.as_ref().unwrap().summary, "Close closes.");
        assert_eq!(rc.members[2].visibility, Visibility::Private);

        let n = find(&f.items, "Number");
        assert_eq!(n.members[0].name, "~int | ~float64");
        assert_eq!(n.members[0].visibility, Visibility::Inherited);

        let kind = find(&f.items, "Kind");
        assert_eq!(kind.kind, ApiKind::TypeAlias);
        assert_eq!(kind.signature, "type Kind int");
        let id = find(&f.items, "ID");
        assert_eq!(id.signature, "type ID string");
        assert_eq!(id.doc.as_ref().unwrap().summary, "ID identifies.");
        let alias = find(&f.items, "Alias");
        assert_eq!(alias.kind, ApiKind::TypeAlias);
        assert_eq!(alias.signature, "type Alias = ID");
        let h = find(&f.items, "Handler");
        assert_eq!(h.signature, "type Handler func(w io.Writer) error");
        assert_eq!(
            h.doc.as_ref().unwrap().markdown,
            "Handler handles.\nIt is a block comment."
        );
    }

    #[test]
    fn escaping_and_sentences() {
        let mut links = Vec::new();
        assert_eq!(
            inline("returns *T or <nil>, `a*b` stays", &mut links, true),
            "returns \\*T or \\<nil>, `a*b` stays"
        );
        assert_eq!(
            inline("snake_case and _x", &mut links, true),
            "snake_case and \\_x"
        );
        assert_eq!(
            first_sentence("Handles U.S. data. More."),
            "Handles U.S. data."
        );
        assert_eq!(first_sentence("No period"), "No period");
        assert_eq!(escape_line_start("# not heading"), "\\# not heading");
        assert!(is_directive("go:generate stringer"));
        assert!(is_directive("nolint:errcheck"));
        assert!(!is_directive(" Note: this"));
        assert!(!doc_link_target("i"));
        assert!(doc_link_target("pkg.Name.Method"));
    }

    #[test]
    fn testable_examples() {
        let src = "package client_test\n\n// Basic use.\nfunc ExampleClient_Do() {\n\tc := New()\n\tfmt.Println(c.Do())\n\t// Output: ok\n}\n\nfunc Example() {}\n\nfunc Examplelower() {}\n\nfunc ExampleHelper(t int) {}\n";
        let ex = examples(src).unwrap();
        let names: Vec<&str> = ex.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["Client_Do", ""]);
        assert_eq!(ex[0].code, "c := New()\nfmt.Println(c.Do())\n// Output: ok");
        assert_eq!(ex[0].doc.as_deref(), Some("Basic use."));
    }

    #[test]
    fn signatures_drop_comments_and_split_names() {
        let src = "package p\n\n// Copy copies.\n//\n//go:noinline\nfunc Copy(\n\tdst, src []byte, // buffers\n\t/* inline */ n int,\n) (written int, err error) {\n\treturn 0, nil\n}\n\ntype Tagged struct {\n\tURL string `json:\"url\" // not a comment`\n}\n";
        let f = extract(src).unwrap();
        let c = find(&f.items, "Copy");
        assert_eq!(
            c.signature,
            "func Copy( dst, src []byte, n int, ) (written int, err error)"
        );
        assert_eq!(c.doc.as_ref().unwrap().markdown, "Copy copies.");
        let sig = c.sig.as_ref().unwrap();
        let p: Vec<(&str, &str)> = sig
            .params
            .iter()
            .map(|p| (p.name.as_str(), p.ty.as_str()))
            .collect();
        assert_eq!(p, vec![("dst", "[]byte"), ("src", "[]byte"), ("n", "int")]);
        assert_eq!(sig.returns.as_deref(), Some("(written int, err error)"));
        let t = find(&f.items, "Tagged");
        assert_eq!(
            t.members[0].signature,
            "URL string `json:\"url\" // not a comment`"
        );
    }

    /// Extract every non-test `.go` file under `$REFLEX_GO_TREE` (default: a Kubernetes
    /// checkout), skipping `vendor/`. Read-only. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn extract_go_tree() {
        let root = std::env::var("REFLEX_GO_TREE").unwrap_or_else(|_| {
            format!(
                "{}/Code/misc/test/kubernetes",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(rd) = std::fs::read_dir(dir) else {
                return;
            };
            for e in rd.flatten() {
                let p = e.path();
                let name = e.file_name().to_string_lossy().to_string();
                if p.is_dir() {
                    if !name.starts_with('.') && name != "vendor" {
                        walk(&p, out);
                    }
                } else if name.ends_with(".go") && !name.ends_with("_test.go") {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(std::path::Path::new(&root), &mut files);
        let sources: Vec<String> = files
            .iter()
            .filter_map(|f| std::fs::read_to_string(f).ok())
            .collect();
        let start = std::time::Instant::now();
        let (mut items, mut documented, mut public, mut failed) = (0, 0, 0, 0);
        fn count(items: &[ApiItem], n: &mut usize, d: &mut usize, p: &mut usize) {
            for i in items {
                *n += 1;
                if i.doc.is_some() {
                    *d += 1;
                }
                if i.visibility == Visibility::Public {
                    *p += 1;
                }
                count(&i.members, n, d, p);
            }
        }
        for src in &sources {
            match extract(src) {
                Ok(api) => count(&api.items, &mut items, &mut documented, &mut public),
                Err(_) => failed += 1,
            }
        }
        println!(
            "{} files ({failed} failed), {items} items ({documented} documented, {public} exported) in {:?}",
            sources.len(),
            start.elapsed()
        );
    }
}
