//! Python docstrings → Markdown.
//!
//! Docstrings are reStructuredText-flavoured and follow one of three conventions. All
//! of them become the Markdown sections the reference renders for Rust:
//!
//! | Convention | Written as | Becomes |
//! | --- | --- | --- |
//! | Google | `Args:` / `Returns:` / `Raises:` + indented body | `# Parameters`, `# Returns`, `# Errors` |
//! | NumPy | `Parameters` + `----------` underline | the same |
//! | Sphinx | `:param x:`, `:type x:`, `:returns:`, `:rtype:`, `:raises E:` | the same, appended |
//!
//! Parameters, attributes and exceptions become lists (`` - `name` (`type`) — desc ``).
//! `>>>` doctest runs become ```` ```python ```` fences and doctest [`CodeExample`]s;
//! `.. code-block::` and `::` literal blocks become fences too. Cross-reference roles
//! (`` :class:`Foo` ``, `` :meth:`A.b` ``, `` :func:`~pkg.f` ``) become intra-doc
//! links (`` [`Foo`] ``) that the reference resolves like rustdoc links. Admonitions
//! (`.. note::`, `.. versionadded::`) become block quotes; `.. deprecated:: 1.2 msg`
//! becomes a [`Deprecation`] and leaves the text. Docstrings already written in
//! Markdown pass through: fenced code is copied untouched.

use super::{CodeExample, Deprecation, DocComment, doc};
use regex::Regex;
use std::sync::LazyLock;

/// A converted docstring, plus the structured parts the extractor reuses.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PyDoc {
    pub doc: Option<DocComment>,
    /// From `.. deprecated::`.
    pub deprecated: Option<Deprecation>,
    /// `Args` / `Parameters` / `:param:` entries: fill parameter types that are not
    /// annotated.
    pub params: Vec<Entry>,
    /// `Attributes` / `:ivar:` entries: document fields that have no docstring.
    pub attributes: Vec<Entry>,
}

/// One documented name: a parameter, attribute or exception.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Entry {
    pub name: String,
    pub ty: Option<String>,
    /// Markdown.
    pub desc: String,
}

/// The value of a string literal as written (`"""…"""`, `r'…'`), or `None` for bytes and
/// f-strings, which are never docstrings.
pub fn literal_value(raw: &str) -> Option<String> {
    let at = raw.find(['"', '\''])?;
    let prefix = raw[..at].to_ascii_lowercase();
    if prefix.contains('b') || prefix.contains('f') {
        return None;
    }
    let body = &raw[at..];
    let quote = if body.starts_with("\"\"\"") || body.starts_with("'''") {
        &body[..3]
    } else {
        &body[..1]
    };
    let inner = body.strip_prefix(quote)?;
    let inner = inner.strip_suffix(quote).unwrap_or(inner);
    Some(if prefix.contains('r') {
        inner.to_string()
    } else {
        unescape(inner)
    })
}

/// The escapes that matter in prose: `\\`, quotes, and line continuations.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('\\') | Some('"') | Some('\'') => out.push(chars.next().unwrap_or('\\')),
            Some('\n') => {
                chars.next();
            }
            _ => out.push('\\'),
        }
    }
    out
}

/// `inspect.cleandoc`: strip the first line, remove the common indentation of the rest,
/// and drop leading and trailing blank lines.
pub fn cleandoc(s: &str) -> Vec<String> {
    let s = s.replace('\t', "        ");
    let lines: Vec<&str> = s.lines().collect();
    let common = lines
        .iter()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| indent(l))
        .min()
        .unwrap_or(0);
    let mut out: Vec<String> = lines
        .iter()
        .enumerate()
        .map(|(i, l)| {
            if i == 0 {
                l.trim().to_string()
            } else {
                l.get(common..)
                    .unwrap_or_else(|| l.trim_start())
                    .trim_end()
                    .to_string()
            }
        })
        .collect();
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    let first = out.iter().position(|l| !l.is_empty()).unwrap_or(out.len());
    out.drain(..first);
    out
}

/// Convert a docstring literal (quotes included) found at `start_line..=end_line`.
pub fn parse(raw_literal: &str, start_line: u32, end_line: u32) -> PyDoc {
    match literal_value(raw_literal) {
        Some(v) => from_text(&v, start_line, end_line),
        None => PyDoc::default(),
    }
}

/// Convert a docstring's value.
pub fn from_text(text: &str, start_line: u32, end_line: u32) -> PyDoc {
    let lines = cleandoc(text);
    let mut st = State::default();
    let md = convert(&lines, &mut st);
    let markdown = tidy(&md);
    let doc = (!markdown.trim().is_empty()).then(|| DocComment {
        summary: doc::summary(&markdown),
        sections: doc::sections(&markdown),
        examples: examples(&markdown),
        links: st.links,
        markdown,
        start_line,
        end_line,
    });
    PyDoc {
        doc,
        deprecated: st.deprecated,
        params: st.params,
        attributes: st.attributes,
    }
}

#[derive(Default)]
struct State {
    links: Vec<String>,
    deprecated: Option<Deprecation>,
    params: Vec<Entry>,
    attributes: Vec<Entry>,
}

impl State {
    fn link(&mut self, target: &str) {
        if !self.links.iter().any(|l| l == target) {
            self.links.push(target.to_string());
        }
    }
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

fn is_fence(t: &str) -> bool {
    t.starts_with("```") || t.starts_with("~~~")
}

/// Remove the common indentation of non-blank lines, and surrounding blank lines.
fn dedent(lines: &[String]) -> Vec<String> {
    let common = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| indent(l))
        .min()
        .unwrap_or(0);
    let mut out: Vec<String> = lines
        .iter()
        .map(|l| l.get(common..).unwrap_or(l.trim_start()).to_string())
        .collect();
    while out.last().is_some_and(|l| l.trim().is_empty()) {
        out.pop();
    }
    let first = out
        .iter()
        .position(|l| !l.trim().is_empty())
        .unwrap_or(out.len());
    out.drain(..first);
    out
}

/// End (exclusive) of the block after `from` whose lines are blank or indented deeper
/// than `base`. Trailing blank lines are left out.
fn block_end(lines: &[String], from: usize, base: usize) -> usize {
    let mut end = from;
    let mut j = from;
    while j < lines.len() {
        let l = &lines[j];
        if l.trim().is_empty() {
            j += 1;
            continue;
        }
        if indent(l) <= base {
            break;
        }
        j += 1;
        end = j;
    }
    end
}

/// Index after the closing fence of the fence opening at `start`.
fn fence_end(lines: &[String], start: usize) -> usize {
    let open = lines[start].trim();
    let marker = if open.starts_with("~~~") {
        "~~~"
    } else {
        "```"
    };
    (start + 1..lines.len())
        .find(|&j| lines[j].trim().starts_with(marker))
        .map(|j| j + 1)
        .unwrap_or(lines.len())
}

/// Collapse blank-line runs and trim.
fn tidy(lines: &[String]) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut in_fence = false;
    for l in lines {
        if is_fence(l.trim()) {
            in_fence = !in_fence;
        }
        if !in_fence && l.trim().is_empty() && out.last().is_none_or(|p| p.trim().is_empty()) {
            continue;
        }
        out.push(l.as_str());
    }
    out.join("\n").trim().to_string()
}

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    /// Title as rendered.
    Params(&'static str),
    Returns,
    Yields,
    Raises,
    Warns,
    Examples,
    Attributes,
    SeeAlso,
    /// A titled prose section (`Notes`, `Warning`, an RST heading).
    Text(String),
}

fn section_kind(title: &str) -> Option<Kind> {
    Some(match title.trim().to_ascii_lowercase().as_str() {
        "args" | "arguments" | "parameters" | "params" | "param" => Kind::Params("Parameters"),
        "keyword args" | "keyword arguments" | "keyword parameters" | "kwargs" => {
            Kind::Params("Keyword arguments")
        }
        "other parameters" | "other params" | "other arguments" => Kind::Params("Other parameters"),
        "returns" | "return" => Kind::Returns,
        "yields" | "yield" => Kind::Yields,
        "raises" | "raise" | "exceptions" | "except" => Kind::Raises,
        "warns" | "warn" => Kind::Warns,
        "examples" | "example" => Kind::Examples,
        "attributes" => Kind::Attributes,
        "see also" => Kind::SeeAlso,
        "note" | "notes" => Kind::Text("Notes".into()),
        "warning" | "warnings" => Kind::Text("Warning".into()),
        "todo" => Kind::Text("Todo".into()),
        "references" => Kind::Text("References".into()),
        "methods" => Kind::Text("Methods".into()),
        _ => return None,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Style {
    Google,
    NumPy,
}

static GOOGLE_HEADER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([A-Za-z][A-Za-z ]{1,24}):$").expect("valid regex"));

/// A `Title` line underlined with `---`/`===` (NumPy sections, RST headings).
fn underlined(lines: &[String], i: usize) -> bool {
    let (Some(title), Some(under)) = (lines.get(i), lines.get(i + 1)) else {
        return false;
    };
    let (t, u) = (title.trim(), under.trim());
    let Some(c) = u.chars().next() else {
        return false;
    };
    !t.is_empty()
        && !t.starts_with(">>>")
        && indent(title) == indent(under)
        && "=-~^*+#".contains(c)
        && u.len() >= 3
        && u.chars().all(|x| x == c)
}

/// Docstring lines → Markdown lines: sections at the top level, prose within.
fn convert(lines: &[String], st: &mut State) -> Vec<String> {
    let mut out = Vec::new();
    let mut prose: Vec<String> = Vec::new();
    let mut sphinx = Sphinx::default();
    let flush = |prose: &mut Vec<String>, out: &mut Vec<String>, st: &mut State| {
        if !prose.is_empty() {
            out.extend(convert_prose(prose, st));
            out.push(String::new());
            prose.clear();
        }
    };
    let mut i = 0;
    while i < lines.len() {
        let line = &lines[i];
        let t = line.trim();
        if is_fence(t) {
            let end = fence_end(lines, i);
            prose.extend(lines[i..end].iter().cloned());
            i = end;
            continue;
        }
        if underlined(lines, i) {
            flush(&mut prose, &mut out, st);
            let kind = section_kind(t).unwrap_or_else(|| Kind::Text(t.to_string()));
            let base = indent(line);
            let mut end = i + 2;
            // The section runs to the next heading, or to a directive at its own level.
            while end < lines.len()
                && !((underlined(lines, end) || lines[end].trim_start().starts_with(".. "))
                    && indent(&lines[end]) <= base)
            {
                end += 1;
            }
            let body = dedent(&lines[i + 2..end]);
            section(&kind, &body, Style::NumPy, st, &mut out);
            i = end;
            continue;
        }
        if let Some(cap) = GOOGLE_HEADER_RE.captures(t)
            && let Some(kind) = section_kind(&cap[1])
        {
            let end = block_end(lines, i + 1, indent(line));
            if end > i + 1 {
                flush(&mut prose, &mut out, st);
                let body = dedent(&lines[i + 1..end]);
                section(&kind, &body, Style::Google, st, &mut out);
                i = end;
                continue;
            }
        }
        if t.starts_with(':')
            && let Some(used) = sphinx.take(lines, i)
        {
            i += used;
            continue;
        }
        prose.push(line.clone());
        i += 1;
    }
    flush(&mut prose, &mut out, st);
    sphinx.emit(st, &mut out);
    out
}

static DIRECTIVE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\.\.\s+([\w:-]+)::\s*(.*)$").expect("valid regex"));

/// Prose lines → Markdown: doctests, literal blocks, directives, inline markup.
fn convert_prose(lines: &[String], st: &mut State) -> Vec<String> {
    let mut out = Vec::new();
    // Indentation of the paragraph that ended with `::`.
    let mut literal_after: Option<usize> = None;
    let mut i = 0;
    while i < lines.len() {
        let line = &lines[i];
        let t = line.trim();
        let ind = indent(line);
        if is_fence(t) {
            let end = fence_end(lines, i);
            out.extend(lines[i..end].iter().cloned());
            i = end;
            continue;
        }
        if t.is_empty() {
            out.push(String::new());
            i += 1;
            continue;
        }
        if let Some(base) = literal_after.take()
            && ind > base
        {
            let end = block_end(lines, i, base);
            let body = dedent(&lines[i..end]);
            let lang = if t.starts_with(">>>") {
                "python"
            } else {
                "text"
            };
            fence(&mut out, lang, &body);
            i = end;
            continue;
        }
        if t.starts_with(">>>") {
            let mut j = i;
            while j < lines.len() && !lines[j].trim().is_empty() && indent(&lines[j]) >= ind {
                j += 1;
            }
            fence(&mut out, "python", &dedent(&lines[i..j]));
            i = j;
            continue;
        }
        if let Some(cap) = DIRECTIVE_RE.captures(t) {
            let end = block_end(lines, i + 1, ind);
            let body = dedent(&lines[i + 1..end]);
            directive(&cap[1], cap[2].trim(), &body, st, &mut out);
            i = end;
            continue;
        }
        if t == ".." || t.starts_with(".. ") {
            // A comment or a link target.
            i = block_end(lines, i + 1, ind);
            continue;
        }
        if let Some(text) = t.strip_suffix("::") {
            literal_after = Some(ind);
            let text = text.trim_end();
            if !text.is_empty() {
                // `Example::` reads `Example:`; `Example ::` reads `Example`.
                let shown = if t.ends_with(" ::") {
                    text.to_string()
                } else {
                    format!("{text}:")
                };
                out.push(format!("{}{}", &line[..ind], inline(&shown, st)));
            }
            i += 1;
            continue;
        }
        out.push(inline(line, st));
        i += 1;
    }
    out
}

fn fence(out: &mut Vec<String>, lang: &str, body: &[String]) {
    if out.last().is_some_and(|l| !l.trim().is_empty()) {
        out.push(String::new());
    }
    out.push(format!("```{lang}"));
    out.extend(body.iter().cloned());
    out.push("```".into());
    out.push(String::new());
}

fn directive(name: &str, arg: &str, body: &[String], st: &mut State, out: &mut Vec<String>) {
    let (first, rest) = arg.split_once(char::is_whitespace).unwrap_or((arg, ""));
    let label = match name {
        "deprecated" => {
            let mut note: Vec<String> = Vec::new();
            if !rest.trim().is_empty() {
                note.push(rest.trim().to_string());
            }
            note.extend(body.iter().map(|l| l.trim().to_string()));
            let note = inline(note.join(" ").trim(), st);
            st.deprecated = Some(Deprecation {
                since: (!first.is_empty()).then(|| first.to_string()),
                note: (!note.is_empty()).then_some(note),
            });
            return;
        }
        "code-block" | "code" | "sourcecode" | "testcode" | "doctest" | "ipython" => {
            let lang = if first.is_empty() || name == "doctest" {
                "python"
            } else {
                first
            };
            let code: Vec<String> = body
                .iter()
                .skip_while(|l| l.trim_start().starts_with(':') || l.trim().is_empty())
                .cloned()
                .collect();
            fence(out, lang, &dedent(&code));
            return;
        }
        "math" => {
            let mut code = Vec::new();
            if !arg.is_empty() {
                code.push(arg.to_string());
            }
            code.extend(body.iter().cloned());
            fence(out, "text", &code);
            return;
        }
        "rubric" => {
            out.push(format!("**{}**", inline(arg, st)));
            out.push(String::new());
            return;
        }
        "note" => "Note".to_string(),
        "warning" => "Warning".into(),
        "tip" => "Tip".into(),
        "hint" => "Hint".into(),
        "important" => "Important".into(),
        "attention" => "Attention".into(),
        "caution" => "Caution".into(),
        "danger" => "Danger".into(),
        "error" => "Error".into(),
        "seealso" => "See also".into(),
        "todo" => "Todo".into(),
        "admonition" => arg.to_string(),
        "versionadded" => format!("Added in version {first}"),
        "versionchanged" => format!("Changed in version {first}"),
        "versionremoved" => format!("Removed in version {first}"),
        // autodoc, toctree, image, …: nothing a reader of the reference needs.
        _ => return,
    };
    let mut content: Vec<String> = Vec::new();
    let text = match name {
        "admonition" => "",
        n if n.starts_with("version") => rest.trim(),
        _ => arg,
    };
    if !text.is_empty() {
        content.push(text.to_string());
    }
    content.extend(body.iter().cloned());
    let converted = convert_prose(&content, st);
    if out.last().is_some_and(|l| !l.trim().is_empty()) {
        out.push(String::new());
    }
    let mut first_line = true;
    for l in converted.iter().skip_while(|l| l.trim().is_empty()) {
        if first_line {
            out.push(format!("> **{}:** {}", label.trim(), l.trim_start()));
            first_line = false;
        } else if l.trim().is_empty() {
            out.push(">".into());
        } else {
            out.push(format!("> {l}"));
        }
    }
    if first_line {
        out.push(format!("> **{}.**", label.trim()));
    }
    while out.last().is_some_and(|l| l == ">") {
        out.pop();
    }
    out.push(String::new());
}

static GOOGLE_ENTRY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(\*{0,2}[A-Za-z_][\w.]*)\s*(?:\((.+?)\))?\s*:(?:\s+(.*))?$").expect("valid regex")
});

/// Entries of a list-shaped section. `returns` changes how a NumPy line without ` : `
/// reads: a type, not a name.
fn entries(body: &[String], style: Style, returns: bool) -> Vec<Entry> {
    let mut out: Vec<Entry> = Vec::new();
    let mut desc: Vec<Vec<String>> = Vec::new();
    for line in body {
        let t = line.trim();
        if indent(line) > 0 || t.is_empty() {
            if let Some(d) = desc.last_mut() {
                d.push(line.clone());
            }
            continue;
        }
        let entry = match style {
            Style::Google => match GOOGLE_ENTRY_RE.captures(t) {
                Some(c) => Entry {
                    name: c[1].to_string(),
                    ty: c.get(2).map(|m| m.as_str().trim().to_string()),
                    desc: c.get(3).map(|m| m.as_str().to_string()).unwrap_or_default(),
                },
                // A bare name (`ValueError`) starts an entry; other text continues one.
                None if !out.is_empty() && !XREF_TARGET_RE.is_match(t) => {
                    if let Some(d) = desc.last_mut() {
                        d.push(format!("  {t}"));
                    }
                    continue;
                }
                None => Entry {
                    name: t.to_string(),
                    ..Entry::default()
                },
            },
            Style::NumPy => {
                let (name, ty) = match t.split_once(" : ") {
                    Some((n, ty)) => (n.trim(), Some(ty.trim())),
                    None => match t.strip_suffix(':').or_else(|| t.strip_suffix(" :")) {
                        Some(n) => (n.trim(), None),
                        None if returns => ("", Some(t)),
                        None => (t, None),
                    },
                };
                Entry {
                    name: name.to_string(),
                    ty: ty.filter(|t| !t.is_empty()).map(str::to_string),
                    desc: String::new(),
                }
            }
        };
        out.push(entry);
        desc.push(Vec::new());
    }
    for (e, d) in out.iter_mut().zip(desc) {
        let mut parts: Vec<String> = Vec::new();
        if !e.desc.is_empty() {
            parts.push(e.desc.clone());
        }
        parts.extend(dedent(&d).iter().map(|l| l.trim().to_string()));
        e.desc = parts
            .into_iter()
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
    }
    out
}

/// A type written in a docstring: code, unless it already carries markup.
fn ty_md(ty: &str, st: &mut State) -> String {
    if ty.contains('`') {
        inline(ty, st)
    } else {
        format!("`{ty}`")
    }
}

/// A name that may be a cross-reference (an exception, a see-also target).
fn name_link(name: &str, st: &mut State) -> String {
    let n = name.trim();
    if n.contains('`') {
        return inline(n, st);
    }
    if XREF_TARGET_RE.is_match(n) {
        st.link(n);
        format!("[`{n}`]")
    } else {
        format!("`{n}`")
    }
}

fn entry_list(entries: &[Entry], st: &mut State, link_names: bool) -> Vec<String> {
    entries
        .iter()
        .map(|e| {
            let mut s = String::from("- ");
            if !e.name.is_empty() {
                s.push_str(&if link_names {
                    name_link(&e.name, st)
                } else {
                    format!("`{}`", e.name)
                });
            }
            if let Some(ty) = &e.ty {
                let t = ty_md(ty, st);
                if e.name.is_empty() {
                    s.push_str(&t);
                } else {
                    s.push_str(&format!(" ({t})"));
                }
            }
            if !e.desc.is_empty() {
                s.push_str(" — ");
                s.push_str(&inline(&e.desc, st));
            }
            s
        })
        .collect()
}

/// `bool: True when…` → (`bool`, `True when…`); plain prose → no type.
fn google_return(first: &str) -> (Option<&str>, &str) {
    if let Some((ty, rest)) = first.split_once(':') {
        let bare: String = {
            let mut depth = 0i32;
            ty.chars()
                .filter(|c| {
                    match c {
                        '[' | '(' => depth += 1,
                        ']' | ')' => depth -= 1,
                        _ => {}
                    }
                    depth == 0
                })
                .collect()
        };
        let words = bare.split_whitespace().count();
        if !ty.trim().is_empty() && !ty.ends_with(' ') && words <= 3 && !bare.contains(". ") {
            return (Some(ty.trim()), rest.trim());
        }
    }
    (None, first)
}

fn heading(out: &mut Vec<String>, title: &str) {
    if out.last().is_some_and(|l| !l.trim().is_empty()) {
        out.push(String::new());
    }
    out.push(format!("# {title}"));
    out.push(String::new());
}

fn section(kind: &Kind, body: &[String], style: Style, st: &mut State, out: &mut Vec<String>) {
    match kind {
        Kind::Params(title) => {
            let es = entries(body, style, false);
            heading(out, title);
            out.extend(entry_list(&es, st, false));
            st.params.extend(es);
        }
        Kind::Attributes => {
            let es = entries(body, style, false);
            heading(out, "Attributes");
            out.extend(entry_list(&es, st, false));
            st.attributes.extend(es);
        }
        Kind::Raises | Kind::Warns => {
            let es = entries(body, style, false);
            heading(
                out,
                if *kind == Kind::Raises {
                    "Errors"
                } else {
                    "Warns"
                },
            );
            out.extend(entry_list(&es, st, true));
        }
        Kind::Returns | Kind::Yields => {
            heading(
                out,
                if *kind == Kind::Returns {
                    "Returns"
                } else {
                    "Yields"
                },
            );
            if style == Style::NumPy {
                let es = entries(body, style, true);
                if let [only] = es.as_slice()
                    && only.name.is_empty()
                {
                    let mut s = only.ty.as_deref().map(|t| ty_md(t, st)).unwrap_or_default();
                    if !only.desc.is_empty() {
                        s.push_str(" — ");
                        s.push_str(&inline(&only.desc, st));
                    }
                    out.push(s);
                } else {
                    out.extend(entry_list(&es, st, false));
                }
            } else {
                let mut lines = body.to_vec();
                if let Some(first) = lines.first().cloned() {
                    let (ty, rest) = google_return(first.trim());
                    if let Some(ty) = ty {
                        let t = ty_md(ty, st);
                        lines[0] = if rest.is_empty() {
                            t
                        } else {
                            format!("{t} — {rest}")
                        };
                    }
                }
                // Continuation lines of the first paragraph are indented under it.
                let joined: Vec<String> = lines.iter().map(|l| l.trim().to_string()).collect();
                out.extend(convert_prose(&joined, st));
            }
        }
        Kind::Examples => {
            heading(out, "Examples");
            out.extend(convert_prose(body, st));
        }
        Kind::SeeAlso => {
            heading(out, "See also");
            let es = entries(body, Style::NumPy, false);
            let looks_like_refs = !es.is_empty()
                && es.iter().all(|e| {
                    e.name
                        .split(',')
                        .all(|n| XREF_TARGET_RE.is_match(n.trim()) || n.contains(":`"))
                });
            if looks_like_refs {
                for e in es {
                    let names: Vec<String> = e
                        .name
                        .split(',')
                        .filter(|n| !n.trim().is_empty())
                        .map(|n| name_link(n, st))
                        .collect();
                    let mut s = format!("- {}", names.join(", "));
                    let desc = [e.ty.unwrap_or_default(), e.desc].join(" ");
                    if !desc.trim().is_empty() {
                        s.push_str(" — ");
                        s.push_str(&inline(desc.trim(), st));
                    }
                    out.push(s);
                }
            } else {
                out.extend(convert_prose(body, st));
            }
        }
        Kind::Text(title) => {
            heading(out, title);
            out.extend(convert_prose(body, st));
        }
    }
    out.push(String::new());
}

static SPHINX_PARAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^:(?:param|parameter|arg|argument|key|keyword)\s+(?:(.+?)\s+)?(\*{0,2}[A-Za-z_]\w*)\s*:\s*(.*)$",
    )
    .expect("valid regex")
});
static SPHINX_TYPE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^:(type|vartype)\s+(\*{0,2}[A-Za-z_]\w*)\s*:\s*(.*)$").expect("valid regex")
});
static SPHINX_RET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^:(returns?|rtype|yields?|ytype)\s*:\s*(.*)$").expect("valid regex")
});
static SPHINX_RAISES_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^:(?:raises?|except|exception)\s+(.+?)\s*:\s*(.*)$").expect("valid regex")
});
static SPHINX_VAR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^:(?:ivar|cvar|var)\s+(?:(.+?)\s+)?([A-Za-z_]\w*)\s*:\s*(.*)$")
        .expect("valid regex")
});
static SPHINX_META_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^:(?:meta\s+[\w-]+|noindex|no-index)\s*:").expect("valid regex"));

/// Sphinx info fields, collected wherever they appear and emitted at the end.
#[derive(Default)]
struct Sphinx {
    params: Vec<Entry>,
    types: Vec<(String, String)>,
    returns: Option<String>,
    rtype: Option<String>,
    yields: Option<String>,
    ytype: Option<String>,
    raises: Vec<Entry>,
    vars: Vec<Entry>,
    vartypes: Vec<(String, String)>,
}

impl Sphinx {
    /// Parse the field at `lines[i]`; returns the lines it spans, or `None` when the
    /// line is not a known field.
    fn take(&mut self, lines: &[String], i: usize) -> Option<usize> {
        let t = lines[i].trim();
        let end = block_end(lines, i + 1, indent(&lines[i]));
        let with_cont = |first: &str| {
            std::iter::once(first)
                .chain(lines[i + 1..end].iter().map(|l| l.trim()))
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        };
        if let Some(c) = SPHINX_PARAM_RE.captures(t) {
            let desc = with_cont(&c[3]);
            self.params.push(Entry {
                name: c[2].to_string(),
                ty: c.get(1).map(|m| m.as_str().to_string()),
                desc,
            });
        } else if let Some(c) = SPHINX_TYPE_RE.captures(t) {
            let ty = with_cont(&c[3]);
            let list = if &c[1] == "type" {
                &mut self.types
            } else {
                &mut self.vartypes
            };
            list.push((c[2].to_string(), ty));
        } else if let Some(c) = SPHINX_RET_RE.captures(t) {
            let v = Some(with_cont(&c[2]));
            match &c[1] {
                "return" | "returns" => self.returns = v,
                "rtype" => self.rtype = v,
                "yield" | "yields" => self.yields = v,
                _ => self.ytype = v,
            }
        } else if let Some(c) = SPHINX_RAISES_RE.captures(t) {
            let desc = with_cont(&c[2]);
            for name in c[1].split(',').map(str::trim).filter(|n| !n.is_empty()) {
                self.raises.push(Entry {
                    name: name.to_string(),
                    ty: None,
                    desc: desc.clone(),
                });
            }
        } else if let Some(c) = SPHINX_VAR_RE.captures(t) {
            let desc = with_cont(&c[3]);
            self.vars.push(Entry {
                name: c[2].to_string(),
                ty: c.get(1).map(|m| m.as_str().to_string()),
                desc,
            });
        } else if SPHINX_META_RE.is_match(t) {
        } else {
            return None;
        }
        Some(end - i)
    }

    fn emit(mut self, st: &mut State, out: &mut Vec<String>) {
        fn apply(entries: &mut Vec<Entry>, types: &[(String, String)]) {
            for (name, ty) in types {
                match entries
                    .iter_mut()
                    .find(|e| e.name.trim_start_matches('*') == name.trim_start_matches('*'))
                {
                    Some(e) => e.ty = Some(ty.clone()),
                    None => entries.push(Entry {
                        name: name.clone(),
                        ty: Some(ty.clone()),
                        desc: String::new(),
                    }),
                }
            }
        }
        apply(&mut self.params, &self.types);
        apply(&mut self.vars, &self.vartypes);
        if !self.params.is_empty() {
            heading(out, "Parameters");
            out.extend(entry_list(&self.params, st, false));
            st.params.extend(self.params);
            out.push(String::new());
        }
        for (title, value, ty) in [
            ("Returns", self.returns, self.rtype),
            ("Yields", self.yields, self.ytype),
        ] {
            if value.is_none() && ty.is_none() {
                continue;
            }
            heading(out, title);
            let mut s = ty.map(|t| ty_md(&t, st)).unwrap_or_default();
            if let Some(v) = value.filter(|v| !v.is_empty()) {
                if !s.is_empty() {
                    s.push_str(" — ");
                }
                s.push_str(&inline(&v, st));
            }
            out.push(s);
            out.push(String::new());
        }
        if !self.raises.is_empty() {
            heading(out, "Errors");
            out.extend(entry_list(&self.raises, st, true));
            out.push(String::new());
        }
        if !self.vars.is_empty() {
            heading(out, "Attributes");
            out.extend(entry_list(&self.vars, st, false));
            st.attributes.extend(self.vars);
            out.push(String::new());
        }
    }
}

static ROLE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r":(?:py:)?([a-z][a-z-]*):`([^`]+)`").expect("valid regex"));
static HYPERLINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`([^`<]+?)\s*<([^`>]+)>`__?").expect("valid regex"));
static NAMED_REF_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`([^`]+)`__?(\W|$)").expect("valid regex"));
static DOUBLE_TICK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"``([^`]+)``").expect("valid regex"));
static XREF_TARGET_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z_][\w.]*(?:\(\))?$").expect("valid regex"));

const XREF_ROLES: &[&str] = &[
    "class",
    "func",
    "function",
    "meth",
    "method",
    "attr",
    "attribute",
    "mod",
    "module",
    "exc",
    "exception",
    "data",
    "const",
    "obj",
    "type",
    "any",
];

/// Inline reST → Markdown: cross-reference roles, hyperlinks, ``literals``.
fn inline(s: &str, st: &mut State) -> String {
    if !s.contains('`') {
        return s.to_string();
    }
    let s = ROLE_RE.replace_all(s, |c: &regex::Captures| {
        let (role, inner) = (&c[1], c[2].trim());
        // `Title <target>`: the target is what links.
        let (title, target) = match inner.rsplit_once('<') {
            Some((t, rest)) if inner.ends_with('>') => {
                (Some(t.trim()), rest.trim_end_matches('>').trim())
            }
            _ => (None, inner),
        };
        if XREF_ROLES.contains(&role) {
            if let Some(plain) = target.strip_prefix('!') {
                return format!("`{plain}`");
            }
            let short = target.starts_with('~');
            let path = target.trim_start_matches(['~', '.']);
            if !XREF_TARGET_RE.is_match(path) {
                return format!("`{}`", title.unwrap_or(target));
            }
            st.link(path.trim_end_matches("()"));
            let shown = match title {
                Some(t) => t.to_string(),
                None if short => path.rsplit('.').next().unwrap_or(path).to_string(),
                None => path.to_string(),
            };
            if shown == path {
                format!("[`{path}`]")
            } else if XREF_TARGET_RE.is_match(&shown) {
                format!("[`{shown}`]({})", path.trim_end_matches("()"))
            } else {
                format!("[`{path}`]")
            }
        } else {
            match role {
                "pep" => format!("PEP {inner}"),
                "rfc" => format!("RFC {inner}"),
                "ref" | "doc" | "term" => title.unwrap_or(target).to_string(),
                _ => format!("`{}`", title.unwrap_or(target)),
            }
        }
    });
    let s = HYPERLINK_RE.replace_all(&s, "[$1]($2)");
    let s = DOUBLE_TICK_RE.replace_all(&s, "`$1`");
    // `name`_ (a reST reference) reads as its name.
    let s = NAMED_REF_RE.replace_all(&s, "$1$2");
    s.into_owned()
}

/// Fenced code blocks; a `python` fence opening with `>>>` is a doctest.
fn examples(markdown: &str) -> Vec<CodeExample> {
    let mut out = Vec::new();
    let mut current: Option<(String, Vec<&str>)> = None;
    for line in markdown.lines() {
        let t = line.trim_start();
        let fence = t.strip_prefix("```").or_else(|| t.strip_prefix("~~~"));
        match (fence, current.take()) {
            (Some(info), None) => {
                let lang = info
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .find(|s| !s.is_empty())
                    .unwrap_or("python")
                    .to_string();
                current = Some((lang, Vec::new()));
            }
            (Some(_), Some((lang, body))) => {
                let code = body.join("\n");
                let doctest = matches!(lang.as_str(), "python" | "py" | "pycon")
                    && code.trim_start().starts_with(">>>");
                out.push(CodeExample {
                    lang,
                    code,
                    doctest,
                });
            }
            (None, Some((lang, mut body))) => {
                body.push(line);
                current = Some((lang, body));
            }
            (None, None) => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn md(s: &str) -> PyDoc {
        from_text(s, 1, 10)
    }

    #[test]
    fn literal_values() {
        assert_eq!(literal_value(r#""""Doc.""""#).as_deref(), Some("Doc."));
        assert_eq!(literal_value("r'''a\\b'''").as_deref(), Some("a\\b"));
        assert_eq!(
            literal_value(r#""say \"hi\"""#).as_deref(),
            Some("say \"hi\"")
        );
        assert_eq!(literal_value(r#"f"x""#), None);
        assert_eq!(literal_value(r#"b"x""#), None);
    }

    #[test]
    fn cleandoc_like_inspect() {
        let lines = cleandoc("  Summary.\n\n    Body line.\n      Indented.\n    ");
        assert_eq!(lines, vec!["Summary.", "", "Body line.", "  Indented."]);
    }

    #[test]
    fn google_style() {
        let d = md(
            "Fetch rows.\n\n    Longer text with :class:`Table` and ``code``.\n\n    Args:\n        table (Table): The table to read. Spans\n            two lines.\n        limit: Max rows.\n        *args: Extra.\n\n    Returns:\n        list[Row]: The rows, in order.\n\n    Raises:\n        KeyError: If missing.\n        ValueError\n\n    Example:\n        >>> fetch(t, 2)\n        [Row(1), Row(2)]\n\n    Note:\n        Slow on big tables.\n    ",
        );
        let doc = d.doc.unwrap();
        assert_eq!(doc.summary, "Fetch rows.");
        let m = &doc.markdown;
        assert!(m.contains("Longer text with [`Table`] and `code`."), "{m}");
        assert!(
            m.contains("# Parameters\n\n- `table` (`Table`) — The table to read. Spans two lines.\n- `limit` — Max rows.\n- `*args` — Extra."),
            "{m}"
        );
        assert!(
            m.contains("# Returns\n\n`list[Row]` — The rows, in order."),
            "{m}"
        );
        assert!(
            m.contains("# Errors\n\n- [`KeyError`] — If missing.\n- [`ValueError`]"),
            "{m}"
        );
        assert!(
            m.contains("# Examples\n\n```python\n>>> fetch(t, 2)\n[Row(1), Row(2)]\n```"),
            "{m}"
        );
        assert!(m.contains("# Notes\n\nSlow on big tables."), "{m}");
        let names: Vec<&str> = doc.sections.iter().map(|(h, _)| h.as_str()).collect();
        assert_eq!(
            names,
            vec!["parameters", "returns", "errors", "examples", "notes"]
        );
        assert_eq!(doc.examples.len(), 1);
        assert_eq!(doc.examples[0].lang, "python");
        assert!(doc.examples[0].doctest);
        assert!(doc.links.contains(&"Table".to_string()));
        assert_eq!(d.params[0].ty.as_deref(), Some("Table"));
        assert_eq!(d.params.len(), 3);
    }

    #[test]
    fn numpy_style() {
        let d = md(
            "Add arrays.\n\nParameters\n----------\nx, y : array_like\n    Inputs.\nout : ndarray, optional\n    Where to write.\n\nReturns\n-------\nndarray\n    The sum.\n\nRaises\n------\nValueError\n    Shapes differ.\n\nSee Also\n--------\nsubtract : The opposite.\nnumpy.multiply\n\nExamples\n--------\n>>> add(1, 2)\n3\n",
        );
        let doc = d.doc.unwrap();
        let m = &doc.markdown;
        assert!(
            m.contains("# Parameters\n\n- `x, y` (`array_like`) — Inputs.\n- `out` (`ndarray, optional`) — Where to write."),
            "{m}"
        );
        assert!(m.contains("# Returns\n\n`ndarray` — The sum."), "{m}");
        assert!(
            m.contains("# Errors\n\n- [`ValueError`] — Shapes differ."),
            "{m}"
        );
        assert!(
            m.contains("# See also\n\n- [`subtract`] — The opposite.\n- [`numpy.multiply`]"),
            "{m}"
        );
        assert!(m.contains("```python\n>>> add(1, 2)\n3\n```"), "{m}");
        assert!(doc.examples[0].doctest);
        assert!(!m.contains("----"), "{m}");
    }

    #[test]
    fn sphinx_fields() {
        let d = md(
            "Open a connection.\n\n:param host: Server name,\n    or address.\n:type host: str\n:param int port: Port.\n:returns: A live connection.\n:rtype: :class:`Conn`\n:raises ConnectionError: When unreachable.\n:ivar timeout: Seconds.\n:vartype timeout: float\n",
        );
        let doc = d.doc.unwrap();
        let m = &doc.markdown;
        assert!(
            m.starts_with("Open a connection.\n\n# Parameters\n\n- `host` (`str`) — Server name, or address.\n- `port` (`int`) — Port."),
            "{m}"
        );
        assert!(
            m.contains("# Returns\n\n[`Conn`] — A live connection."),
            "{m}"
        );
        assert!(
            m.contains("# Errors\n\n- [`ConnectionError`] — When unreachable."),
            "{m}"
        );
        assert!(
            m.contains("# Attributes\n\n- `timeout` (`float`) — Seconds."),
            "{m}"
        );
        assert_eq!(d.params[0].ty.as_deref(), Some("str"));
        assert_eq!(d.attributes[0].name, "timeout");
        assert!(doc.links.contains(&"Conn".to_string()));
    }

    #[test]
    fn roles_become_intra_doc_links() {
        let d = md(
            "Uses :class:`Foo`, :meth:`A.b`, :func:`~pkg.mod.run`, :py:attr:`x`,\n:class:`!Plain`, :ref:`guide <label>`, :pep:`8`, and `docs <https://x.org>`_.",
        );
        let doc = d.doc.unwrap();
        assert_eq!(
            doc.markdown,
            "Uses [`Foo`], [`A.b`], [`run`](pkg.mod.run), [`x`],\n`Plain`, guide, PEP 8, and [docs](https://x.org)."
        );
        assert_eq!(doc.links, vec!["Foo", "A.b", "pkg.mod.run", "x"]);
    }

    #[test]
    fn directives_and_literal_blocks() {
        let d = md(
            "Old API.\n\n.. deprecated:: 1.2\n   Use :func:`new` instead.\n\n.. note:: Thread-safe.\n\n.. versionadded:: 1.0\n\nFor example::\n\n    x = old()\n\n.. code-block:: python\n   :linenos:\n\n   y = 1\n\n.. _target:\n\nEnd.",
        );
        let dep = d.deprecated.unwrap();
        assert_eq!(dep.since.as_deref(), Some("1.2"));
        assert_eq!(dep.note.as_deref(), Some("Use [`new`] instead."));
        let m = d.doc.unwrap().markdown;
        assert!(!m.contains("deprecated"), "{m}");
        assert!(m.contains("> **Note:** Thread-safe."), "{m}");
        assert!(m.contains("> **Added in version 1.0.**"), "{m}");
        assert!(m.contains("For example:\n\n```text\nx = old()\n```"), "{m}");
        assert!(m.contains("```python\ny = 1\n```"), "{m}");
        assert!(!m.contains("_target"), "{m}");
        assert!(m.ends_with("End."), "{m}");
    }

    #[test]
    fn markdown_docstrings_pass_through() {
        let d = md("Summary.\n\n```py\n# comment\nx = 1\n```\n\n## Usage\n\nCall [`run`].");
        let doc = d.doc.unwrap();
        assert!(doc.markdown.contains("```py\n# comment\nx = 1\n```"));
        assert_eq!(doc.sections[0].0, "usage");
        assert_eq!(doc.examples[0].lang, "py");
    }

    #[test]
    fn prose_returns_keep_their_text() {
        let d = md("Do it.\n\nReturns:\n    The number of rows written. It may be zero.\n");
        assert!(
            d.doc
                .unwrap()
                .markdown
                .contains("# Returns\n\nThe number of rows written. It may be zero.")
        );
    }

    #[test]
    fn empty_is_none() {
        assert!(md("   \n  ").doc.is_none());
    }
}
