//! Symbol-extraction equivalence: the `Vec<SearchResult>` a file yields must not
//! change when the extractors are made faster.
//!
//! The corpus snapshot was generated on the per-file `Query::new` / line-skip
//! preview implementation (2.0.0) and pins the output of the cached-query and
//! byte-offset-preview implementation that replaced it. The synthetic snapshot
//! covers the line-ending and offset shapes the corpus lacks.

use reflex::models::Language;
use reflex::parsers::ParserFactory;
use std::fs;
use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some(".reflex") {
                continue;
            }
            walk(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn dump(rel: &str, source: &str) -> String {
    let language = Language::from_path(Path::new(rel));
    let symbols = ParserFactory::parse(rel, source, language).unwrap_or_else(|e| {
        panic!("parse {rel}: {e}");
    });
    format!(
        "== {rel} ({:?}, {} symbols)\n{}\n",
        language,
        symbols.len(),
        serde_json::to_string_pretty(&symbols).unwrap()
    )
}

#[test]
fn corpus_symbols_are_stable() {
    let corpus = Path::new("tests/corpus");
    let mut files = Vec::new();
    walk(corpus, &mut files);
    files.sort();
    let mut out = String::new();
    for path in files {
        let rel = path
            .strip_prefix(corpus)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let Ok(source) = fs::read_to_string(&path) else {
            continue; // binary / non-UTF-8 fixtures are not symbol inputs
        };
        out.push_str(&dump(&rel, &source));
    }
    insta::assert_snapshot!("corpus_symbols", out);
}

/// Shapes the corpus lacks: CRLF, a lone `\r`, no trailing newline, a symbol on
/// the last line and at byte 0, blank lines and multi-byte characters before a
/// symbol, and script offsets inside `.vue` / `.svelte`.
#[test]
fn synthetic_symbols_are_stable() {
    let cases: Vec<(&str, String)> = vec![
        (
            "crlf.rs",
            "// top\r\nfn alpha() {\r\n    let x = 1;\r\n}\r\n\r\nstruct Beta {\r\n    a: u32,\r\n}\r\n".to_string(),
        ),
        (
            "lone_cr.py",
            "def first():\r    pass\n\ndef second():\n    value = 1\r\n    return value\n".to_string(),
        ),
        ("no_newline.go", "package p\n\nfunc last() int { return 1 }".to_string()),
        ("byte_zero.ts", "function atZero() {}\nconst k = 1;\n\n\n\nexport class Late {\n  m() {}\n}".to_string()),
        (
            "unicode.rs",
            "// ééé 😀 日本語\n\n\n/// docs ✓\npub fn after_unicode(名前: &str) -> usize { 名前.len() }\nconst Ω: u32 = 1;\n".to_string(),
        ),
        (
            "widget.vue",
            "<template>\n  <div>{{ msg }}</div>\n</template>\n\n<script setup lang=\"ts\">\nimport { ref } from 'vue'\n\nconst msg = ref('hi')\nfunction greet(name: string) {\n  return `hi ${name}`\n}\n</script>\n\n<style>\n.a { color: red }\n</style>\n".to_string(),
        ),
        (
            "widget.svelte",
            "<script lang=\"ts\">\n  export let count = 0;\n  function increment() {\n    count += 1;\n  }\n</script>\n\n<button on:click={increment}>{count}</button>\n".to_string(),
        ),
        (
            "long_lines.js",
            format!("const big = \"{}\";\nfunction afterBig() {{ return big; }}\n", "x".repeat(700)),
        ),
    ];
    let mut out = String::new();
    for (name, source) in &cases {
        out.push_str(&dump(name, source));
    }
    insta::assert_snapshot!("synthetic_symbols", out);
}

/// Languages the corpus does not cover, so the combined-query conversion of
/// their extractors is pinned too.
#[test]
fn more_languages_are_stable() {
    let cases: Vec<(&str, &str)> = vec![
        (
            "shapes.cpp",
            "#include <vector>\nnamespace geo {\nclass Shape {\npublic:\n  virtual double area() const = 0;\n  int sides = 0;\n};\nstruct Point { double x, y; };\nenum class Kind { Circle, Square };\nusing Points = std::vector<Point>;\ndouble total(const Points& ps) {\n  double sum = 0;\n  for (auto& p : ps) { sum += p.x; }\n  return sum;\n}\n}\n",
        ),
        (
            "Service.cs",
            "using System;\nnamespace App.Core {\n  [Serializable]\n  public class Service : IService {\n    public string Name { get; set; }\n    public event EventHandler Changed;\n    public int this[int i] => i;\n    public void Run() { var local = 1; Console.WriteLine(local); }\n  }\n  public interface IService { void Run(); }\n  public struct Pair { public int A; }\n  public enum Mode { Fast, Slow }\n  public record Person(string First);\n  public delegate void Handler(int x);\n}\n",
        ),
        (
            "model.rb",
            "require 'json'\nmodule Shop\n  class Cart\n    attr_accessor :items\n    @@count = 0\n    LIMIT = 10\n    def initialize\n      @items = []\n      total = 0\n    end\n    def self.build\n      Cart.new\n    end\n  end\nend\n",
        ),
        (
            "Main.kt",
            "package demo\nimport kotlin.math.abs\n@Target(AnnotationTarget.CLASS)\nannotation class Tag\n@Tag\nclass Account(val id: Int) {\n  val balance: Double = 0.0\n  fun deposit(amount: Double): Double {\n    val next = balance + amount\n    return next\n  }\n}\ninterface Ledger { fun post() }\nobject Registry { val all = listOf<Account>() }\nfun main() { println(abs(-1)) }\n",
        ),
        (
            "lib.zig",
            "const std = @import(\"std\");\nconst Point = struct { x: i32, y: i32 };\nconst Color = enum { red, green };\nvar counter: u32 = 0;\npub fn add(a: i32, b: i32) i32 {\n    const sum = a + b;\n    return sum;\n}\ntest \"add works\" {\n    try std.testing.expect(add(1, 2) == 3);\n}\n",
        ),
        (
            "Widget.java",
            "package app;\nimport java.util.List;\n@interface Marker {}\n@Marker\npublic class Widget implements Runnable {\n  private int size = 0;\n  public Widget() { size = 1; }\n  @Override\n  public void run() { int local = size; }\n}\ninterface Runnable2 { void go(); }\nenum State { ON, OFF }\n",
        ),
        (
            "app.php",
            "<?php\nnamespace App;\nuse App\\Base;\n#[Attribute]\nclass Route {}\ninterface Handler { public function handle(); }\ntrait Loggable { public function log() {} }\nenum Status { case Active; }\nconst VERSION = '1';\nclass Controller extends Base {\n  public $name = 'x';\n  public function index() { $local = 1; return $local; }\n}\nfunction helper() {}\n",
        ),
        (
            "util.c",
            "#include <stdio.h>\n#define MAX 10\ntypedef struct Node { int v; struct Node *next; } Node;\nunion U { int i; float f; };\nenum E { A, B };\nstatic int counter = 0;\nint sum(int a, int b) {\n  int local = a + b;\n  return local;\n}\n",
        ),
        (
            "script.py",
            "import os\nGLOBAL = 1\nclass Thing:\n    def method(self):\n        local = 1\n        return local\ndef func(a, b=2):\n    inner = a\n    return inner\nsquare = lambda x: x * x\n",
        ),
    ];
    let mut out = String::new();
    for (name, source) in &cases {
        out.push_str(&dump(name, source));
    }
    insta::assert_snapshot!("more_languages", out);
}
