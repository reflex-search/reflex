//! Capabilities: what a module does, proved by what it imports.
//!
//! "This module is an HTTP server" is a claim; `use axum::Router` at
//! `src/cli/serve.rs:30` is evidence. Capabilities come only from imports of known
//! frameworks, each with the import line as provenance, never from file names (a file
//! called `app.rs` is not a web app). The grounded writer may state a capability only
//! when a fact like this is cited.

use super::Corpus;
use serde::Serialize;

/// One capability, and the import prefixes that prove it.
pub struct Capability {
    pub id: &'static str,
    pub label: &'static str,
    /// Import prefixes; a match is the whole prefix or the prefix followed by `::`,
    /// `.`, `/` or `-`.
    pub imports: &'static [&'static str],
}

pub const CAPABILITIES: &[Capability] = &[
    Capability {
        id: "http-server",
        label: "HTTP server",
        imports: &[
            "axum",
            "actix_web",
            "actix-web",
            "warp",
            "rocket",
            "poem",
            "tide",
            "hyper::server",
            "fastapi",
            "flask",
            "django",
            "starlette",
            "tornado",
            "aiohttp.web",
            "express",
            "koa",
            "fastify",
            "@nestjs/core",
            "hono",
            "github.com/gin-gonic/gin",
            "github.com/labstack/echo",
            "github.com/gofiber/fiber",
            "github.com/go-chi/chi",
            "github.com/gorilla/mux",
        ],
    },
    Capability {
        id: "http-client",
        label: "HTTP client",
        imports: &[
            "reqwest",
            "ureq",
            "surf",
            "requests",
            "httpx",
            "aiohttp",
            "axios",
            "node-fetch",
            "got",
            "undici",
        ],
    },
    Capability {
        id: "cli",
        label: "command-line interface",
        imports: &[
            "clap",
            "structopt",
            "argh",
            "click",
            "typer",
            "argparse",
            "commander",
            "yargs",
            "@oclif/core",
            "github.com/spf13/cobra",
            "github.com/urfave/cli",
        ],
    },
    Capability {
        id: "tui",
        label: "terminal user interface",
        imports: &[
            "ratatui",
            "tui",
            "crossterm",
            "cursive",
            "textual",
            "curses",
            "ink",
            "blessed",
            "github.com/charmbracelet/bubbletea",
            "github.com/rivo/tview",
        ],
    },
    Capability {
        id: "database",
        label: "database",
        imports: &[
            "rusqlite",
            "sqlx",
            "diesel",
            "sea_orm",
            "postgres",
            "tokio_postgres",
            "redis",
            "mongodb",
            "sqlalchemy",
            "sqlite3",
            "psycopg",
            "psycopg2",
            "pymongo",
            "pg",
            "mysql2",
            "mongoose",
            "@prisma/client",
            "knex",
            "typeorm",
            "sequelize",
            "better-sqlite3",
            "database/sql",
            "gorm.io/gorm",
            "go.mongodb.org/mongo-driver",
            "github.com/jackc/pgx",
        ],
    },
    Capability {
        id: "async",
        label: "async runtime",
        imports: &["tokio", "async_std", "smol", "asyncio", "trio", "anyio"],
    },
    Capability {
        id: "llm",
        label: "LLM API client",
        imports: &[
            "anthropic",
            "openai",
            "async_openai",
            "langchain",
            "@anthropic-ai/sdk",
            "@google/generative-ai",
        ],
    },
    Capability {
        id: "parsing",
        label: "source-code parsing",
        imports: &[
            "tree_sitter",
            "tree-sitter",
            "syn",
            "nom",
            "pest",
            "web-tree-sitter",
            "@babel/parser",
            "go/ast",
        ],
    },
    Capability {
        id: "file-watching",
        label: "file watching",
        imports: &[
            "notify",
            "watchdog",
            "chokidar",
            "github.com/fsnotify/fsnotify",
        ],
    },
    Capability {
        id: "web-ui",
        label: "web user interface",
        imports: &["react", "react-dom", "vue", "svelte", "solid-js", "preact"],
    },
];

/// A capability seen in one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Evidence {
    pub capability: &'static str,
    pub label: &'static str,
    /// The import as written (`axum::Router`).
    pub import: String,
    pub file: usize,
    pub line: u32,
}

fn matches(import: &str, prefix: &str) -> bool {
    import == prefix
        || import.strip_prefix(prefix).is_some_and(|rest| {
            rest.starts_with("::") || rest.starts_with('.') || rest.starts_with('/')
        })
}

/// Capabilities per file, first import line wins.
pub fn detect(corpus: &Corpus) -> Vec<Evidence> {
    let mut out: Vec<Evidence> = Vec::new();
    for imp in &corpus.imports {
        for cap in CAPABILITIES {
            if cap.imports.iter().any(|p| matches(&imp.path, p))
                && !out
                    .iter()
                    .any(|e| e.file == imp.file && e.capability == cap.id)
            {
                out.push(Evidence {
                    capability: cap.id,
                    label: cap.label,
                    import: imp.path.clone(),
                    file: imp.file,
                    line: imp.line,
                });
            }
        }
    }
    out.sort_by(|a, b| (a.file, a.capability).cmp(&(b.file, b.capability)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_matching() {
        assert!(matches("axum", "axum"));
        assert!(matches("axum::Router", "axum"));
        assert!(!matches("axumx", "axum"));
        assert!(matches("github.com/spf13/cobra", "github.com/spf13/cobra"));
        assert!(matches("database/sql", "database/sql"));
        assert!(matches("fastapi.routing", "fastapi"));
        assert!(!matches("tokio_util", "tokio"), "a different crate");
    }
}
