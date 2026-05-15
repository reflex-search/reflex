use anyhow::Result;
use serde_json::json;
use crate::cache::CacheManager;
use crate::indexer::Indexer;
use crate::models::{IndexConfig, Language};
use crate::query::{QueryEngine, QueryFilter};


/// Handle the `serve` subcommand
pub(super) fn handle_serve(port: u16, host: String) -> Result<()> {
    log::info!("Starting HTTP server on {}:{}", host, port);

    println!("Starting Reflex HTTP server...");
    println!("  Address: http://{}:{}", host, port);
    println!("\nEndpoints:");
    println!("  GET  /query?q=<pattern>&lang=<lang>&kind=<kind>&limit=<n>&symbols=true&regex=true&exact=true&contains=true&expand=true&file=<pattern>&timeout=<secs>&glob=<pattern>&exclude=<pattern>&paths=true&dependencies=true");
    println!("  GET  /stats");
    println!("  GET  /health");
    println!("  POST /index");
    println!("\nPress Ctrl+C to stop.");

    // Start the server using tokio runtime
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        run_server(port, host).await
    })
}


/// Run the HTTP server
async fn run_server(port: u16, host: String) -> Result<()> {
    use axum::{
        extract::{Query as AxumQuery, State},
        http::StatusCode,
        response::{IntoResponse, Json},
        routing::{get, post},
        Router,
    };
    use tower_http::cors::{CorsLayer, Any};
    use std::sync::Arc;

    // Server state shared across requests
    #[derive(Clone)]
    struct AppState {
        cache_path: String,
    }

    // Query parameters for GET /query
    #[derive(Debug, serde::Deserialize)]
    struct QueryParams {
        q: String,
        #[serde(default)]
        lang: Option<String>,
        #[serde(default)]
        kind: Option<String>,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        offset: Option<usize>,
        #[serde(default)]
        symbols: bool,
        #[serde(default)]
        regex: bool,
        #[serde(default)]
        exact: bool,
        #[serde(default)]
        contains: bool,
        #[serde(default)]
        expand: bool,
        #[serde(default)]
        file: Option<String>,
        #[serde(default = "default_timeout")]
        timeout: u64,
        #[serde(default)]
        glob: Vec<String>,
        #[serde(default)]
        exclude: Vec<String>,
        #[serde(default)]
        paths: bool,
        #[serde(default)]
        force: bool,
        #[serde(default)]
        dependencies: bool,
    }

    // Default timeout for HTTP queries (30 seconds)
    fn default_timeout() -> u64 {
        30
    }

    // Request body for POST /index
    #[derive(Debug, serde::Deserialize, Default)]
    struct IndexRequest {
        #[serde(default)]
        force: bool,
        #[serde(default)]
        languages: Vec<String>,
    }

    // GET /query endpoint
    async fn handle_query_endpoint(
        State(state): State<Arc<AppState>>,
        AxumQuery(params): AxumQuery<QueryParams>,
    ) -> Result<Json<crate::models::QueryResponse>, (StatusCode, Json<serde_json::Value>)> {
        log::info!("Query request: pattern={}", params.q);

        // REF-63: reject empty queries with 400 instead of passing them to the engine
        if params.q.trim().is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": { "kind": "QuerySyntaxError", "message": "Query parameter 'q' cannot be empty" } })),
            ));
        }

        let cache = CacheManager::new(&state.cache_path);
        let engine = QueryEngine::new(cache);

        // Parse language filter
        let language = if let Some(lang_str) = params.lang.as_deref() {
            match Language::from_name(lang_str) {
                Some(l) => Some(l),
                None => return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": { "kind": "QuerySyntaxError", "message": format!("Unknown language '{}'. Supported: {}", lang_str, Language::supported_names_help()) } })),
                )),
            }
        } else {
            None
        };

        // Parse symbol kind
        let kind = params.kind.as_deref().and_then(|s| {
            let capitalized = {
                let mut chars = s.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars.flat_map(|c| c.to_lowercase())).collect(),
                }
            };

            capitalized.parse::<crate::models::SymbolKind>()
                .ok()
                .or_else(|| {
                    log::debug!("Treating '{}' as unknown symbol kind for filtering", s);
                    Some(crate::models::SymbolKind::Unknown(s.to_string()))
                })
        });

        // Smart behavior: --kind implies --symbols
        let symbols_mode = params.symbols || kind.is_some();

        // Smart limit handling (same as CLI and MCP)
        let final_limit = if params.paths && params.limit.is_none() {
            None  // --paths without explicit limit means no limit
        } else if let Some(user_limit) = params.limit {
            Some(user_limit)  // Use user-specified limit
        } else {
            Some(100)  // Default: limit to 100 results for token efficiency
        };

        let filter = QueryFilter {
            language,
            kind,
            use_ast: false,
            use_regex: params.regex,
            limit: final_limit,
            symbols_mode,
            expand: params.expand,
            file_pattern: params.file,
            exact: params.exact,
            use_contains: params.contains,
            timeout_secs: params.timeout,
            glob_patterns: params.glob,
            exclude_patterns: params.exclude,
            paths_only: params.paths,
            offset: params.offset,
            force: params.force,
            suppress_output: true,  // HTTP API always returns JSON, suppress warnings
            include_dependencies: params.dependencies,
            ..Default::default()
        };

        match engine.search_with_metadata(&params.q, filter) {
            Ok(response) => Ok(Json(response)),
            Err(e) => {
                log::error!("Query error: {}", e);
                let (status, kind, msg) = classify_error(&e);
                Err((status, Json(json!({ "error": { "kind": kind, "message": msg } }))))
            }
        }
    }

    // GET /stats endpoint
    async fn handle_stats_endpoint(
        State(state): State<Arc<AppState>>,
    ) -> Result<Json<crate::models::IndexStats>, (StatusCode, Json<serde_json::Value>)> {
        log::info!("Stats request");

        let cache = CacheManager::new(&state.cache_path);

        if !cache.exists() {
            return Err((StatusCode::NOT_FOUND, Json(json!({ "error": { "kind": "IndexNotFound", "message": "No index found. Run 'rfx index' first." } }))));
        }

        match cache.stats() {
            Ok(stats) => Ok(Json(stats)),
            Err(e) => {
                log::error!("Stats error: {}", e);
                let (status, kind, msg) = classify_error(&e);
                Err((status, Json(json!({ "error": { "kind": kind, "message": msg } }))))
            }
        }
    }

    // POST /index endpoint
    // REF-72: accept requests with no Content-Type header (treat as empty/default request).
    async fn handle_index_endpoint(
        State(state): State<Arc<AppState>>,
        request: axum::extract::Request,
    ) -> Result<Json<crate::models::IndexStats>, (StatusCode, Json<serde_json::Value>)> {
        let has_content_type = request
            .headers()
            .contains_key(axum::http::header::CONTENT_TYPE);

        let req = if !has_content_type {
            IndexRequest::default()
        } else {
            let body_bytes = axum::body::to_bytes(request.into_body(), 1 << 20)
                .await
                .map_err(|e| (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": { "kind": "IoError", "message": format!("{}", e) } })),
                ))?;
            if body_bytes.is_empty() {
                IndexRequest::default()
            } else {
                serde_json::from_slice::<IndexRequest>(&body_bytes).map_err(|e| (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": { "kind": "ParseError", "message": format!("Invalid JSON body: {}", e) } })),
                ))?
            }
        };

        log::info!("Index request: force={}, languages={:?}", req.force, req.languages);

        let cache = CacheManager::new(&state.cache_path);

        if req.force {
            log::info!("Force rebuild requested, clearing existing cache");
            if let Err(e) = cache.clear() {
                return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": { "kind": "IoError", "message": format!("Failed to clear cache: {}", e) } }))));
            }
        }

        // Parse language filters
        let lang_filters: Vec<Language> = req.languages
            .iter()
            .filter_map(|s| match s.to_lowercase().as_str() {
                "rust" | "rs" => Some(Language::Rust),
                "python" | "py" => Some(Language::Python),
                "javascript" | "js" => Some(Language::JavaScript),
                "typescript" | "ts" => Some(Language::TypeScript),
                "vue" => Some(Language::Vue),
                "svelte" => Some(Language::Svelte),
                "go" => Some(Language::Go),
                "java" => Some(Language::Java),
                "php" => Some(Language::PHP),
                "c" => Some(Language::C),
                "cpp" | "c++" => Some(Language::Cpp),
                _ => {
                    log::warn!("Unknown language: {}", s);
                    None
                }
            })
            .collect();

        let config = IndexConfig {
            languages: lang_filters,
            ..Default::default()
        };

        let indexer = Indexer::new(cache, config);
        let path = std::path::PathBuf::from(&state.cache_path);

        match indexer.index(&path, false) {
            Ok(stats) => Ok(Json(stats)),
            Err(e) => {
                log::error!("Index error: {}", e);
                let (status, kind, msg) = classify_error(&e);
                Err((status, Json(json!({ "error": { "kind": kind, "message": msg } }))))
            }
        }
    }


    /// Classify an anyhow error into (StatusCode, kind, message) for structured JSON responses
    fn classify_error(e: &anyhow::Error) -> (axum::http::StatusCode, &'static str, String) {
        if let Some(re) = e.downcast_ref::<crate::errors::ReflexError>() {
            let status = match re {
                crate::errors::ReflexError::IndexNotFound => axum::http::StatusCode::NOT_FOUND,
                crate::errors::ReflexError::QuerySyntaxError(_) => axum::http::StatusCode::BAD_REQUEST,
                _ => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            };
            (status, re.kind(), re.to_string())
        } else {
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "IoError", e.to_string())
        }
    }

    // Health check endpoint — REF-70: returns JSON for consistency with other endpoints
    async fn handle_health() -> impl IntoResponse {
        (StatusCode::OK, Json(json!({ "status": "ok", "service": "reflex" })))
    }

    // 404 fallback — REF-71: unknown routes return JSON error body
    async fn handle_not_found() -> impl IntoResponse {
        (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": { "kind": "NotFound", "message": "Endpoint not found" } })),
        )
    }

    // Create shared state
    let state = Arc::new(AppState {
        cache_path: ".".to_string(),
    });

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build the router
    let app = Router::new()
        .route("/query", get(handle_query_endpoint))
        .route("/stats", get(handle_stats_endpoint))
        .route("/index", post(handle_index_endpoint))
        .route("/health", get(handle_health))
        .fallback(handle_not_found)  // REF-71: JSON 404 for unknown routes
        .layer(cors)
        .with_state(state);

    // Bind to the specified address
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await
        .map_err(|e| anyhow::anyhow!("Failed to bind to {}: {}", addr, e))?;

    // REF-64: warn conspicuously when bound to a non-loopback address (no auth)
    let is_loopback = host == "127.0.0.1" || host == "::1" || host.to_lowercase() == "localhost";
    if !is_loopback {
        eprintln!();
        eprintln!("WARNING: Reflex server exposed on {} — NO authentication.", addr);
        eprintln!("WARNING: Any client on this network can read your codebase index.");
        eprintln!("WARNING: Do not expose this on shared or internet-facing machines.");
        eprintln!();
    }

    log::info!("Server listening on {}", addr);

    // Run the server
    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Server error: {}", e))?;

    Ok(())
}
