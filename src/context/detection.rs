//! Project type and framework detection

use anyhow::Result;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::cache::CacheManager;

/// Detect project type and return formatted string
pub fn detect_project_type(_cache: &CacheManager, root: &Path) -> Result<String> {
    let indicators = detect_project_type_indicators(root);

    if indicators.is_empty() {
        return Ok("Unknown project type".to_string());
    }

    let mut output = Vec::new();
    output.push(format!("{}\n", indicators[0].category));

    if indicators.len() > 1 || !indicators[0].details.is_empty() {
        output.push("Indicators:".to_string());
        for indicator in &indicators {
            for detail in &indicator.details {
                output.push(format!("- {}", detail));
            }
        }
    }

    Ok(output.join("\n"))
}

/// Detect project type and return JSON
pub fn detect_project_type_json(_cache: &CacheManager, root: &Path) -> Result<Value> {
    let indicators = detect_project_type_indicators(root);

    if indicators.is_empty() {
        return Ok(json!({
            "category": "unknown",
            "indicators": []
        }));
    }

    let primary = &indicators[0];
    let all_details: Vec<String> = indicators.iter().flat_map(|i| i.details.clone()).collect();

    Ok(json!({
        "category": primary.category,
        "indicators": all_details,
    }))
}

struct ProjectIndicator {
    category: String,
    details: Vec<String>,
}

fn detect_project_type_indicators(root: &Path) -> Vec<ProjectIndicator> {
    let mut indicators = Vec::new();

    // Check for Rust project
    if root.join("Cargo.toml").exists() {
        let has_main = root.join("src/main.rs").exists();
        let has_lib = root.join("src/lib.rs").exists();

        let (category, details) = if has_main && has_lib {
            (
                "Rust CLI Tool with Library API".to_string(),
                vec![
                    "Binary entry point: src/main.rs".to_string(),
                    "Library API: src/lib.rs".to_string(),
                ],
            )
        } else if has_main {
            (
                "Rust CLI Tool".to_string(),
                vec!["Binary entry point: src/main.rs".to_string()],
            )
        } else if has_lib {
            (
                "Rust Library".to_string(),
                vec!["Library API: src/lib.rs".to_string()],
            )
        } else {
            ("Rust Project".to_string(), vec![])
        };

        indicators.push(ProjectIndicator { category, details });
    }

    // Check for JavaScript/TypeScript project
    if root.join("package.json").exists() {
        let mut details = Vec::new();
        let category;

        // Read package.json to detect framework
        if let Ok(content) = fs::read_to_string(root.join("package.json")) {
            if content.contains("\"next\"") {
                category = "Next.js Application".to_string();
                details.push("Framework: Next.js".to_string());
            } else if content.contains("\"react\"") {
                category = "React Application".to_string();
                details.push("Framework: React".to_string());
            } else if content.contains("\"vue\"") {
                category = "Vue Application".to_string();
                details.push("Framework: Vue".to_string());
            } else if content.contains("\"express\"") {
                category = "Express.js API".to_string();
                details.push("Framework: Express".to_string());
            } else if root.join("src").exists() || root.join("index.ts").exists() {
                category = "TypeScript/JavaScript Project".to_string();
            } else {
                category = "Node.js Project".to_string();
            }

            indicators.push(ProjectIndicator { category, details });
        }
    }

    // Check for Python project
    if root.join("pyproject.toml").exists()
        || root.join("setup.py").exists()
        || root.join("requirements.txt").exists()
    {
        let mut details = Vec::new();
        let category;

        if root.join("manage.py").exists() {
            category = "Django Application".to_string();
            details.push("Framework: Django".to_string());
            details.push("Entry point: manage.py".to_string());
        } else if root.join("app.py").exists() {
            category = "Flask Application".to_string();
            details.push("Entry point: app.py".to_string());
        } else if root.join("__main__.py").exists() || root.join("main.py").exists() {
            category = "Python CLI Tool".to_string();
        } else {
            category = "Python Project".to_string();
        }

        indicators.push(ProjectIndicator { category, details });
    }

    // Check for Go project
    if root.join("go.mod").exists() {
        let has_cmd = root.join("cmd").exists();
        let has_main_go = root.join("main.go").exists();

        let (category, details) = if has_cmd {
            (
                "Go CLI Tool".to_string(),
                vec!["Entry points in cmd/".to_string()],
            )
        } else if has_main_go {
            (
                "Go Application".to_string(),
                vec!["Entry point: main.go".to_string()],
            )
        } else {
            ("Go Library".to_string(), vec![])
        };

        indicators.push(ProjectIndicator { category, details });
    }

    // Check for monorepo
    if is_monorepo(root) {
        let project_count = count_subprojects(root);
        indicators.push(ProjectIndicator {
            category: format!("Monorepo ({} projects)", project_count),
            details: vec!["Multiple package files detected".to_string()],
        });
    }

    indicators
}

/// Check if this is a monorepo
fn is_monorepo(root: &Path) -> bool {
    count_subprojects(root) >= 2
}

/// Count number of subprojects (by counting package files in subdirectories)
fn count_subprojects(root: &Path) -> usize {
    let package_files = ["package.json", "Cargo.toml", "go.mod", "pyproject.toml"];
    let mut count = 0;

    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                for pkg_file in &package_files {
                    if path.join(pkg_file).exists() {
                        count += 1;
                        break;
                    }
                }
            }
        }
    }

    count
}

/// Find entry point files
pub fn find_entry_points(root: &Path) -> Result<Vec<String>> {
    let mut entry_points = Vec::new();

    // Common entry points by language
    let entry_files = [
        ("src/main.rs", "Rust binary"),
        ("src/lib.rs", "Rust library"),
        ("main.rs", "Rust binary"),
        ("index.ts", "TypeScript"),
        ("index.js", "JavaScript"),
        ("main.ts", "TypeScript"),
        ("server.ts", "TypeScript server"),
        ("app.ts", "TypeScript app"),
        ("src/index.ts", "TypeScript"),
        ("main.py", "Python"),
        ("__main__.py", "Python module"),
        ("app.py", "Python app"),
        ("manage.py", "Django"),
        ("main.go", "Go"),
    ];

    for (file, description) in &entry_files {
        let path = root.join(file);
        if path.exists() {
            if let Ok(_metadata) = fs::metadata(&path) {
                let lines = count_lines_in_file(&path).unwrap_or(0);
                entry_points.push(format!("- {} ({}, {} lines)", file, description, lines));
            }
        }
    }

    // Check for bin/ directories (Rust)
    let bin_dir = root.join("src/bin");
    if bin_dir.exists() {
        if let Ok(entries) = fs::read_dir(&bin_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let name = entry.file_name();
                entry_points.push(format!(
                    "- src/bin/{} (Rust binary)",
                    name.to_string_lossy()
                ));
            }
        }
    }

    // Check for cmd/ directories (Go)
    let cmd_dir = root.join("cmd");
    if cmd_dir.exists() {
        if let Ok(entries) = fs::read_dir(&cmd_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                if entry.path().is_dir() {
                    let name = entry.file_name();
                    entry_points.push(format!("- cmd/{} (Go binary)", name.to_string_lossy()));
                }
            }
        }
    }

    Ok(entry_points)
}

/// Find entry points (JSON format)
pub fn find_entry_points_json(root: &Path) -> Result<Value> {
    let entry_points = find_entry_points(root)?;

    let parsed: Vec<Value> = entry_points
        .iter()
        .filter_map(|ep| {
            // Parse "- path (description, N lines)" format
            let parts: Vec<&str> = ep.split(" (").collect();
            if parts.len() >= 2 {
                let path = parts[0].trim_start_matches("- ");
                let desc_lines: Vec<&str> = parts[1].trim_end_matches(')').split(", ").collect();
                let description = desc_lines[0];
                let lines = desc_lines
                    .get(1)
                    .and_then(|s| s.trim_end_matches(" lines").parse::<usize>().ok());

                Some(json!({
                    "path": path,
                    "type": description,
                    "lines": lines,
                }))
            } else {
                None
            }
        })
        .collect();

    Ok(json!(parsed))
}

/// Get file type distribution
pub fn get_file_distribution(cache: &CacheManager) -> Result<String> {
    use crate::semantic::context::CodebaseContext;

    let context = CodebaseContext::extract(cache)?;

    let mut output = Vec::new();

    // Add language breakdown
    for lang in &context.languages {
        let label = if lang.percentage > 60.0 {
            format!(
                "{} files ({:.1}%) - Primary language",
                lang.file_count, lang.percentage
            )
        } else if lang.percentage > 20.0 {
            format!("{} files ({:.1}%)", lang.file_count, lang.percentage)
        } else {
            format!("{} files ({:.1}%)", lang.file_count, lang.percentage)
        };

        output.push(format!("- {}: {}", lang.name, label));
    }

    // Add total
    let total_lines: usize = context
        .languages
        .iter()
        .map(|l| l.file_count * 50) // Rough estimate
        .sum();
    output.push(format!(
        "\nTotal: {} files, ~{} lines",
        context.total_files, total_lines
    ));

    Ok(output.join("\n"))
}

/// Get file distribution (JSON format)
pub fn get_file_distribution_json(cache: &CacheManager) -> Result<Value> {
    use crate::semantic::context::CodebaseContext;

    let context = CodebaseContext::extract(cache)?;

    let languages: Vec<Value> = context
        .languages
        .iter()
        .map(|lang| {
            json!({
                "language": lang.name,
                "count": lang.file_count,
                "percentage": lang.percentage,
            })
        })
        .collect();

    Ok(json!(languages))
}

/// Detect test layout
pub fn detect_test_layout(root: &Path) -> Result<String> {
    let mut output = Vec::new();

    // Check for test directories
    let test_dirs = ["tests", "test", "__tests__", "spec", "benches"];
    let mut found_test_dirs = Vec::new();

    for dir in &test_dirs {
        let test_path = root.join(dir);
        if test_path.exists() && test_path.is_dir() {
            let count = count_files_recursive(&test_path)?;
            found_test_dirs.push(format!("{}/ ({} files)", dir, count));
        }
    }

    // Detect test patterns
    let has_inline_tests = has_inline_tests(root)?;
    let has_separate_tests = !found_test_dirs.is_empty();

    let pattern = match (has_separate_tests, has_inline_tests) {
        (true, true) => "Separate test directory + inline test modules",
        (true, false) => "Separate test directory",
        (false, true) => "Inline test modules only",
        (false, false) => "No tests detected",
    };

    output.push(format!("Pattern: {}", pattern));

    if !found_test_dirs.is_empty() {
        output.push(format!("Test directories: {}", found_test_dirs.join(", ")));
    }

    // Count test files vs source files
    let test_file_count: usize = found_test_dirs.len();
    let src_file_count = count_files_recursive(&root.join("src")).unwrap_or(100);

    if test_file_count > 0 && src_file_count > 0 {
        let ratio = test_file_count as f64 / src_file_count as f64;
        output.push(format!("Test-to-source ratio: {:.2}", ratio));
    }

    Ok(output.join("\n"))
}

/// Detect test layout (JSON format)
pub fn detect_test_layout_json(root: &Path) -> Result<Value> {
    let has_inline = has_inline_tests(root)?;
    let test_dirs = ["tests", "test", "__tests__", "spec"];

    let mut found_dirs = Vec::new();
    let mut total_test_files = 0;

    for dir in &test_dirs {
        let path = root.join(dir);
        if path.exists() {
            let count = count_files_recursive(&path)?;
            total_test_files += count;
            found_dirs.push(format!("{}/", dir));
        }
    }

    let pattern = match (!found_dirs.is_empty(), has_inline) {
        (true, true) => "separate_directory_plus_inline",
        (true, false) => "separate_directory",
        (false, true) => "inline_only",
        (false, false) => "none",
    };

    let src_files = count_files_recursive(&root.join("src")).unwrap_or(100);
    let ratio = if src_files > 0 {
        total_test_files as f64 / src_files as f64
    } else {
        0.0
    };

    Ok(json!({
        "pattern": pattern,
        "test_files": total_test_files,
        "test_directories": found_dirs,
        "test_to_source_ratio": ratio,
    }))
}

/// Check if project has inline tests (e.g., #[cfg(test)] in Rust)
fn has_inline_tests(root: &Path) -> Result<bool> {
    // Simple heuristic: check if any .rs files contain #[cfg(test)]
    let src_dir = root.join("src");
    if !src_dir.exists() {
        return Ok(false);
    }

    if let Ok(entries) = fs::read_dir(&src_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                if let Ok(content) = fs::read_to_string(&path) {
                    if content.contains("#[cfg(test)]") || content.contains("#[test]") {
                        return Ok(true);
                    }
                }
            }
        }
    }

    Ok(false)
}

/// Detect frameworks
pub fn detect_frameworks(root: &Path) -> Result<String> {
    let frameworks = detect_frameworks_list(root)?;

    if frameworks.is_empty() {
        return Ok("No frameworks detected".to_string());
    }

    let output: Vec<String> = frameworks
        .iter()
        .map(|(name, category)| format!("- {}: {}", category, name))
        .collect();

    Ok(output.join("\n"))
}

/// Detect frameworks (JSON format)
pub fn detect_frameworks_json(root: &Path) -> Result<Value> {
    let frameworks = detect_frameworks_list(root)?;

    let json_frameworks: Vec<Value> = frameworks
        .iter()
        .map(|(name, category)| {
            json!({
                "name": name,
                "category": category,
            })
        })
        .collect();

    Ok(json!(json_frameworks))
}

fn detect_frameworks_list(root: &Path) -> Result<Vec<(String, String)>> {
    let mut frameworks = Vec::new();

    detect_rust_frameworks(root, &mut frameworks);
    detect_js_ts_frameworks(root, &mut frameworks);
    detect_python_frameworks(root, &mut frameworks);
    detect_php_frameworks(root, &mut frameworks);
    detect_go_frameworks(root, &mut frameworks);
    detect_java_frameworks(root, &mut frameworks);
    detect_csharp_frameworks(root, &mut frameworks);
    detect_ruby_frameworks(root, &mut frameworks);
    detect_kotlin_frameworks(root, &mut frameworks);
    detect_c_cpp_frameworks(root, &mut frameworks);
    detect_zig_frameworks(root, &mut frameworks);

    Ok(frameworks)
}

/// Detect Rust frameworks from Cargo.toml
fn detect_rust_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    let cargo_toml = root.join("Cargo.toml");
    if !cargo_toml.exists() {
        return;
    }

    if let Ok(content) = fs::read_to_string(&cargo_toml) {
        // Async runtimes
        if content.contains("tokio") {
            frameworks.push(("tokio".to_string(), "Async Runtime".to_string()));
        }
        if content.contains("async-std") {
            frameworks.push(("async-std".to_string(), "Async Runtime".to_string()));
        }

        // Web frameworks
        if content.contains("axum") {
            frameworks.push(("axum".to_string(), "Web Framework".to_string()));
        }
        if content.contains("actix-web") {
            frameworks.push(("actix-web".to_string(), "Web Framework".to_string()));
        }
        if content.contains("rocket") {
            frameworks.push(("Rocket".to_string(), "Web Framework".to_string()));
        }
        if content.contains("warp") {
            frameworks.push(("Warp".to_string(), "Web Framework".to_string()));
        }

        // CLI frameworks
        if content.contains("clap") {
            frameworks.push(("clap".to_string(), "CLI Framework".to_string()));
        }

        // ORMs
        if content.contains("diesel") {
            frameworks.push(("Diesel".to_string(), "ORM".to_string()));
        }
        if content.contains("sqlx") {
            frameworks.push(("SQLx".to_string(), "ORM".to_string()));
        }
        if content.contains("sea-orm") {
            frameworks.push(("SeaORM".to_string(), "ORM".to_string()));
        }

        // Testing
        if content.contains("criterion") {
            frameworks.push(("Criterion".to_string(), "Benchmarking".to_string()));
        }
    }
}

/// Detect JavaScript/TypeScript frameworks from package.json
fn detect_js_ts_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    let package_json = root.join("package.json");
    if !package_json.exists() {
        return;
    }

    if let Ok(content) = fs::read_to_string(&package_json) {
        // Meta-frameworks (check these first as they may include base frameworks)
        if content.contains("\"next\"") {
            frameworks.push(("Next.js".to_string(), "Web Framework".to_string()));
        }
        if content.contains("\"nuxt\"") {
            frameworks.push(("Nuxt".to_string(), "Vue Framework".to_string()));
        }
        if content.contains("\"@sveltejs/kit\"") {
            frameworks.push(("SvelteKit".to_string(), "Svelte Framework".to_string()));
        }
        if content.contains("\"@remix-run/react\"") {
            frameworks.push(("Remix".to_string(), "Web Framework".to_string()));
        }
        if content.contains("\"astro\"") {
            frameworks.push(("Astro".to_string(), "Web Framework".to_string()));
        }

        // UI libraries/frameworks
        if content.contains("\"react\"") {
            frameworks.push(("React".to_string(), "UI Library".to_string()));
        }
        if content.contains("\"vue\"") {
            frameworks.push(("Vue".to_string(), "UI Framework".to_string()));
        }
        if content.contains("\"svelte\"") {
            frameworks.push(("Svelte".to_string(), "UI Framework".to_string()));
        }
        if content.contains("\"@angular/core\"") {
            frameworks.push(("Angular".to_string(), "Web Framework".to_string()));
        }

        // Backend frameworks
        if content.contains("\"express\"") {
            frameworks.push(("Express".to_string(), "Web Framework".to_string()));
        }
        if content.contains("\"@nestjs/core\"") {
            frameworks.push(("NestJS".to_string(), "Web Framework".to_string()));
        }
        if content.contains("\"koa\"") {
            frameworks.push(("Koa".to_string(), "Web Framework".to_string()));
        }
        if content.contains("\"fastify\"") {
            frameworks.push(("Fastify".to_string(), "Web Framework".to_string()));
        }

        // Testing frameworks
        if content.contains("\"jest\"") {
            frameworks.push(("Jest".to_string(), "Testing Framework".to_string()));
        }
        if content.contains("\"vitest\"") {
            frameworks.push(("Vitest".to_string(), "Testing Framework".to_string()));
        }
        if content.contains("\"@playwright/test\"") {
            frameworks.push(("Playwright".to_string(), "E2E Testing".to_string()));
        }
        if content.contains("\"cypress\"") {
            frameworks.push(("Cypress".to_string(), "E2E Testing".to_string()));
        }

        // Build tools
        if content.contains("\"vite\"") {
            frameworks.push(("Vite".to_string(), "Build Tool".to_string()));
        }
    }
}

/// Detect Python frameworks from requirements.txt and pyproject.toml
fn detect_python_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    let reqs_files = ["requirements.txt", "pyproject.toml"];

    for file in &reqs_files {
        let path = root.join(file);
        if !path.exists() {
            continue;
        }

        if let Ok(content) = fs::read_to_string(&path) {
            // Web frameworks
            if content.contains("django") {
                frameworks.push(("Django".to_string(), "Web Framework".to_string()));
            }
            if content.contains("flask") {
                frameworks.push(("Flask".to_string(), "Web Framework".to_string()));
            }
            if content.contains("fastapi") {
                frameworks.push(("FastAPI".to_string(), "Web Framework".to_string()));
            }
            if content.contains("tornado") {
                frameworks.push(("Tornado".to_string(), "Web Framework".to_string()));
            }

            // Testing
            if content.contains("pytest") {
                frameworks.push(("pytest".to_string(), "Testing Framework".to_string()));
            }

            // ORMs
            if content.contains("sqlalchemy") {
                frameworks.push(("SQLAlchemy".to_string(), "ORM".to_string()));
            }

            // CLI
            if content.contains("click") {
                frameworks.push(("Click".to_string(), "CLI Framework".to_string()));
            }
            if content.contains("typer") {
                frameworks.push(("Typer".to_string(), "CLI Framework".to_string()));
            }
        }
    }
}

/// Detect PHP frameworks from composer.json
fn detect_php_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    let composer_json = root.join("composer.json");
    if !composer_json.exists() {
        return;
    }

    if let Ok(content) = fs::read_to_string(&composer_json) {
        // Web frameworks
        if content.contains("\"laravel/framework\"") {
            frameworks.push(("Laravel".to_string(), "Web Framework".to_string()));
        }
        if content.contains("\"symfony/symfony\"") {
            frameworks.push(("Symfony".to_string(), "Web Framework".to_string()));
        }
        if content.contains("\"slim/slim\"") {
            frameworks.push(("Slim".to_string(), "Web Framework".to_string()));
        }
        if content.contains("\"cakephp/cakephp\"") {
            frameworks.push(("CakePHP".to_string(), "Web Framework".to_string()));
        }

        // Testing
        if content.contains("\"phpunit/phpunit\"") {
            frameworks.push(("PHPUnit".to_string(), "Testing Framework".to_string()));
        }
        if content.contains("\"pestphp/pest\"") {
            frameworks.push(("Pest".to_string(), "Testing Framework".to_string()));
        }

        // ORM
        if content.contains("\"doctrine/orm\"") {
            frameworks.push(("Doctrine ORM".to_string(), "ORM".to_string()));
        }
    }
}

/// Detect Go frameworks from go.mod
fn detect_go_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    let go_mod = root.join("go.mod");
    if !go_mod.exists() {
        return;
    }

    if let Ok(content) = fs::read_to_string(&go_mod) {
        // Web frameworks
        if content.contains("gin-gonic/gin") {
            frameworks.push(("Gin".to_string(), "Web Framework".to_string()));
        }
        if content.contains("labstack/echo") {
            frameworks.push(("Echo".to_string(), "Web Framework".to_string()));
        }
        if content.contains("gofiber/fiber") {
            frameworks.push(("Fiber".to_string(), "Web Framework".to_string()));
        }
        if content.contains("go-chi/chi") {
            frameworks.push(("Chi".to_string(), "Web Framework".to_string()));
        }
        if content.contains("gorilla/mux") {
            frameworks.push(("Gorilla Mux".to_string(), "Web Framework".to_string()));
        }

        // CLI frameworks
        if content.contains("spf13/cobra") {
            frameworks.push(("Cobra".to_string(), "CLI Framework".to_string()));
        }
        if content.contains("urfave/cli") {
            frameworks.push(("urfave/cli".to_string(), "CLI Framework".to_string()));
        }

        // ORM
        if content.contains("go-gorm/gorm") || content.contains("gorm.io/gorm") {
            frameworks.push(("GORM".to_string(), "ORM".to_string()));
        }

        // Testing
        if content.contains("stretchr/testify") {
            frameworks.push(("Testify".to_string(), "Testing Framework".to_string()));
        }
    }
}

/// Detect Java frameworks from pom.xml and build.gradle
fn detect_java_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    // Check pom.xml
    let pom_xml = root.join("pom.xml");
    if pom_xml.exists() {
        if let Ok(content) = fs::read_to_string(&pom_xml) {
            detect_java_frameworks_from_content(&content, frameworks);
        }
    }

    // Check build.gradle
    let build_gradle = root.join("build.gradle");
    if build_gradle.exists() {
        if let Ok(content) = fs::read_to_string(&build_gradle) {
            detect_java_frameworks_from_content(&content, frameworks);
        }
    }

    // Check build.gradle.kts
    let build_gradle_kts = root.join("build.gradle.kts");
    if build_gradle_kts.exists() {
        if let Ok(content) = fs::read_to_string(&build_gradle_kts) {
            detect_java_frameworks_from_content(&content, frameworks);
        }
    }
}

fn detect_java_frameworks_from_content(content: &str, frameworks: &mut Vec<(String, String)>) {
    // Web frameworks
    if content.contains("spring-boot") {
        frameworks.push(("Spring Boot".to_string(), "Web Framework".to_string()));
    }
    if content.contains("quarkus") {
        frameworks.push(("Quarkus".to_string(), "Web Framework".to_string()));
    }
    if content.contains("micronaut") {
        frameworks.push(("Micronaut".to_string(), "Web Framework".to_string()));
    }

    // Testing
    if content.contains("junit-jupiter") {
        frameworks.push(("JUnit 5".to_string(), "Testing Framework".to_string()));
    } else if content.contains("junit") {
        frameworks.push(("JUnit".to_string(), "Testing Framework".to_string()));
    }
    if content.contains("mockito") {
        frameworks.push(("Mockito".to_string(), "Testing Framework".to_string()));
    }

    // ORM
    if content.contains("hibernate") {
        frameworks.push(("Hibernate".to_string(), "ORM".to_string()));
    }
}

/// Detect C# frameworks from .csproj files
fn detect_csharp_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("csproj") {
                if let Ok(content) = fs::read_to_string(&path) {
                    // Web frameworks
                    if content.contains("Microsoft.AspNetCore") {
                        frameworks.push(("ASP.NET Core".to_string(), "Web Framework".to_string()));
                    }

                    // Testing
                    if content.contains("xUnit") || content.contains("xunit") {
                        frameworks.push(("xUnit".to_string(), "Testing Framework".to_string()));
                    }
                    if content.contains("NUnit") {
                        frameworks.push(("NUnit".to_string(), "Testing Framework".to_string()));
                    }
                    if content.contains("MSTest") {
                        frameworks.push(("MSTest".to_string(), "Testing Framework".to_string()));
                    }

                    // ORM
                    if content.contains("EntityFrameworkCore") {
                        frameworks.push(("Entity Framework Core".to_string(), "ORM".to_string()));
                    }
                }
                break; // Only check first .csproj
            }
        }
    }
}

/// Detect Ruby frameworks from Gemfile
fn detect_ruby_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    let gemfile = root.join("Gemfile");
    if !gemfile.exists() {
        return;
    }

    if let Ok(content) = fs::read_to_string(&gemfile) {
        // Web frameworks
        if content.contains("gem 'rails'") || content.contains("gem \"rails\"") {
            frameworks.push(("Rails".to_string(), "Web Framework".to_string()));
        }
        if content.contains("gem 'sinatra'") || content.contains("gem \"sinatra\"") {
            frameworks.push(("Sinatra".to_string(), "Web Framework".to_string()));
        }
        if content.contains("gem 'hanami'") || content.contains("gem \"hanami\"") {
            frameworks.push(("Hanami".to_string(), "Web Framework".to_string()));
        }

        // Testing
        if content.contains("gem 'rspec'") || content.contains("gem \"rspec\"") {
            frameworks.push(("RSpec".to_string(), "Testing Framework".to_string()));
        }
        if content.contains("gem 'minitest'") || content.contains("gem \"minitest\"") {
            frameworks.push(("Minitest".to_string(), "Testing Framework".to_string()));
        }

        // Background jobs
        if content.contains("gem 'sidekiq'") || content.contains("gem \"sidekiq\"") {
            frameworks.push(("Sidekiq".to_string(), "Background Jobs".to_string()));
        }
    }
}

/// Detect Kotlin frameworks from build.gradle.kts
fn detect_kotlin_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    let build_gradle_kts = root.join("build.gradle.kts");
    if !build_gradle_kts.exists() {
        return;
    }

    if let Ok(content) = fs::read_to_string(&build_gradle_kts) {
        // Web frameworks
        if content.contains("ktor") {
            frameworks.push(("Ktor".to_string(), "Web Framework".to_string()));
        }

        // Testing
        if content.contains("kotest") {
            frameworks.push(("Kotest".to_string(), "Testing Framework".to_string()));
        }
        if content.contains("mockk") {
            frameworks.push(("MockK".to_string(), "Testing Framework".to_string()));
        }

        // Coroutines
        if content.contains("kotlinx-coroutines") {
            frameworks.push(("Kotlin Coroutines".to_string(), "Async Runtime".to_string()));
        }
    }
}

/// Detect C/C++ frameworks from CMakeLists.txt and vcpkg.json
fn detect_c_cpp_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    // Check CMakeLists.txt
    let cmake_lists = root.join("CMakeLists.txt");
    if cmake_lists.exists() {
        if let Ok(content) = fs::read_to_string(&cmake_lists) {
            // Testing
            if content.contains("GTest") || content.contains("gtest") {
                frameworks.push(("Google Test".to_string(), "Testing Framework".to_string()));
            }
            if content.contains("Catch2") {
                frameworks.push(("Catch2".to_string(), "Testing Framework".to_string()));
            }

            // Libraries
            if content.contains("Boost") {
                frameworks.push(("Boost".to_string(), "C++ Libraries".to_string()));
            }

            // GUI
            if content.contains("Qt") || content.contains("qt") {
                frameworks.push(("Qt".to_string(), "GUI Framework".to_string()));
            }
            if content.contains("wxWidgets") {
                frameworks.push(("wxWidgets".to_string(), "GUI Framework".to_string()));
            }
        }
    }

    // Check vcpkg.json
    let vcpkg_json = root.join("vcpkg.json");
    if vcpkg_json.exists() {
        if let Ok(content) = fs::read_to_string(&vcpkg_json) {
            if content.contains("\"gtest\"") {
                frameworks.push(("Google Test".to_string(), "Testing Framework".to_string()));
            }
            if content.contains("\"catch2\"") {
                frameworks.push(("Catch2".to_string(), "Testing Framework".to_string()));
            }
            if content.contains("\"boost\"") {
                frameworks.push(("Boost".to_string(), "C++ Libraries".to_string()));
            }
        }
    }
}

/// Detect Zig frameworks from build.zig
fn detect_zig_frameworks(root: &Path, frameworks: &mut Vec<(String, String)>) {
    let build_zig = root.join("build.zig");
    if !build_zig.exists() {
        return;
    }

    if let Ok(content) = fs::read_to_string(&build_zig) {
        // Web frameworks (limited ecosystem)
        if content.contains("zap") {
            frameworks.push(("Zap".to_string(), "Web Framework".to_string()));
        }
        if content.contains("zhp") {
            frameworks.push(("ZHP".to_string(), "Web Framework".to_string()));
        }
    }
}

/// Find configuration files
pub fn find_config_files(root: &Path) -> Result<String> {
    let configs = find_config_files_list(root)?;

    if configs.is_empty() {
        return Ok("No configuration files found".to_string());
    }

    // Group by category
    let mut grouped: HashMap<String, Vec<String>> = HashMap::new();
    for (path, category) in configs {
        grouped.entry(category).or_default().push(path);
    }

    let mut output = Vec::new();
    for (category, files) in grouped {
        output.push(format!("{}:", category));
        for file in files {
            output.push(format!("- {}", file));
        }
        output.push(String::new()); // Blank line
    }

    Ok(output.join("\n"))
}

/// Find configuration files (JSON format)
pub fn find_config_files_json(root: &Path) -> Result<Value> {
    let configs = find_config_files_list(root)?;

    let json_configs: Vec<Value> = configs
        .iter()
        .map(|(path, category)| {
            json!({
                "path": path,
                "category": category,
            })
        })
        .collect();

    Ok(json!(json_configs))
}

fn find_config_files_list(root: &Path) -> Result<Vec<(String, String)>> {
    let mut configs = Vec::new();

    // Project manifests
    let manifests = [
        ("Cargo.toml", "Project Manifest"),
        ("package.json", "Project Manifest"),
        ("pyproject.toml", "Project Manifest"),
        ("go.mod", "Project Manifest"),
        ("pom.xml", "Project Manifest"),
        ("build.gradle", "Project Manifest"),
    ];

    for (file, category) in &manifests {
        if root.join(file).exists() {
            configs.push((file.to_string(), category.to_string()));
        }
    }

    // Tool configuration
    let tool_configs = [
        (".gitignore", "Version Control"),
        (".gitattributes", "Version Control"),
        ("rustfmt.toml", "Code Formatting"),
        (".prettierrc", "Code Formatting"),
        (".eslintrc", "Code Linting"),
        ("tsconfig.json", "TypeScript Config"),
        (".reflex/config.toml", "Tool Config"),
    ];

    for (file, category) in &tool_configs {
        if root.join(file).exists() {
            configs.push((file.to_string(), category.to_string()));
        }
    }

    // Documentation
    let docs = [
        ("README.md", "Documentation"),
        ("CLAUDE.md", "Documentation"),
        ("CONTRIBUTING.md", "Documentation"),
        ("LICENSE", "Documentation"),
    ];

    for (file, category) in &docs {
        if root.join(file).exists() {
            configs.push((file.to_string(), category.to_string()));
        }
    }

    Ok(configs)
}

/// Count lines in a file
fn count_lines_in_file(path: &Path) -> Result<usize> {
    let content = fs::read_to_string(path)?;
    Ok(content.lines().count())
}

/// Count files recursively in a directory
fn count_files_recursive(dir: &Path) -> Result<usize> {
    let mut count = 0;

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                count += count_files_recursive(&path)?;
            } else {
                count += 1;
            }
        }
    }

    Ok(count)
}
