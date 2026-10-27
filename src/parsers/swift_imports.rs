//! Swift imports and SwiftPM targets, read from the source text.
//!
//! This build has no tree-sitter grammar for Swift (`swift.rs` is disabled), so an
//! import is read from the lines of the source after its comments and string
//! literals are blanked. An import declaration is
//! `attributes? access-level? import kind? Module(.Name)*`.
//!
//! A module is a SwiftPM target of the workspace, an Apple SDK framework or a
//! toolchain module. A target's files are one directory: the target's `path:`,
//! else `Sources/<name>` (`Tests/<name>` for a test target, `Plugins/<name>` for a
//! plugin). So an import of a target resolves to the package key `swift:<dir>`,
//! and a `.swift` file is a member of the key of the target whose directory holds
//! it. Like a Go package, a target compiles as one unit: its files use each other
//! with no import.

use crate::models::ImportType;
use crate::parsers::{DependencyExtractor, ImportInfo};
use anyhow::Result;
use regex::Regex;
use std::sync::LazyLock;

/// Toolchain modules, Apple SDK frameworks and the SDKs' C modules: `Stdlib`. A
/// workspace target of the same name still wins (see `ResolverContext::reclassify`).
const SYSTEM_MODULES: &[&str] = &[
    // The toolchain
    "Swift",
    "_Concurrency",
    "_StringProcessing",
    "_Differentiation",
    "_Backtracing",
    "RegexBuilder",
    "Observation",
    "Synchronization",
    "Distributed",
    "Cxx",
    "CxxStdlib",
    "Foundation",
    "FoundationEssentials",
    "FoundationInternationalization",
    "FoundationNetworking",
    "FoundationXML",
    "Dispatch",
    "XCTest",
    "Testing",
    "PackageDescription",
    "PackagePlugin",
    "CompilerPluginSupport",
    // C libraries of each platform
    "Darwin",
    "Glibc",
    "Musl",
    "Bionic",
    "Android",
    "WASILibc",
    "WinSDK",
    "ucrt",
    "CRT",
    "os",
    "ObjectiveC",
    "simd",
    // Clang modules of the SDKs (`import C.os.lock`, `import zlib`)
    "C",
    "zlib",
    "SQLite3",
    "CommonCrypto",
    "MachO",
    "libkern",
    "SwiftShims",
    // Apple SDK frameworks
    "Accelerate",
    "Accessibility",
    "ActivityKit",
    "AdServices",
    "AdSupport",
    "AppClip",
    "AppIntents",
    "AppKit",
    "AppTrackingTransparency",
    "ApplicationServices",
    "ARKit",
    "AssetsLibrary",
    "AudioToolbox",
    "AudioUnit",
    "AuthenticationServices",
    "AVFAudio",
    "AVFoundation",
    "AVKit",
    "AVRouting",
    "BackgroundAssets",
    "BackgroundTasks",
    "CallKit",
    "Carbon",
    "CarPlay",
    "CFNetwork",
    "Charts",
    "ClassKit",
    "ClockKit",
    "CloudKit",
    "Cocoa",
    "Combine",
    "Contacts",
    "ContactsUI",
    "CoreAudio",
    "CoreAudioKit",
    "CoreAudioTypes",
    "CoreBluetooth",
    "CoreData",
    "CoreFoundation",
    "CoreGraphics",
    "CoreHaptics",
    "CoreImage",
    "CoreLocation",
    "CoreLocationUI",
    "CoreMedia",
    "CoreMIDI",
    "CoreML",
    "CoreMotion",
    "CoreNFC",
    "CoreServices",
    "CoreSpotlight",
    "CoreTelephony",
    "CoreText",
    "CoreTransferable",
    "CoreVideo",
    "CoreWLAN",
    "CreateML",
    "CreateMLComponents",
    "CryptoKit",
    "CryptoTokenKit",
    "DataDetection",
    "DeveloperToolsSupport",
    "DeviceActivity",
    "DeviceCheck",
    "DiskArbitration",
    "EventKit",
    "EventKitUI",
    "ExtensionFoundation",
    "ExtensionKit",
    "ExternalAccessory",
    "FamilyControls",
    "FileProvider",
    "FileProviderUI",
    "FinanceKit",
    "GameController",
    "GameKit",
    "GameplayKit",
    "GLKit",
    "GroupActivities",
    "HealthKit",
    "HealthKitUI",
    "HomeKit",
    "IdentityLookup",
    "ImageCaptureCore",
    "ImageIO",
    "Intents",
    "IntentsUI",
    "IOKit",
    "IOSurface",
    "JavaScriptCore",
    "LinkPresentation",
    "LocalAuthentication",
    "ManagedSettings",
    "MapKit",
    "MediaPlayer",
    "MediaToolbox",
    "MessageUI",
    "Messages",
    "Metal",
    "MetalFX",
    "MetalKit",
    "MetalPerformanceShaders",
    "MetalPerformanceShadersGraph",
    "MLCompute",
    "MobileCoreServices",
    "ModelIO",
    "MultipeerConnectivity",
    "MusicKit",
    "NaturalLanguage",
    "NearbyInteraction",
    "Network",
    "NetworkExtension",
    "NotificationCenter",
    "OpenGLES",
    "OSLog",
    "PassKit",
    "PDFKit",
    "PencilKit",
    "Photos",
    "PhotosUI",
    "PushKit",
    "QuartzCore",
    "QuickLook",
    "QuickLookThumbnailing",
    "RealityKit",
    "ReplayKit",
    "SafariServices",
    "SceneKit",
    "ScreenCaptureKit",
    "ScreenTime",
    "Security",
    "SensitiveContentAnalysis",
    "SharedWithYou",
    "ShazamKit",
    "Social",
    "SoundAnalysis",
    "Speech",
    "SpriteKit",
    "StoreKit",
    "SwiftData",
    "SwiftUI",
    "SystemConfiguration",
    "TabularData",
    "TipKit",
    "TVMLKit",
    "TVServices",
    "TVUIKit",
    "UIKit",
    "UniformTypeIdentifiers",
    "UserNotifications",
    "UserNotificationsUI",
    "VideoToolbox",
    "Vision",
    "VisionKit",
    "WatchConnectivity",
    "WatchKit",
    "WeatherKit",
    "WebKit",
    "WidgetKit",
    "WorkoutKit",
    "XPC",
];

static IMPORT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?P<attrs>(?:@\w+(?:\([^)]*\))?\s*)*)(?:(?:public|package|internal|fileprivate|private)\s+)?import\s+(?:(?P<kind>typealias|struct|class|enum|protocol|let|var|func)\s+)?(?P<path>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)",
    )
    .expect("valid regex")
});

static TARGET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"\.(?P<kind>target|executableTarget|testTarget|macro|plugin|systemLibrary)\s*\(\s*name\s*:\s*"(?P<name>[^"\\]+)""#,
    )
    .expect("valid regex")
});

static PATH_ARG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"^path\s*:\s*"([^"\\]*)""#).expect("valid regex"));

pub struct SwiftDependencyExtractor;

impl DependencyExtractor for SwiftDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let code = blanked(source, &lex(source.as_bytes()), false);
        let mut imports = Vec::new();
        for (line_idx, line) in code.lines().enumerate() {
            for statement in line.split(';') {
                let Some(caps) = IMPORT.captures(statement.trim()) else {
                    continue;
                };
                let path = caps["path"].to_string();
                let testable = caps["attrs"].contains("@testable");
                // `import struct Foundation.URL` brings in one declaration
                let imported_symbols = caps
                    .name("kind")
                    .and_then(|_| path.rsplit_once('.'))
                    .map(|(_, name)| vec![name.to_string()]);
                imports.push(ImportInfo {
                    import_type: classify(swift_module_name(&path), testable),
                    imported_path: path,
                    line_number: line_idx + 1,
                    imported_symbols,
                });
            }
        }
        Ok(imports)
    }
}

/// The module an import names: `Foundation` of `Foundation.URL`, `UIKit` of the
/// submodule `UIKit.UIGestureRecognizerSubclass`.
pub fn swift_module_name(imported_path: &str) -> &str {
    imported_path.split('.').next().unwrap_or(imported_path)
}

/// `Stdlib` for a toolchain module or SDK framework, `Internal` for an
/// `@testable` import (only a module the workspace builds can be one), else
/// `External` until a workspace target of that name makes it `Internal`.
fn classify(module: &str, testable: bool) -> ImportType {
    if SYSTEM_MODULES.contains(&module) {
        ImportType::Stdlib
    } else if testable {
        ImportType::Internal
    } else {
        ImportType::External
    }
}

/// A target of a `Package.swift`: its module name and its source directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwiftModule {
    /// Directory of the `Package.swift`, relative to the index root (`""` at it).
    pub package_dir: String,
    pub name: String,
    /// Directory of the target's sources, relative to the index root.
    pub dir: String,
    /// An executable, test, plugin or macro target: a root of the import graph,
    /// whose `@main` file need not be `main.swift`.
    pub entry: bool,
}

/// The `package_members` member of every file of an entry target, beside its
/// `""` row. No import names it, so it adds no edge; `analyze` reads it.
pub const ENTRY_MEMBER: &str = "@entry";

impl SwiftModule {
    /// The package key an import of this target resolves to.
    pub fn package_key(&self) -> String {
        format!("swift:{}", self.dir)
    }
}

/// The targets of the `Package.swift` at `rel` (relative to the index root):
/// each `.target`, `.executableTarget`, `.testTarget`, `.macro`, `.plugin` or
/// `.systemLibrary` call whose first argument is `name: "…"`, with the call's own
/// `path:` argument or SwiftPM's default directory. A call inside another one is
/// a dependency (`dependencies: [.target(name: "Core")]`), not a target. A text
/// scan.
pub fn parse_package_swift(rel: &str, source: &str) -> Vec<SwiftModule> {
    let package_dir = crate::dependency_resolve::parent_dir(rel);
    let mask = lex(source.as_bytes());
    let text = blanked(source, &mask, true);
    let bytes = text.as_bytes();
    let mut modules = Vec::new();
    let mut previous_end = 0;
    for caps in TARGET.captures_iter(&text) {
        let call = caps.get(0).expect("whole match");
        if mask[call.start()] != CODE || call.start() < previous_end {
            continue;
        }
        let name = &caps["name"];
        // The call's own arguments are at depth 1 from its `(`
        let open = call.start() + text[call.start()..].find('(').expect("matched `(`");
        let mut depth = 0usize;
        let mut path = None;
        previous_end = bytes.len();
        for j in open..bytes.len() {
            if mask[j] != CODE {
                continue;
            }
            match bytes[j] {
                b'(' | b'[' => depth += 1,
                b')' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        previous_end = j;
                        break;
                    }
                }
                b'p' if depth == 1
                    && path.is_none()
                    && !bytes[j - 1].is_ascii_alphanumeric()
                    && bytes[j - 1] != b'_' =>
                {
                    path = PATH_ARG.captures(&text[j..]).map(|c| c[1].to_string());
                }
                _ => {}
            }
        }
        let kind = &caps["kind"];
        let path = path.unwrap_or_else(|| {
            let parent = match kind {
                "testTarget" => "Tests",
                "plugin" => "Plugins",
                _ => "Sources",
            };
            format!("{parent}/{name}")
        });
        if let Some(dir) = crate::dependency_resolve::fold_path(package_dir, &path) {
            modules.push(SwiftModule {
                package_dir: package_dir.to_string(),
                name: name.to_string(),
                dir,
                entry: matches!(kind, "executableTarget" | "testTarget" | "plugin" | "macro"),
            });
        }
    }
    modules
}

/// The target `name` means in the file at `rel`: the one in the nearest
/// `Package.swift` above the file, else the first one found (a local package
/// dependency such as `.package(path: "../Core")`).
pub fn find_swift_module<'a>(
    name: &str,
    rel: &str,
    modules: &'a [SwiftModule],
) -> Option<&'a SwiftModule> {
    let named = || modules.iter().filter(move |m| m.name == name);
    named()
        .filter(|m| m.package_dir.is_empty() || rel.starts_with(&format!("{}/", m.package_dir)))
        .max_by_key(|m| m.package_dir.len())
        .or_else(|| named().next())
}

/// The target whose sources hold the file at `rel`: the deepest target
/// directory above it.
pub fn swift_target_of<'a>(rel: &str, modules: &'a [SwiftModule]) -> Option<&'a SwiftModule> {
    modules
        .iter()
        .filter(|m| rel.starts_with(&format!("{}/", m.dir)))
        .max_by_key(|m| m.dir.len())
}

const CODE: u8 = 0;
const COMMENT: u8 = 1;
const STRING: u8 = 2;

/// `source` with comments (and, unless `keep_strings`, string literals) replaced
/// by spaces. Newlines stay, so line numbers and byte offsets do not move.
fn blanked(source: &str, mask: &[u8], keep_strings: bool) -> String {
    let bytes: Vec<u8> = source
        .bytes()
        .zip(mask)
        .map(|(b, &kind)| match kind {
            CODE => b,
            STRING if keep_strings => b,
            _ if b == b'\n' => b,
            _ => b' ',
        })
        .collect();
    // Every blanked run starts and ends at an ASCII byte, so this is UTF-8
    String::from_utf8_lossy(&bytes).into_owned()
}

/// What each byte of `source` is: code, comment or string literal. String
/// interpolations count as string. Swift's delimiters are ASCII, so bytes do.
fn lex(source: &[u8]) -> Vec<u8> {
    let mut mask = vec![CODE; source.len()];
    let mut i = 0;
    lex_code(source, &mut mask, &mut i, false);
    mask
}

/// Code from `*i`. In an interpolation, stops after the `)` that closes it.
fn lex_code(s: &[u8], mask: &mut [u8], i: &mut usize, interpolation: bool) {
    let mut depth = 0usize;
    while *i < s.len() {
        let start = *i;
        match s[*i] {
            b'/' if s.get(*i + 1) == Some(&b'/') => {
                while *i < s.len() && s[*i] != b'\n' {
                    *i += 1;
                }
                mask[start..*i].fill(COMMENT);
            }
            b'/' if s.get(*i + 1) == Some(&b'*') => {
                // Block comments nest in Swift
                let mut nesting = 0usize;
                while *i < s.len() {
                    if s[*i..].starts_with(b"/*") {
                        nesting += 1;
                        *i += 2;
                    } else if s[*i..].starts_with(b"*/") {
                        nesting -= 1;
                        *i += 2;
                        if nesting == 0 {
                            break;
                        }
                    } else {
                        *i += 1;
                    }
                }
                mask[start..*i].fill(COMMENT);
            }
            b'"' | b'#' if starts_string(s, *i) => {
                lex_string(s, mask, i);
                mask[start..*i].fill(STRING);
            }
            b'(' => {
                depth += 1;
                *i += 1;
            }
            b')' => {
                *i += 1;
                if depth == 0 {
                    if interpolation {
                        return;
                    }
                } else {
                    depth -= 1;
                }
            }
            _ => *i += 1,
        }
    }
}

/// Whether a string literal starts at `i`: `"`, or `#`s then `"` (a raw string).
fn starts_string(s: &[u8], i: usize) -> bool {
    let hashes = s[i..].iter().take_while(|&&b| b == b'#').count();
    s.get(i + hashes) == Some(&b'"')
}

/// Moves `*i` past the string literal there: `"…"`, `"""…"""`, or either with
/// `#` delimiters (a raw string, whose escape is `\#`).
fn lex_string(s: &[u8], mask: &mut [u8], i: &mut usize) {
    let hashes = s[*i..].iter().take_while(|&&b| b == b'#').count();
    *i += hashes;
    let quotes = if s[*i..].starts_with(b"\"\"\"") { 3 } else { 1 };
    *i += quotes;
    let delimited = |at: usize| (1..=hashes).all(|k| s.get(at + k) == Some(&b'#'));
    while *i < s.len() {
        if s[*i] == b'\\' && delimited(*i) {
            *i += 1 + hashes;
            if s.get(*i) == Some(&b'(') {
                *i += 1;
                lex_code(s, mask, i, true);
            } else {
                *i += 1;
            }
        } else if s[*i..].starts_with(&b"\"\"\""[..quotes])
            && (0..hashes).all(|k| s.get(*i + quotes + k) == Some(&b'#'))
        {
            *i += quotes + hashes;
            return;
        } else if s[*i] == b'\n' && quotes == 1 {
            // An unterminated literal ends at its line
            return;
        } else {
            *i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn imports(source: &str) -> Vec<(String, ImportType, usize, Option<Vec<String>>)> {
        SwiftDependencyExtractor::extract_dependencies(source)
            .unwrap()
            .into_iter()
            .map(|i| {
                (
                    i.imported_path,
                    i.import_type,
                    i.line_number,
                    i.imported_symbols,
                )
            })
            .collect()
    }

    fn paths(source: &str) -> Vec<String> {
        imports(source).into_iter().map(|i| i.0).collect()
    }

    #[test]
    fn every_import_form() {
        let source = "\
import Foundation
@testable import MyApp
@_exported import Core
@_implementationOnly import Secret
@preconcurrency import Dispatch
@_spi(Private) @testable import Kit
public import Shared
package import Pkg
internal import Inside
import struct Foundation.URL
import func Darwin.sqrt
import UIKit.UIGestureRecognizerSubclass
import Alamofire; import SnapKit
";
        assert_eq!(
            imports(source),
            vec![
                ("Foundation".into(), ImportType::Stdlib, 1, None),
                ("MyApp".into(), ImportType::Internal, 2, None),
                ("Core".into(), ImportType::External, 3, None),
                ("Secret".into(), ImportType::External, 4, None),
                ("Dispatch".into(), ImportType::Stdlib, 5, None),
                ("Kit".into(), ImportType::Internal, 6, None),
                ("Shared".into(), ImportType::External, 7, None),
                ("Pkg".into(), ImportType::External, 8, None),
                ("Inside".into(), ImportType::External, 9, None),
                (
                    "Foundation.URL".into(),
                    ImportType::Stdlib,
                    10,
                    Some(vec!["URL".into()])
                ),
                (
                    "Darwin.sqrt".into(),
                    ImportType::Stdlib,
                    11,
                    Some(vec!["sqrt".into()])
                ),
                (
                    "UIKit.UIGestureRecognizerSubclass".into(),
                    ImportType::Stdlib,
                    12,
                    None
                ),
                ("Alamofire".into(), ImportType::External, 13, None),
                ("SnapKit".into(), ImportType::External, 13, None),
            ]
        );
    }

    #[test]
    fn conditional_imports_are_static_imports() {
        let source = "#if canImport(UIKit)\nimport UIKit\n#else\nimport AppKit\n#endif\n";
        assert_eq!(paths(source), vec!["UIKit", "AppKit"]);
    }

    #[test]
    fn comments_and_strings_are_not_imports() {
        let source = r####"
// import LineComment
/* import Block
   /* nested */
import StillInBlock */
let s = "import InString"
let m = """
import InMultiline
"""
let r = #""import" \#(x) import InRaw"#
let rm = #"""
import InRawMultiline
"""#
let i = "\(f("import X")) import AfterInterpolation"
import Real // import Trailing
"####;
        assert_eq!(paths(source), vec!["Real"]);
    }

    #[test]
    fn identifiers_that_start_with_import_are_not_imports() {
        assert!(paths("importer.run()\nlet imports = 1\nimport\n").is_empty());
    }

    #[test]
    fn package_swift_targets_and_their_directories() {
        let source = r#"
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Demo",
    dependencies: [.package(url: "https://github.com/a/b", from: "1.0.0")],
    targets: [
        .target(name: "Core", dependencies: [.product(name: "B", package: "b")]),
        .target(
            name: "Custom",
            exclude: ["x"],
            path: "Lib/Custom"
        ),
        .executableTarget(name: "App", dependencies: [.target(name: "Custom"), "Core"]),
        .testTarget(name: "CoreTests", dependencies: ["Core"]),
        .plugin(name: "Gen", capability: .buildTool()),
        // .target(name: "Commented"),
        .binaryTarget(name: "Bin", path: "Bin.xcframework"),
    ]
)
"#;
        let modules = parse_package_swift("pkgs/demo/Package.swift", source);
        let got: Vec<(&str, &str, bool)> = modules
            .iter()
            .map(|m| (m.name.as_str(), m.dir.as_str(), m.entry))
            .collect();
        assert_eq!(
            got,
            vec![
                ("Core", "pkgs/demo/Sources/Core", false),
                ("Custom", "pkgs/demo/Lib/Custom", false),
                ("App", "pkgs/demo/Sources/App", true),
                ("CoreTests", "pkgs/demo/Tests/CoreTests", true),
                ("Gen", "pkgs/demo/Plugins/Gen", true),
            ]
        );
        assert!(modules.iter().all(|m| m.package_dir == "pkgs/demo"));
    }

    #[test]
    fn nearest_package_wins_then_the_first_one() {
        let module = |package_dir: &str, name: &str| SwiftModule {
            package_dir: package_dir.into(),
            name: name.into(),
            dir: format!("{package_dir}/Sources/{name}"),
            entry: false,
        };
        let modules = vec![
            module("a", "Core"),
            module("b", "Core"),
            module("c", "Util"),
        ];
        let dir = |m: Option<&SwiftModule>| m.map(|m| m.dir.clone());
        assert_eq!(
            dir(find_swift_module("Core", "b/Sources/App/x.swift", &modules)),
            Some("b/Sources/Core".into())
        );
        assert_eq!(
            dir(find_swift_module("Core", "z/x.swift", &modules)),
            Some("a/Sources/Core".into())
        );
        assert_eq!(find_swift_module("Nope", "a/x.swift", &modules), None);
    }

    #[test]
    fn a_file_belongs_to_the_deepest_target_above_it() {
        let module = |dir: &str| SwiftModule {
            package_dir: String::new(),
            name: dir.rsplit('/').next().unwrap().into(),
            dir: dir.into(),
            entry: false,
        };
        let modules = vec![module("Sources/Core"), module("Sources/Core/Plugin")];
        let dir = |rel: &str| swift_target_of(rel, &modules).map(|m| m.dir.clone());
        assert_eq!(
            dir("Sources/Core/Sub/Model.swift"),
            Some("Sources/Core".into())
        );
        assert_eq!(
            dir("Sources/Core/Plugin/P.swift"),
            Some("Sources/Core/Plugin".into())
        );
        assert_eq!(dir("Sources/CoreKit/X.swift"), None);
        assert_eq!(dir("main.swift"), None);
    }
}
