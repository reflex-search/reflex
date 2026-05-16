//! Pagefind binary management: download, cache, and version-check
//!
//! Downloads a pinned Pagefind release from GitHub and caches it at `~/.reflex/bin/pagefind`.
//! Handles platform detection, tar.gz extraction, and executable permissions.
//!
//! ## Hash-pinning approach
//!
//! Every supported platform asset has its SHA256 hash embedded at compile time in
//! `expected_sha256`. After the archive is downloaded, `verify_sha256` computes the
//! digest of the raw bytes and compares it to the expected value before any
//! extraction or execution takes place. A mismatch aborts with a clear error;
//! the archive is never written to disk.
//!
//! To update to a new Pagefind version: bump `PAGEFIND_VERSION`, download each
//! platform archive (or fetch the `.sha256` files from the GitHub release), and
//! update the match arms in `expected_sha256`.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Pinned Pagefind version
const PAGEFIND_VERSION: &str = "1.5.0";

/// Expected SHA256 hashes (hex-encoded) for each Pagefind v1.5.0 platform archive.
///
/// Sourced from the official `.sha256` sidecar files published alongside each
/// GitHub release asset (e.g. `pagefind-v1.5.0-x86_64-unknown-linux-musl.tar.gz.sha256`).
fn expected_sha256(asset_name: &str) -> Option<&'static str> {
    match asset_name {
        "pagefind-v1.5.0-x86_64-unknown-linux-musl.tar.gz" => {
            Some("0a10c6d780bc2a61378cfafe01be620a4de8400de4a1eafd180a0015d002000b")
        }
        "pagefind-v1.5.0-aarch64-unknown-linux-musl.tar.gz" => {
            Some("11379079316e3a10a72649ee52ff12f74fc5ad884dc3b6281ff0550a87681a2b")
        }
        "pagefind-v1.5.0-x86_64-apple-darwin.tar.gz" => {
            Some("d8c3717eb99c1fcd97a3d13fc1f9ffe1316b3d3fed86d583a5c3af013d324322")
        }
        "pagefind-v1.5.0-aarch64-apple-darwin.tar.gz" => {
            Some("90c2c5c784ece0ca300c430f1043f043ef21200791f0591d8d8c88c5f555c38f")
        }
        _ => None,
    }
}

/// Compute the lowercase hex SHA256 digest of `bytes`.
fn compute_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// Verify `bytes` match `expected_hex` before extraction. Returns an error
/// (without writing anything to disk) if the digest does not match.
fn verify_sha256(bytes: &[u8], expected_hex: &str) -> Result<()> {
    let actual = compute_sha256(bytes);
    if actual != expected_hex {
        anyhow::bail!(
            "SHA256 mismatch for downloaded Pagefind archive.\n  Expected: {}\n  Got:      {}\n\
             This may indicate a corrupted download or a supply-chain attack. \
             Delete ~/.reflex/bin/ and retry, or file a bug if the issue persists.",
            expected_hex,
            actual,
        );
    }
    Ok(())
}

/// Return the path to the Pagefind binary, downloading it if needed.
///
/// 1. Check if `~/.reflex/bin/pagefind` exists
/// 2. If yes, verify version matches PAGEFIND_VERSION
/// 3. If no or version mismatch, download, verify SHA256, then extract
pub fn ensure_pagefind() -> Result<PathBuf> {
    let bin_dir = get_bin_dir()?;
    let pagefind_path = bin_dir.join("pagefind");

    if pagefind_path.exists() {
        // Check version
        if check_version(&pagefind_path)? {
            return Ok(pagefind_path);
        }
        eprintln!("Pagefind version mismatch, re-downloading...");
    }

    // Download and extract
    std::fs::create_dir_all(&bin_dir).context("Failed to create ~/.reflex/bin/")?;

    download_pagefind(&pagefind_path)?;

    Ok(pagefind_path)
}

/// Get the ~/.reflex/bin/ directory path
fn get_bin_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".reflex").join("bin"))
}

/// Check if the installed Pagefind binary matches the pinned version
fn check_version(pagefind_path: &PathBuf) -> Result<bool> {
    let output = std::process::Command::new(pagefind_path)
        .arg("--version")
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let version_str = String::from_utf8_lossy(&output.stdout);
            Ok(version_str.contains(PAGEFIND_VERSION))
        }
        _ => Ok(false),
    }
}

/// Determine the correct asset name for this platform
fn get_asset_name() -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let target = match (os, arch) {
        ("linux", "x86_64") => "x86_64-unknown-linux-musl",
        ("linux", "aarch64") => "aarch64-unknown-linux-musl",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        _ => anyhow::bail!(
            "Unsupported platform: {}-{}. Install Pagefind manually: https://pagefind.app/docs/installation/",
            os,
            arch
        ),
    };

    Ok(format!("pagefind-v{}-{}.tar.gz", PAGEFIND_VERSION, target))
}

/// Download Pagefind from GitHub releases, verify SHA256, then extract to `pagefind_path`.
fn download_pagefind(pagefind_path: &PathBuf) -> Result<()> {
    let asset_name = get_asset_name()?;

    let hash = expected_sha256(&asset_name).ok_or_else(|| {
        anyhow::anyhow!(
            "No SHA256 hash pinned for asset \'{}\'. Cannot safely download.",
            asset_name
        )
    })?;

    let url = format!(
        "https://github.com/CloudCannon/pagefind/releases/download/v{}/{}",
        PAGEFIND_VERSION, asset_name
    );

    eprintln!("Downloading Pagefind v{} from {}...", PAGEFIND_VERSION, url);

    let rt = tokio::runtime::Runtime::new()?;
    let bytes = rt.block_on(async {
        let response = reqwest::get(&url)
            .await
            .context("Failed to download Pagefind")?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download Pagefind: HTTP {} from {}",
                response.status(),
                url
            );
        }

        response
            .bytes()
            .await
            .context("Failed to read Pagefind download")
    })?;

    // Verify integrity before touching the filesystem
    verify_sha256(&bytes, hash)?;

    eprintln!("SHA256 verified. Extracting Pagefind binary...");

    let decoder = flate2::read::GzDecoder::new(&bytes[..]);
    let mut archive = tar::Archive::new(decoder);

    let mut found = false;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;

        if path.file_name().map(|n| n == "pagefind").unwrap_or(false) {
            let mut file =
                std::fs::File::create(pagefind_path).context("Failed to create pagefind binary")?;
            std::io::copy(&mut entry, &mut file)?;
            found = true;
            break;
        }
    }

    if !found {
        anyhow::bail!("Could not find 'pagefind' binary in the downloaded archive");
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(pagefind_path, perms)
            .context("Failed to set executable permission on pagefind binary")?;
    }

    eprintln!(
        "Pagefind v{} installed at {}",
        PAGEFIND_VERSION,
        pagefind_path.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_asset_name() {
        let result = get_asset_name();
        if cfg!(any(target_os = "linux", target_os = "macos")) {
            assert!(result.is_ok(), "Should detect platform: {:?}", result.err());
            let name = result.unwrap();
            assert!(name.contains(PAGEFIND_VERSION));
            assert!(name.ends_with(".tar.gz"));
        } else {
            // Pagefind has no Windows release asset; the helper should
            // explicitly report the platform as unsupported.
            let err = result.expect_err("expected unsupported platform error");
            assert!(err.to_string().contains("Unsupported platform"));
        }
    }

    #[test]
    fn test_bin_dir() {
        let dir = get_bin_dir().unwrap();
        assert!(dir.ends_with(".reflex/bin"));
    }

    #[test]
    fn test_sha256_mismatch_returns_error() {
        let bytes = b"this is a tampered binary payload";
        let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        let result = verify_sha256(bytes, wrong_hash);
        assert!(
            result.is_err(),
            "Tampered binary must not pass verification"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("SHA256 mismatch"),
            "Expected mismatch error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_sha256_correct_hash_passes() {
        let bytes = b"known content for testing";
        let correct_hash = compute_sha256(bytes);
        assert!(
            verify_sha256(bytes, &correct_hash).is_ok(),
            "Correct hash must pass"
        );
    }

    #[test]
    fn test_expected_sha256_known_assets() {
        let assets = [
            "pagefind-v1.5.0-x86_64-unknown-linux-musl.tar.gz",
            "pagefind-v1.5.0-aarch64-unknown-linux-musl.tar.gz",
            "pagefind-v1.5.0-x86_64-apple-darwin.tar.gz",
            "pagefind-v1.5.0-aarch64-apple-darwin.tar.gz",
        ];
        for asset in &assets {
            let hash = expected_sha256(asset);
            assert!(hash.is_some(), "Missing hash for {}", asset);
            assert_eq!(hash.unwrap().len(), 64, "Invalid hash length for {}", asset);
        }
    }

    #[test]
    fn test_expected_sha256_unknown_asset_returns_none() {
        assert!(expected_sha256("pagefind-v9.9.9-unknown.tar.gz").is_none());
    }
}
