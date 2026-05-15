//! Zola binary management: download, cache, and version-check
//!
//! Downloads a pinned Zola release from GitHub and caches it at `~/.reflex/bin/zola`.
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
//! To update to a new Zola version: bump `ZOLA_VERSION`, download each platform
//! archive, run `sha256sum`, and update the match arms in `expected_sha256`.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Pinned Zola version
const ZOLA_VERSION: &str = "0.19.2";

/// Expected SHA256 hashes (hex-encoded) for each Zola v0.19.2 platform archive.
///
/// Computed from official GitHub release assets; Zola does not publish a separate
/// checksums file, so these were produced by downloading each asset and running
/// `sha256sum`.
///
/// NOTE: Zola v0.19.2 has no aarch64-unknown-linux-gnu release asset. Linux/aarch64
/// receives an explicit "unsupported platform" error from `get_asset_name`.
fn expected_sha256(asset_name: &str) -> Option<&'static str> {
    match asset_name {
        "zola-v0.19.2-x86_64-unknown-linux-gnu.tar.gz" => {
            Some("0798e69b86c628ddcb264ebd86c8cc8dce7670b9049060bf94faa73f6857cd9c")
        }
        "zola-v0.19.2-x86_64-apple-darwin.tar.gz" => {
            Some("38194f1d424bb4303c190fec149d90134ab33dd2d329831309deb409bcf416f8")
        }
        "zola-v0.19.2-aarch64-apple-darwin.tar.gz" => {
            Some("82c173381aced5edb28394c3202417e6dce31f0a5941ae58dd4e5e9969f5f375")
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
            "SHA256 mismatch for downloaded Zola archive.\n  Expected: {}\n  Got:      {}\n\
             This may indicate a corrupted download or a supply-chain attack. \
             Delete ~/.reflex/bin/ and retry, or file a bug if the issue persists.",
            expected_hex,
            actual,
        );
    }
    Ok(())
}

/// Return the path to the Zola binary, downloading it if needed.
///
/// 1. Check if `~/.reflex/bin/zola` exists
/// 2. If yes, verify version matches ZOLA_VERSION
/// 3. If no or version mismatch, download, verify SHA256, then extract
pub fn ensure_zola() -> Result<PathBuf> {
    let bin_dir = get_bin_dir()?;
    let zola_path = bin_dir.join("zola");

    if zola_path.exists() {
        // Check version
        if check_version(&zola_path)? {
            return Ok(zola_path);
        }
        eprintln!("Zola version mismatch, re-downloading...");
    }

    // Download and extract
    std::fs::create_dir_all(&bin_dir).context("Failed to create ~/.reflex/bin/")?;

    download_zola(&zola_path)?;

    Ok(zola_path)
}

/// Get the ~/.reflex/bin/ directory path
fn get_bin_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".reflex").join("bin"))
}

/// Check if the installed Zola binary matches the pinned version
fn check_version(zola_path: &PathBuf) -> Result<bool> {
    let output = std::process::Command::new(zola_path)
        .arg("--version")
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let version_str = String::from_utf8_lossy(&output.stdout);
            // zola --version outputs "zola 0.19.2"
            Ok(version_str.contains(ZOLA_VERSION))
        }
        _ => Ok(false),
    }
}

/// Determine the correct asset name for this platform
fn get_asset_name() -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let target = match (os, arch) {
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        // Zola v0.19.2 has no aarch64-linux release asset.
        ("linux", "aarch64") => anyhow::bail!(
            "Zola v{} does not have an aarch64-unknown-linux-gnu release. \
             Install Zola manually: https://www.getzola.org/documentation/getting-started/installation/",
            ZOLA_VERSION
        ),
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        _ => anyhow::bail!(
            "Unsupported platform: {}-{}. Install Zola manually: https://www.getzola.org/documentation/getting-started/installation/",
            os,
            arch
        ),
    };

    Ok(format!("zola-v{}-{}.tar.gz", ZOLA_VERSION, target))
}

/// Download Zola from GitHub releases, verify SHA256, then extract to `zola_path`.
fn download_zola(zola_path: &PathBuf) -> Result<()> {
    let asset_name = get_asset_name()?;

    let hash = expected_sha256(&asset_name).ok_or_else(|| {
        anyhow::anyhow!(
            "No SHA256 hash pinned for asset \'{}\'. Cannot safely download.",
            asset_name
        )
    })?;

    let url = format!(
        "https://github.com/getzola/zola/releases/download/v{}/{}",
        ZOLA_VERSION, asset_name
    );

    eprintln!("Downloading Zola v{} from {}...", ZOLA_VERSION, url);

    let rt = tokio::runtime::Runtime::new()?;
    let bytes = rt.block_on(async {
        let response = reqwest::get(&url)
            .await
            .context("Failed to download Zola")?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download Zola: HTTP {} from {}",
                response.status(),
                url
            );
        }

        response
            .bytes()
            .await
            .context("Failed to read Zola download")
    })?;

    // Verify integrity before touching the filesystem
    verify_sha256(&bytes, hash)?;

    eprintln!("SHA256 verified. Extracting Zola binary...");

    let decoder = flate2::read::GzDecoder::new(&bytes[..]);
    let mut archive = tar::Archive::new(decoder);

    let mut found = false;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;

        if path.file_name().map(|n| n == "zola").unwrap_or(false) {
            let mut file =
                std::fs::File::create(zola_path).context("Failed to create zola binary")?;
            std::io::copy(&mut entry, &mut file)?;
            found = true;
            break;
        }
    }

    if !found {
        anyhow::bail!("Could not find 'zola' binary in the downloaded archive");
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(zola_path, perms)
            .context("Failed to set executable permission on zola binary")?;
    }

    eprintln!(
        "Zola v{} installed at {}",
        ZOLA_VERSION,
        zola_path.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_asset_name() {
        let result = get_asset_name();
        assert!(result.is_ok(), "Should detect platform: {:?}", result.err());
        let name = result.unwrap();
        assert!(name.contains(ZOLA_VERSION));
        assert!(name.ends_with(".tar.gz"));
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
            "zola-v0.19.2-x86_64-unknown-linux-gnu.tar.gz",
            "zola-v0.19.2-x86_64-apple-darwin.tar.gz",
            "zola-v0.19.2-aarch64-apple-darwin.tar.gz",
        ];
        for asset in &assets {
            let hash = expected_sha256(asset);
            assert!(hash.is_some(), "Missing hash for {}", asset);
            assert_eq!(hash.unwrap().len(), 64, "Invalid hash length for {}", asset);
        }
    }

    #[test]
    fn test_expected_sha256_unknown_asset_returns_none() {
        assert!(expected_sha256("zola-v9.9.9-unknown.tar.gz").is_none());
    }
}
