#!/usr/bin/env bash
# Windows clippy check, run from Linux or macOS with a mingw cross-compiler.
#
# Tests written on Unix often use Unix-only code (std::os::unix, a variable
# only a #[cfg(unix)] block reads). The Windows CI job then fails to compile.
# This catches that before a push. It never blocks: with no compiler or Rust
# target it prints why it skipped and exits 0. CI stays the final check.
#
# Compiler: x86_64-w64-mingw32-gcc on the PATH, else nix (pkgsCross.mingwW64).
#   Debian/Ubuntu: apt install gcc-mingw-w64-x86-64
#   Fedora:        dnf install mingw64-gcc
#   Arch:          pacman -S mingw-w64-gcc
#   macOS:         brew install mingw-w64
# Rust target: rustup target add x86_64-pc-windows-gnu
set -euo pipefail

target=x86_64-pc-windows-gnu
skip() {
    echo "Windows check skipped: $1" >&2
    exit 0
}

case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) skip "on Windows, plain cargo clippy already checks it" ;;
esac

command -v rustup >/dev/null || skip "rustup not found"
rustup target list --installed | grep -qx "$target" ||
    skip "Rust target missing (rustup target add $target)"

cd "$(dirname "$0")/.."
check="CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc \
CXX_x86_64_pc_windows_gnu=x86_64-w64-mingw32-g++ \
AR_x86_64_pc_windows_gnu=x86_64-w64-mingw32-ar \
cargo clippy --quiet --all-targets --target $target -- -D warnings"

if command -v x86_64-w64-mingw32-gcc >/dev/null; then
    exec bash -c "$check"
elif command -v nix >/dev/null; then
    exec nix --extra-experimental-features 'nix-command flakes' \
        shell nixpkgs#pkgsCross.mingwW64.stdenv.cc -c bash -c "$check"
else
    skip "no mingw cross-compiler (see scripts/windows-check.sh for install commands)"
fi
