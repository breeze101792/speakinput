#!/usr/bin/env bash
# Build a single-file frozen binary of speakinput with PyInstaller and
# stage it in dist/ alongside config.example.toml.
#
# - Builds for the current host (macOS arm64 / x86_64, Linux x86_64).
#   Cross-compiling isn't supported by PyInstaller; if you need another
#   platform, run this script on that host.
# - Output: dist/speakinput (the binary) + dist/config.example.toml
#   + a copy of README.md. No installer — users untar, run, done.
# - Models and the model cache are NOT bundled. The first run downloads
#   the configured model into the user's pywhispercpp cache, exactly as
#   `pip install speakinput` would. The frozen binary is otherwise
#   self-contained.
#
# Usage: ./release.sh [--help] [CLEAN=1]
#   --help, -h   Print usage and exit (also reachable via `release help`).
#   CLEAN=1      Wipe build/ and dist/ before building.

set -euo pipefail

cd "$(dirname "$0")"

log() { printf '\033[1;34m[release.sh]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[release.sh]\033[0m %s\n' "$*" >&2; }

usage() {
    cat <<EOF
release.sh — build a single-file frozen speakinput binary with PyInstaller.

Usage:
    ./release.sh [--help|-h|help]
    ./release.sh [CLEAN=1]

Options:
    --help, -h, help   Print this message and exit.
    CLEAN=1            Wipe build/ and dist/ before building.

Output:
    dist/speakinput            The frozen binary (~15 MB on macOS arm64).
    dist/config.example.toml   Copy of the example config (ship with the binary).
    dist/README.md             Copy of the README (ship with the binary).

Notes:
    - Cross-compiling isn't supported. Run on the host you want to ship for.
    - Whisper models download on first run into the user's pywhispercpp cache.
    - macOS Gatekeeper may reject the unsigned binary on first run; either
      right-click → Open, or sign with:
          codesign --force --deep --sign - dist/speakinput
EOF
}

find_python() {
    # Prefer the venv if it exists, since the user may have bootstrapped
    # with a different interpreter.
    if [[ -x .venv/bin/python ]]; then
        echo ".venv/bin/python"
        return
    fi
    local candidate ver higher
    for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            ver=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
            higher=$(printf '%s\n3.11\n' "$ver" | sort -V | tail -n1)
            if [[ "$higher" != "3.11" ]]; then
                command -v "$candidate"
                return
            fi
        fi
    done
    err "Python 3.11+ not found. Install via Homebrew: brew install python@3.12"
    return 1
}

ensure_venv() {
    local py=$1
    if [[ ! -d .venv ]]; then
        log "creating virtualenv in .venv using $py"
        "$py" -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
}

ensure_app_deps() {
    # Skip the editable install if the package and its runtime deps
    # are already importable. Saves minutes on repeat builds.
    if [[ -d src/speakinput.egg-info ]]; then
        if .venv/bin/python -c 'import pywhispercpp, sounddevice, pynput, pyperclip, platformdirs, numpy; import speakinput' 2>/dev/null; then
            return
        fi
    fi
    log "installing speakinput + deps into .venv"
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -e ".[menu]"
}

ensure_pyinstaller() {
    if ! .venv/bin/python -c 'import PyInstaller' 2>/dev/null; then
        log "installing pyinstaller"
        .venv/bin/pip install --quiet "pyinstaller>=6.0"
    fi
}

run_build() {
    local spec=$1
    log "building speakinput (this takes a couple of minutes the first time)"
    .venv/bin/pyinstaller \
        --noconfirm \
        --clean \
        --workpath build \
        --distpath dist \
        "$spec"
}

stage_artifacts() {
    cp config.example.toml dist/config.example.toml
    cp README.md dist/README.md
}

report_artifact() {
    local binary="dist/speakinput"
    if [[ ! -x "$binary" ]]; then
        err "build did not produce $binary"
        return 1
    fi
    local size
    size=$(du -h "$binary" | cut -f1)
    log "done: $binary ($size) — run ./$binary"
}

# --- arg parsing ----------------------------------------------------------
case "${1:-}" in
    --help|-h|help)
        usage
        exit 0
        ;;
esac

# --- main pipeline --------------------------------------------------------
PY=$(find_python)
ensure_venv "$PY"
ensure_app_deps
ensure_pyinstaller

if [[ "${CLEAN:-0}" == "1" ]]; then
    log "cleaning build/ and dist/"
    rm -rf build dist
fi
mkdir -p dist

run_build speakinput.spec
stage_artifacts
report_artifact
