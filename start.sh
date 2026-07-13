#!/usr/bin/env bash
# Bootstrap and launch speakinput.
#
# - Ensures a Python 3.11+ interpreter is available.
# - Creates .venv on first run, upgrades pip, installs the package in
#   editable mode (with the [menu] extra for the optional menu-bar indicator).
# - Runs `speakinput` from the venv.
#
# Idempotent: re-running is fast (skips pip install if already up to date).
# Args are forwarded to `speakinput` (e.g. `./start.sh --diagnose`).

set -euo pipefail

cd "$(dirname "$0")"

log() { printf '\033[1;34m[start.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[start.sh]\033[0m %s\n' "$*" >&2; }
err() { printf '\033[1;31m[start.sh]\033[0m %s\n' "$*" >&2; }

# 1. Find a usable Python (>= 3.11). Prefer the venv if it exists, since the
#    user may have bootstrapped with a different interpreter.
if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    PY=""
    for candidate in python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            ver=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
            if [[ "$(printf '%s\n3.11' "$ver" | sort -V | tail -n1)" == "3.11" ]]; then
                PY=$(command -v "$candidate")
                break
            fi
        fi
    done
    if [[ -z "$PY" ]]; then
        err "Python 3.11+ not found. Install via Homebrew: brew install python@3.12"
        exit 1
    fi
fi

# 2. Create the venv on first run.
if [[ ! -d .venv ]]; then
    log "creating virtualenv in .venv using $PY"
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Ensure the package is installed (editable). Skip if the egg-info exists
#    AND every listed dep imports cleanly — a missing dep will fail the import
#    probe and trigger a reinstall on the next run.
need_install=1
if [[ -d src/speakinput.egg-info ]]; then
    if .venv/bin/python -c 'import pywhispercpp, sounddevice, pynput, pyperclip, platformdirs, numpy' 2>/dev/null; then
        need_install=0
    fi
fi

if [[ $need_install -eq 1 ]]; then
    log "installing speakinput (this may take a minute on first run)"
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -e ".[menu]"
fi

# 4. Forward to the CLI.
exec .venv/bin/speakinput "$@"
