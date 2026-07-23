#!/usr/bin/env bash
# Bootstrap and launch speakinput.
#
# - Ensures a Python 3.11+ interpreter is available.
# - Creates .venv_<hostname> on first run, upgrades pip, installs the
#   package in editable mode (with the [menu] extra for the optional
#   menu-bar indicator).
# - Runs `speakinput` from the venv.
#
# Idempotent: re-running is fast (skips pip install if already up to date).
# Args are forwarded to `speakinput` (e.g. `./start.sh --diagnose`).

set -euo pipefail

cd "$(dirname "$0")"

HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname)
VENV_DIR=".venv_${HOSTNAME_SHORT}"

log() { printf '\033[1;34m[start.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[start.sh]\033[0m %s\n' "$*" >&2; }
err() { printf '\033[1;31m[start.sh]\033[0m %s\n' "$*" >&2; }

# 1. Find a usable Python (>= 3.11). Prefer the venv if it exists, since the
#    user may have bootstrapped with a different interpreter.
if [[ -x "$VENV_DIR/bin/python" ]]; then
    PY="$VENV_DIR/bin/python"
else
    PY=""
    # Probe candidates newest-first; the version check is "is it >= 3.11",
    # not "is it equal to 3.11" — the old `sort -V | tail` check was inverted
    # and rejected Python 3.12/3.13/3.14.
    for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            ver=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
            # `sort -V` puts the higher version last; if the higher of
            # (candidate, 3.11) is the candidate, candidate >= 3.11.
            higher=$(printf '%s\n3.11\n' "$ver" | sort -V | tail -n1)
            if [[ "$higher" != "3.11" ]]; then
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
if [[ ! -d "$VENV_DIR" ]]; then
    log "creating virtualenv in $VENV_DIR using $PY"
    "$PY" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 3. Ensure the package is installed (editable). Skip if the egg-info exists
#    AND every listed dep imports cleanly — a missing dep will fail the import
#    probe and trigger a reinstall on the next run.
need_install=1
if [[ -d src/speakinput.egg-info ]]; then
    if "$VENV_DIR/bin/python" -c 'import pywhispercpp, sounddevice, pynput, pyperclip, platformdirs, numpy' 2>/dev/null; then
        need_install=0
    fi
fi

if [[ $need_install -eq 1 ]]; then
    log "installing speakinput (this may take a minute on first run)"
    "$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -e ".[menu]"
fi

# 4. Copy config.example.toml into the user config dir on first run, so the
#    user can `vim` it to customize. The program also runs fine with no
#    config file at all (defaults are baked in), so this is purely for
#    discoverability — skip silently if the dir can't be created (CI, etc.).
#    Use IFS= and process substitution so the path with a space
#    ("Application Support") survives word-splitting. `read` returns 1 on
#    EOF (no trailing newline) — that's not an error, swallow it.
#    Use $VENV_DIR/bin/python explicitly (not $PY) so the venv's `platformdirs`
#    is on the path; the system python won't have it installed.
IFS= read -r user_cfg_dir < <("$VENV_DIR/bin/python" -c 'from platformdirs import user_config_dir; print(user_config_dir("speakinput", appauthor=False), end="")') || true
if [[ -n "$user_cfg_dir" && ! -f "$user_cfg_dir/config.toml" && -f config.example.toml ]]; then
    if mkdir -p "$user_cfg_dir" 2>/dev/null; then
        cp config.example.toml "$user_cfg_dir/config.toml"
        log "created $user_cfg_dir/config.toml from config.example.toml"
        log "edit it to customize; the program uses defaults if it's missing"
    fi
fi

# 5. Detect an existing speakinput instance. The single-instance guard holds
#    an exclusive `flock` on a lockfile under the user's runtime dir. We
#    probe the same lock with LOCK_NB; if the lock is held, another instance
#    is alive. We do this from inside the venv so `platformdirs` is on the
#    import path (it's a runtime dep installed in step 3). Using bash+flock
#    instead would be simpler, but `flock` isn't shipped on macOS.
#
#    stdout contract: prints either "free" or "held:<pid>" (pid may be
#    "unknown" if the lockfile contents couldn't be read).
probe_instance_lock() {
"$VENV_DIR/bin/python" <<'PY' || true
import os, sys, fcntl
from pathlib import Path
try:
    from platformdirs import user_runtime_dir
except Exception:
    print("no-platformdirs")
    sys.exit(0)
runtime = Path(user_runtime_dir("speakinput", appauthor=False))
runtime.mkdir(parents=True, exist_ok=True)
lock = runtime / "speakinput.lock"
fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except (BlockingIOError, OSError):
    pid = "unknown"
    try:
        os.lseek(fd, 0, 0)
        data = os.read(fd, 64).decode("utf-8", "replace").strip()
        if data:
            pid = data
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass
    print(f"held:{pid}")
    sys.exit(0)
try:
    os.close(fd)
except OSError:
    pass
print("free")
PY
}

instance_status=$(probe_instance_lock)
if [[ "$instance_status" == held:* ]]; then
    existing_pid=${instance_status#held:}

    if [[ ! -t 0 ]]; then
        # Non-interactive (CI, pipe, double-click in Finder, etc.) — don't
        # kill anything without explicit consent.
        err "another speakinput is already running (pid=$existing_pid)."
        err "stop it with: kill $existing_pid"
        err "(or re-run start.sh interactively to be prompted)"
        exit 1
    fi

    if [[ -z "$existing_pid" || "$existing_pid" == "unknown" ]]; then
        err "another speakinput is running but its pid could not be read."
        err "stop it with: pkill -f speakinput"
        exit 1
    fi

    # Re-verify the pid is still alive before we ask. If it just exited,
    # the lock is presumably about to be released and we can re-probe.
    if ! kill -0 "$existing_pid" 2>/dev/null; then
        log "lock was held by pid=$existing_pid, but that process is gone; re-probing"
        instance_status=$(probe_instance_lock)
        if [[ "$instance_status" == free ]]; then
            :
        else
            other=${instance_status#held:}
            err "lock re-acquired by another pid=$other before we could start; aborting"
            exit 1
        fi
    else
        printf '\033[1;33m[start.sh]\033[0m another speakinput is running (pid=%s). Kill it and continue? [y/N] ' "$existing_pid"
        if ! read -r -n 1 answer; then
            printf '\n'
            answer=""
        else
            printf '\n'
        fi
        if [[ ! "$answer" =~ ^[Yy]$ ]]; then
            err "aborted. the other speakinput (pid=$existing_pid) is still running."
            err "stop it with: kill $existing_pid"
            exit 1
        fi

        log "killing existing instance (pid=$existing_pid)"
        # Graceful first, then escalate. Both kill(1) and kill -0 are POSIX
        # and behave the same on macOS and Linux.
        kill -TERM "$existing_pid" 2>/dev/null || true
        for _ in {1..30}; do
            if ! kill -0 "$existing_pid" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        if kill -0 "$existing_pid" 2>/dev/null; then
            warn "pid=$existing_pid did not exit on SIGTERM; sending SIGKILL"
            kill -KILL "$existing_pid" 2>/dev/null || true
            sleep 0.2
        fi

        # Re-probe the lock. The kernel releases the flock when the last
        # fd holding it is closed (i.e. when the killed process actually
        # exits), so a brief grace period before re-probing prevents a
        # spurious "still held" when SIGKILL was needed.
        sleep 0.2
        instance_status=$(probe_instance_lock)
        if [[ "$instance_status" == held:* ]]; then
            other=${instance_status#held:}
            err "lock is still held by pid=$other after kill; aborting"
            err "if that's a different process, stop it manually first."
            exit 1
        fi
    fi
fi

# 6. Forward to the CLI.
exec "$VENV_DIR/bin/speakinput" "$@"
