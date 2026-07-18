#!/usr/bin/env bash
# One-shot environment setup for speakinput.
#
# - Detects the GPU vendor (NVIDIA / AMD / Intel / Apple) and the package
#   manager (pacman / apt / dnf / brew), then installs the right runtime
#   libs and rebuilds the pywhispercpp wheel against the matching backend
#   (CUDA / Vulkan / CoreML). After the rebuild, probes the loaded
#   libwhisper.so to confirm the backend actually got baked in.
# - Assumes `./start.sh` has already been run (it creates the venv and
#   installs the CPU-only wheel; this script just adds the GPU stack on
#   top and rebuilds).
# - Idempotent: re-running skips already-installed packages and the pip
#   rebuild is `--force-reinstall --no-cache` so it's the same wall-clock
#   cost the first time and the second time.
# - Asks before every mutating step — both at the script level (our
#   own [y/n/a] prompt for each install) and at the package-manager
#   level (pacman/apt/dnf's own 'Proceed?' prompt, with no
#   --noconfirm / -y flags). The user sees what's about to change
#   and gets to confirm twice. This script is interactive-only.
#
# Usage: ./setup.sh [--help] [--dry-run]

set -euo pipefail

cd "$(dirname "$0")"

HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname)
VENV_DIR=".venv_${HOSTNAME_SHORT}"

log() { printf '\033[1;34m[setup.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup.sh]\033[0m %s\n' "$*" >&2; }
err() { printf '\033[1;31m[setup.sh]\033[0m %s\n' "$*" >&2; }
step() { printf '\033[1;36m[setup.sh]\033[0m ▶ %s\n' "$*"; }
ok() { printf '\033[1;32m[setup.sh]\033[0m ✓ %s\n' "$*"; }

DRY_RUN=0

usage() {
    cat <<EOF
setup.sh — install the GPU runtime and rebuild pywhispercpp for speakinput.

Detects your GPU vendor and the system package manager, then:
  1. Installs the vendor runtime (CUDA toolkit, Vulkan ICD + loader, or
     the macOS CoreML model downloader on Apple Silicon).
  2. Rebuilds the pywhispercpp wheel against the matching backend
     (GGML_CUDA / GGML_VULKAN / WHISPER_COREML).
  3. Probes the loaded libwhisper.so to confirm the backend is active.

Idempotent: re-running is safe.

Prerequisite: ./start.sh has been run at least once (it creates the venv
that this script installs into).

Usage:
    ./setup.sh [--help|-h|help]
    ./setup.sh [--dry-run]

Options:
    --help, -h, help   Print this message and exit.
    --dry-run          Print the plan and exit without installing anything.
                       Useful for previewing what the script will do on
                       this box, or for running in a non-interactive
                       environment where the install would fail anyway.
                       Note: this script is fundamentally interactive
                       (it asks before every install), so --dry-run is
                       the only way to use it from CI.

This script is interactive-only. It NEVER touches the system Python —
every pip operation goes into the project's venv ($VENV_DIR). System
package installs (CUDA toolkit, Vulkan ICDs) require sudo and are
asked about at TWO levels: first our [y/n/a] prompt, then the
package manager's own 'Proceed with installation? [Y/n]' prompt.
Answer 'n' at either level to skip that step, 'a' to abort.

If you have already reviewed the plan once and just want to re-run
the rebuild on the same machine, answer 'y' at our prompt and then
'Y' at pacman/apt/dnf's prompt — that's two confirmations for two
different systems that will both be changing your machine.

After a successful run, restart with ./start.sh. The startup banner
should now report the GPU backend instead of "cpu".
EOF
}

# --- arg parsing ----------------------------------------------------------
case "${1:-}" in
    --help|-h|help)
        usage
        exit 0
        ;;
    --dry-run)
        DRY_RUN=1
        ;;
    "")
        ;;
    *)
        err "unknown argument: $1"
        usage >&2
        exit 2
        ;;
esac

# --- helpers --------------------------------------------------------------

# Print what would run (dry-run) or actually run it.
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '\033[2m[dry-run]\033[0m %s\n' "$*"
    else
        "$@"
    fi
}

# Ask the user before a mutating step. ALWAYS interactive in real-run
# mode — there is no --yes flag, on purpose. The user wants to be
# asked every time. --dry-run mode never prompts (it just prints
# the plan), so CI can still get a useful preview.
#
# Only valid answers are y / n / a:
#   y = run it
#   n = skip this step (continue script)
#   a = abort the whole script
# Anything else re-prompts. The loop is the right shape: the user
# might fat-finger or be unsure, and we don't want a stray "y" to
# nuke their system. /dev/tty so this still works under
# `curl | bash` style invocations (where stdin is the pipe).
confirm() {
    local prompt=$1
    if [[ $DRY_RUN -eq 1 ]]; then
        return 0
    fi
    if [[ ! -r /dev/tty ]]; then
        err "no TTY available — interactive prompt can't be shown"
        err "this script is interactive-only; run it on a real terminal"
        err "(use --dry-run to print the plan without installing)"
        return 1
    fi
    local reply
    while true; do
        # `printf` rather than `echo` so the prompt isn't subject to
        # the user's shell aliases for echo.
        if ! printf '%s [y/n/a] ' "$prompt" > /dev/tty; then
            err "cannot write to /dev/tty — run this script on a real terminal"
            return 1
        fi
        if ! read -r reply < /dev/tty; then
            err "cannot read from /dev/tty — run this script on a real terminal"
            return 1
        fi
        case "$reply" in
            y|Y|yes|YES) return 0 ;;
            n|N|no|NO)   return 1 ;;
            a|A|abort|ABORT)
                err "aborted by user"
                exit 130
                ;;
            *) printf '   please answer y, n, or a\n' > /dev/tty || true ;;
        esac
    done
}

# Confirm-then-run for system package installs. These are the only
# step that needs sudo, so we surface that explicitly in the prompt
# — the user has to know they'll be entering their password. We do
# NOT pass --noconfirm / -y to the package manager: the package
# manager's own "Proceed with installation? [Y/n]" prompt fires
# after our prompt, so the user is asked twice, at two different
# layers. That's intentional — that's what "ask every time" means.
# If confirm returns 1 (no), we print a warning and the caller
# can continue.
confirm_run() {
    local cmd_summary=$1
    shift
    if ! confirm "about to run: $cmd_summary  (uses sudo)"; then
        warn "skipped: $cmd_summary"
        return 1
    fi
    run "$@"
}

require_venv() {
    if [[ ! -x "$VENV_DIR/bin/pip" ]]; then
        err "no venv at $VENV_DIR/ — run ./start.sh first to create it"
        exit 1
    fi
}

# --- 1. detect platform + GPU + package manager ---------------------------

detect_platform() {
    case "$(uname -s)" in
        Linux)  echo linux ;;
        Darwin) echo macos ;;
        *)      echo other ;;
    esac
}

detect_pkg_manager() {
    if command -v pacman >/dev/null 2>&1; then
        echo pacman
    elif command -v apt >/dev/null 2>&1; then
        echo apt
    elif command -v dnf >/dev/null 2>&1; then
        echo dnf
    elif command -v brew >/dev/null 2>&1; then
        echo brew
    else
        echo unknown
    fi
}

detect_gpu_linux() {
    # `lspci -mm` is the machine-friendly form ("vendor" "device" ...).
    # We grep for VGA / 3D controllers and pick the first vendor. If
    # nvidia-smi is on PATH that's a stronger signal — sometimes the
    # nvidia kernel modules are loaded but the lspci line is buried.
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo nvidia
        return
    fi
    if ! command -v lspci >/dev/null 2>&1; then
        echo no-lspci
        return
    fi
    local line
    line=$(lspci -mm 2>/dev/null | grep -Ei 'VGA|3D' | head -n1 || true)
    if [[ -z "$line" ]]; then
        echo unknown
        return
    fi
    case "$line" in
        *NVIDIA*|*nvidia*) echo nvidia ;;
        *AMD*|*Advanced\ Micro*|*Radeon*|*amd/ati*|*ati*) echo amd ;;
        *Intel*|*intel*)   echo intel ;;
        *ARM*|*Mali*|*arm*) echo arm ;;
        *)                 echo unknown ;;
    esac
}

detect_gpu_macos() {
    # `system_profiler SPDisplaysDataType` prints lines like:
    #   "Chipset Model: Apple M1 Pro"
    #   "Vendor: Apple (0x106b)"
    local info
    info=$(system_profiler SPDisplaysDataType 2>/dev/null || true)
    if echo "$info" | grep -qi 'apple'; then
        # Apple Silicon + Intel Macs both report "Apple" as vendor. Use
        # the CPU brand to disambiguate.
        local brand
        brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
        if echo "$brand" | grep -qi 'apple'; then
            echo apple
        else
            # Intel Mac with an AMD/Radeon dGPU — Vulkan is the best
            # portable backend. (CoreML on Intel Macs exists but is
            # much narrower.)
            echo intel
        fi
        return
    fi
    if echo "$info" | grep -qiE 'amd|radeon'; then
        echo amd
        return
    fi
    if echo "$info" | grep -qiE 'nvidia'; then
        echo nvidia
        return
    fi
    echo unknown
}

prompt_gpu() {
    # Ask the user when detection failed. Tries /dev/tty first so a piped
    # `curl | bash` still works (it'll just fail loudly if there's no
    # interactive input — which is the right behavior).
    local reply
    if [[ -r /dev/tty ]]; then
        read -r -p "No GPU detected. Pick a backend: [n]vidia [a]md [i]ntel [s]kip CPU? " reply < /dev/tty
    else
        warn "no GPU detected and no TTY available — defaulting to CPU-only"
        echo skip
        return
    fi
    case "$reply" in
        n|N|nvidia|NVIDIA) echo nvidia ;;
        a|A|amd|AMD)       echo amd ;;
        i|I|intel|INTEL)   echo intel ;;
        s|S|skip|"")       echo skip ;;
        *)                 warn "unrecognized answer '$reply' — defaulting to skip"; echo skip ;;
    esac
}

# --- 2. install runtime libs per distro -----------------------------------

ensure_lspci() {
    if command -v lspci >/dev/null 2>&1; then
        return
    fi
    warn "lspci is not installed — needed to detect the GPU vendor"
    if ! confirm "install pciutils (provides lspci) via $PKG_MGR?"; then
        warn "skipping lspci install — will fall back to manual vendor prompt"
        return 1
    fi
    case "$PKG_MGR" in
        pacman) confirm_run "sudo pacman -S --needed pciutils" sudo pacman -S --needed pciutils ;;
        apt)    confirm_run "sudo apt install pciutils"               sudo apt install pciutils ;;
        dnf)    confirm_run "sudo dnf install pciutils"               sudo dnf install pciutils ;;
    esac
}

pkg_installed_pacman() {
    pacman -Q "$1" >/dev/null 2>&1
}

pkg_installed_apt() {
    dpkg -s "$1" >/dev/null 2>&1
}

pkg_installed_dnf() {
    rpm -q "$1" >/dev/null 2>&1
}

install_arch() {
    local vendor=$1
    case "$vendor" in
        nvidia)
            local pkgs=()
            pkg_installed_pacman cuda        || pkgs+=(cuda)
            pkg_installed_pacman nvidia-utils || pkgs+=(nvidia-utils)
            pkg_installed_pacman vulkan-icd-loader || pkgs+=(vulkan-icd-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                confirm_run "sudo pacman -S --needed ${pkgs[*]}" \
                    sudo pacman -S --needed "${pkgs[@]}"
            else
                ok "Arch NVIDIA runtime already installed"
            fi
            ;;
        amd)
            local pkgs=()
            pkg_installed_pacman vulkan-radeon      || pkgs+=(vulkan-radeon)
            pkg_installed_pacman vulkan-icd-loader  || pkgs+=(vulkan-icd-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                confirm_run "sudo pacman -S --needed ${pkgs[*]}" \
                    sudo pacman -S --needed "${pkgs[@]}"
            else
                ok "Arch AMD runtime already installed"
            fi
            ;;
        intel)
            local pkgs=()
            pkg_installed_pacman vulkan-intel      || pkgs+=(vulkan-intel)
            pkg_installed_pacman vulkan-icd-loader || pkgs+=(vulkan-icd-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                confirm_run "sudo pacman -S --needed ${pkgs[*]}" \
                    sudo pacman -S --needed "${pkgs[@]}"
            else
                ok "Arch Intel runtime already installed"
            fi
            ;;
        arm)
            local pkgs=()
            pkg_installed_pacman vulkan-mali       || pkgs+=(vulkan-mali)
            pkg_installed_pacman vulkan-icd-loader || pkgs+=(vulkan-icd-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                confirm_run "sudo pacman -S --needed ${pkgs[*]}" \
                    sudo pacman -S --needed "${pkgs[@]}"
            else
                ok "Arch ARM/Mali runtime already installed"
            fi
            ;;
    esac
}

install_deb() {
    local vendor=$1
    case "$vendor" in
        nvidia)
            local pkgs=()
            pkg_installed_apt nvidia-cuda-toolkit || pkgs+=(nvidia-cuda-toolkit)
            pkg_installed_apt nvidia-driver-535   || pkgs+=(nvidia-driver-535)
            pkg_installed_apt vulkan-loader       || pkgs+=(vulkan-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                confirm_run "sudo apt install ${pkgs[*]}" \
                    sudo apt install "${pkgs[@]}"
            else
                ok "Debian/Ubuntu NVIDIA runtime already installed"
            fi
            ;;
        amd|intel)
            local pkgs=()
            pkg_installed_apt mesa-vulkan-drivers || pkgs+=(mesa-vulkan-drivers)
            pkg_installed_apt vulkan-loader       || pkgs+=(vulkan-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                confirm_run "sudo apt install ${pkgs[*]}" \
                    sudo apt install "${pkgs[@]}"
            else
                ok "Debian/Ubuntu Vulkan runtime already installed"
            fi
            ;;
        arm)
            local pkgs=()
            pkg_installed_apt mesa-vulkan-drivers || pkgs+=(mesa-vulkan-drivers)
            pkg_installed_apt vulkan-loader       || pkgs+=(vulkan-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                confirm_run "sudo apt install ${pkgs[*]}" \
                    sudo apt install "${pkgs[@]}"
            else
                ok "Debian/Ubuntu Vulkan runtime already installed"
            fi
            ;;
    esac
}

install_fedora() {
    local vendor=$1
    case "$vendor" in
        nvidia)
            local pkgs=()
            pkg_installed_dnf cuda         || pkgs+=(cuda)
            pkg_installed_dnf akmod-nvidia || pkgs+=(akmod-nvidia)
            pkg_installed_dnf vulkan-loader || pkgs+=(vulkan-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                confirm_run "sudo dnf install ${pkgs[*]}" \
                    sudo dnf install "${pkgs[@]}"
            else
                ok "Fedora NVIDIA runtime already installed"
            fi
            ;;
        amd|intel|arm)
            local pkgs=()
            pkg_installed_dnf mesa-vulkan-drivers || pkgs+=(mesa-vulkan-drivers)
            pkg_installed_dnf vulkan-loader        || pkgs+=(vulkan-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                confirm_run "sudo dnf install ${pkgs[*]}" \
                    sudo dnf install "${pkgs[@]}"
            else
                ok "Fedora Vulkan runtime already installed"
            fi
            ;;
    esac
}

# macOS: drivers are bundled with the OS; nothing to install. The
# pywhispercpp wheel handles the rest. `xcode-select -p` is the only
# thing that could be missing and is checked elsewhere.
install_macos() {
    local vendor=$1
    if ! command -v xcode-select >/dev/null 2>&1; then
        err "xcode-select not found — install Xcode Command Line Tools first:"
        err "    xcode-select --install"
        return 1
    fi
    ok "macOS: no runtime package needed (drivers are part of the OS) for backend=$vendor"
}

# --- 3. rebuild pywhispercpp against the right backend ---------------------

backend_env_for() {
    case "$1" in
        cuda)  echo "GGML_CUDA=1" ;;
        vulkan) echo "GGML_VULKAN=1" ;;
        coreml) echo "WHISPER_COREML=1" ;;
        *)     err "unknown backend: $1"; return 1 ;;
    esac
}

rebuild_pywhispercpp() {
    local backend=$1
    local env_var
    env_var=$(backend_env_for "$backend") || return 1

    step "rebuilding pywhispercpp with $env_var (5-15 minutes on a typical x86 box)"
    # The wheel must be rebuilt with the env var set at *install* time
    # (not at runtime) — that's how pywhispercpp's build script picks up
    # the backend. --force-reinstall + --no-cache so a cached sdist
    # doesn't silently bypass the rebuild.
    #
    # This install is into $VENV_DIR ONLY — system Python is never
    # touched. The venv's pip is at $VENV_DIR/bin/pip and we pass it
    # the full path; we never invoke `pip` unqualified, so there's
    # no risk of accidentally hitting the system interpreter.
    log "pip target: $VENV_DIR/  (system Python untouched)"

    local cmd_desc="env $env_var $VENV_DIR/bin/pip install --force-reinstall --no-cache git+https://github.com/absadiki/pywhispercpp"
    if ! confirm "about to run: $cmd_desc  (this is the slow step — 5-15 min)"; then
        warn "skipped: pip rebuild — the GPU backend will NOT be active until you run this"
        return 1
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        printf '\033[2m[dry-run]\033[0m env %s "%s/bin/pip" install --force-reinstall --no-cache git+https://github.com/absadiki/pywhispercpp\n' \
            "$env_var" "$VENV_DIR"
    else
        # shellcheck disable=SC2086  # env_var is intentionally unquoted
        env $env_var "$VENV_DIR/bin/pip" install --force-reinstall --no-cache \
            git+https://github.com/absadiki/pywhispercpp
    fi
    ok "pywhispercpp rebuilt against $backend"
}

# --- 4. verify -------------------------------------------------------------

verify_backend() {
    local expected=$1
    step "verifying libwhisper.so was built with the $expected backend"
    # The same probe that the test suite uses — gives the user and us
    # one source of truth for "did the build actually pick up the
    # backend?".
    local out
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '\033[2m[dry-run]\033[0m "%s/bin/python" -c "...speakinput.transcriber._probe_gpu_backend()..."\n' "$VENV_DIR"
        return 0
    fi
    out=$("$VENV_DIR/bin/python" - <<'PY' 2>&1
from speakinput.transcriber import _probe_gpu_backend, _gpu_summary
print("detected:", _probe_gpu_backend())
print("banner  :", _gpu_summary(None, 0))
PY
)
    printf '%s\n' "$out"
    local detected
    detected=$(printf '%s\n' "$out" | sed -n 's/^detected: //p')
    if [[ "$detected" == "$expected" ]]; then
        ok "verified: backend=$expected"
    else
        err "verification failed: expected backend=$expected, got '$detected'"
        err "see README → 'GPU acceleration' troubleshooting"
        return 1
    fi
}

# --- 5. main pipeline ------------------------------------------------------

PLATFORM=$(detect_platform)
PKG_MGR=$(detect_pkg_manager)

log "platform=$PLATFORM  pkg_mgr=$PKG_MGR"

case "$PLATFORM" in
    linux)
        # ensure_lspci no-ops if lspci is already installed, prompts
        # the user if not. The GPU detection below will fall back to
        # 'unknown' if the user skips the install (or lspci can't
        # see the GPU for some reason), in which case prompt_gpu
        # takes over.
        ensure_lspci || true
        GPU=$(detect_gpu_linux)
        ;;
    macos)
        GPU=$(detect_gpu_macos)
        ;;
    *)
        err "unsupported platform: $PLATFORM"
        exit 1
        ;;
esac

if [[ "$GPU" == "no-lspci" ]]; then
    # Still no lspci after the install attempt. Fall back to prompt.
    GPU=unknown
fi

if [[ "$GPU" == "unknown" ]]; then
    log "GPU auto-detect failed; asking the user"
    GPU=$(prompt_gpu)
fi

case "$GPU" in
    nvidia) BACKEND=cuda;   ENV_DESC="GGML_CUDA=1" ;;
    amd)    BACKEND=vulkan; ENV_DESC="GGML_VULKAN=1" ;;
    intel)  BACKEND=vulkan; ENV_DESC="GGML_VULKAN=1" ;;
    arm)    BACKEND=vulkan; ENV_DESC="GGML_VULKAN=1" ;;
    apple)  BACKEND=coreml; ENV_DESC="WHISPER_COREML=1" ;;
    skip)
        log "no GPU install performed — pywhispercpp will stay CPU-only"
        log "re-run with a vendor if you change your mind"
        exit 0
        ;;
    *)
        err "unhandled GPU: $GPU"
        exit 1
        ;;
esac

log "GPU=$GPU  backend=$BACKEND  env=$ENV_DESC"

require_venv

# Step 1: runtime libs.
step "installing $BACKEND runtime (vendor=$GPU, pkg_mgr=$PKG_MGR)"
# Print a preview of what the next step will try to install so the
# user sees it in the log before any confirm prompt fires. The
# actual command line is built per-distro and shown again in the
# confirm() prompt, but a one-liner up front helps with scrollback.
case "$GPU:$PKG_MGR" in
    nvidia:pacman) log "will install via pacman: cuda nvidia-utils vulkan-icd-loader (if not present)" ;;
    nvidia:apt)    log "will install via apt:    nvidia-cuda-toolkit nvidia-driver-535 vulkan-loader (if not present)" ;;
    nvidia:dnf)    log "will install via dnf:    cuda akmod-nvidia vulkan-loader (if not present)" ;;
    amd:pacman)    log "will install via pacman: vulkan-radeon vulkan-icd-loader (if not present)" ;;
    amd:apt)       log "will install via apt:    mesa-vulkan-drivers vulkan-loader (if not present)" ;;
    amd:dnf)       log "will install via dnf:    mesa-vulkan-drivers vulkan-loader (if not present)" ;;
    intel:pacman)  log "will install via pacman: vulkan-intel vulkan-icd-loader (if not present)" ;;
    intel:apt)     log "will install via apt:    mesa-vulkan-drivers vulkan-loader (if not present)" ;;
    intel:dnf)     log "will install via dnf:    mesa-vulkan-drivers vulkan-loader (if not present)" ;;
    arm:pacman)    log "will install via pacman: vulkan-mali vulkan-icd-loader (if not present)" ;;
    arm:apt)       log "will install via apt:    mesa-vulkan-drivers vulkan-loader (if not present)" ;;
    arm:dnf)       log "will install via dnf:    mesa-vulkan-drivers vulkan-loader (if not present)" ;;
    *:brew)        log "macOS: no system packages to install (drivers bundled with the OS)" ;;
    *)             log "no system package install planned for this platform/vendor combo" ;;
esac

case "$PKG_MGR" in
    pacman) install_arch   "$GPU" ;;
    apt)    install_deb    "$GPU" ;;
    dnf)    install_fedora "$GPU" ;;
    brew)
        if [[ "$PLATFORM" != "macos" ]]; then
            err "brew is only supported on macOS by this script"
            exit 1
        fi
        install_macos "$GPU"
        ;;
    unknown)
        if [[ "$PLATFORM" == "macos" ]]; then
            install_macos "$GPU"
        else
            err "no supported package manager (pacman/apt/dnf/brew) found"
            err "install your distro's equivalent of vulkan-icd-loader + the"
            err "vendor driver (e.g. cuda / vulkan-radeon / vulkan-intel) by"
            err "hand and re-run"
            exit 1
        fi
        ;;
esac
ok "runtime install step complete"

# Step 2: rebuild pywhispercpp. If the user skipped it (chose 'n'
# at the confirm), there's no point running verify_backend — the
# probe will still say 'cpu' and the user already knows.
PIP_REBUILT=1
if ! rebuild_pywhispercpp "$BACKEND"; then
    PIP_REBUILT=0
    warn "pip rebuild was skipped — GPU backend will not be active"
    warn "re-run ./setup.sh and answer 'y' to the pip rebuild to finish setup"
fi

# Step 3: verify (only if the rebuild actually happened).
if [[ $PIP_REBUILT -eq 1 ]]; then
    verify_backend "$BACKEND"
fi

# Step 4: tell the user what's next.
if [[ $PIP_REBUILT -eq 1 ]]; then
    cat <<EOF

[setup.sh] done.

Next:
  1. Re-run ./start.sh to launch with the new GPU-enabled wheel.
  2. The startup banner should now say:
       [transcribe] $BACKEND (GPU 0, flash_attn=on)
     instead of "cpu (wheel is CPU-only — see README → 'GPU acceleration')".
  3. If it still says "cpu", see README → 'GPU acceleration' troubleshooting.

To re-run this script (e.g. after switching GPUs):
  ./setup.sh

EOF
else
    cat <<EOF

[setup.sh] partial — the runtime install step is done but the pip
rebuild was skipped. The GPU backend is NOT yet active.

To finish:
  1. Re-run ./setup.sh and answer 'y' to the pip rebuild prompt.
     That's the 5-15 minute step.

EOF
fi
