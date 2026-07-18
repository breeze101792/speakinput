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
                       Useful for: (a) previewing what the script will do
                       on this box, (b) running in CI where the actual
                       install would fail anyway.

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

run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '\033[2m[dry-run]\033[0m %s\n' "$*"
    else
        "$@"
    fi
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
    case "$PKG_MGR" in
        pacman) run sudo pacman -S --noconfirm --needed pciutils ;;
        apt)    run sudo apt install -y pciutils ;;
        dnf)    run sudo dnf install -y pciutils ;;
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
                run sudo pacman -S --noconfirm --needed "${pkgs[@]}"
            else
                ok "Arch NVIDIA runtime already installed"
            fi
            ;;
        amd)
            local pkgs=()
            pkg_installed_pacman vulkan-radeon      || pkgs+=(vulkan-radeon)
            pkg_installed_pacman vulkan-icd-loader  || pkgs+=(vulkan-icd-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                run sudo pacman -S --noconfirm --needed "${pkgs[@]}"
            else
                ok "Arch AMD runtime already installed"
            fi
            ;;
        intel)
            local pkgs=()
            pkg_installed_pacman vulkan-intel      || pkgs+=(vulkan-intel)
            pkg_installed_pacman vulkan-icd-loader || pkgs+=(vulkan-icd-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                run sudo pacman -S --noconfirm --needed "${pkgs[@]}"
            else
                ok "Arch Intel runtime already installed"
            fi
            ;;
        arm)
            local pkgs=()
            pkg_installed_pacman vulkan-mali       || pkgs+=(vulkan-mali)
            pkg_installed_pacman vulkan-icd-loader || pkgs+=(vulkan-icd-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                run sudo pacman -S --noconfirm --needed "${pkgs[@]}"
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
                run sudo apt install -y "${pkgs[@]}"
            else
                ok "Debian/Ubuntu NVIDIA runtime already installed"
            fi
            ;;
        amd|intel)
            local pkgs=()
            pkg_installed_apt mesa-vulkan-drivers || pkgs+=(mesa-vulkan-drivers)
            pkg_installed_apt vulkan-loader       || pkgs+=(vulkan-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                run sudo apt install -y "${pkgs[@]}"
            else
                ok "Debian/Ubuntu Vulkan runtime already installed"
            fi
            ;;
        arm)
            local pkgs=()
            pkg_installed_apt mesa-vulkan-drivers || pkgs+=(mesa-vulkan-drivers)
            pkg_installed_apt vulkan-loader       || pkgs+=(vulkan-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                run sudo apt install -y "${pkgs[@]}"
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
                run sudo dnf install -y "${pkgs[@]}"
            else
                ok "Fedora NVIDIA runtime already installed"
            fi
            ;;
        amd|intel|arm)
            local pkgs=()
            pkg_installed_dnf mesa-vulkan-drivers || pkgs+=(mesa-vulkan-drivers)
            pkg_installed_dnf vulkan-loader        || pkgs+=(vulkan-loader)
            if [[ ${#pkgs[@]} -gt 0 ]]; then
                run sudo dnf install -y "${pkgs[@]}"
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
        if ! command -v lspci >/dev/null 2>&1; then
            warn "lspci not found — installing pciutils"
            ensure_lspci
        fi
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

# Step 2: rebuild pywhispercpp.
rebuild_pywhispercpp "$BACKEND"

# Step 3: verify.
verify_backend "$BACKEND"

# Step 4: tell the user what's next.
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
