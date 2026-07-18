# Speak Input

Push-to-talk voice transcription that types what you say into the focused field. Think of it as a dictation input method for macOS: hold a key, speak, release, and your words appear where your cursor is.

## Status

v1 — dictation only. Voice commands and live partial transcription are intentionally deferred behind stable interfaces; see "Architecture" below.

## Requirements

- macOS 12+ (Apple Silicon or Intel)
- Python 3.11+
- A working microphone

## Install

```bash
git clone <this-repo> speakinput
cd speakinput
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On first run `./start.sh` copies `config.example.toml` to `~/Library/Application Support/speakinput/config.toml` so you have a discoverable starting point. The program also runs fine with no config file at all — every value in `config.example.toml` is the same as the hard-coded default. Edit the copied file to customize, or leave it alone and use CLI flags (`-m`, `-g`, `-P`, `-S`, etc.) to override per-run. The config file is git-ignored; only `config.example.toml` is committed.

**Want GPU acceleration?** Run `./setup.sh` once after `./start.sh` — it auto-detects your GPU, installs the right runtime, and rebuilds the whisper wheel against the matching backend. See [GPU acceleration](#gpu-acceleration) for what it does and the manual recipe for fine-grained control.

Speak Input then checks whether the configured model file is present. If not, it downloads it from Hugging Face **at startup**, before the push-to-talk listener starts — you will not be surprised by a multi-hundred-MB download in the middle of a recording session. Subsequent runs reuse the cached model and start in seconds.

## Model management

Model files are downloaded into pywhispercpp's cache directory, typically `~/Library/Application Support/pywhispercpp/models/`. By default Speak Input uses the `small` model (~466 MB, multilingual) so it works for English *and* Chinese out of the box. With two profiles, that single `small` file is loaded once and shared between the two `WhisperCppTranscriber` instances — so a typical two-language setup costs ~466 MB resident, not 932 MB. To see the curated model list, run `speakinput -L`.

```bash
speakinput -L                     # show the curated list
speakinput -m base.en             # pick a different model; downloaded on first use
```

If the download fails (no internet, firewall, disk full), Speak Input exits with code 2 and a clear error message — the listener never starts. Fix the underlying problem and re-run.

## GPU acceleration

By default the bundled `pywhispercpp` wheel runs whisper.cpp on the CPU. On Apple Silicon that's fast enough to feel interactive; on x86 Linux with an idle NVIDIA / AMD GPU sitting next to it, CPU-only is much slower than it could be. The startup banner tells you which path is live:

```
[transcribe] cpu (wheel is CPU-only — see README → 'GPU acceleration')
```

If you see `cpu` on Linux and you have a GPU, follow the install instructions below. The rebuild is a one-time per-machine step (5-15 min for the CUDA build, similar for Vulkan). After that, the wheel is GPU-enabled forever — no per-session flag needed, the app auto-detects it.

### Quick setup (recommended)

Run `./setup.sh` once after `./start.sh`. It auto-detects the GPU vendor (via `lspci` on Linux, `system_profiler` on macOS) and the system package manager, installs the right runtime (CUDA toolkit, Vulkan ICD, or CoreML on Apple Silicon), then rebuilds `pywhispercpp` against the matching backend. The script is idempotent — re-running is safe — and is **interactive-only**: it asks before every mutating step at *two* levels, first its own `[y/n/a]` prompt, then the package manager's own `Proceed with installation? [Y/n]` prompt. There is no `--yes` / `--noconfirm` / `-y` flag, by design — the user is asked every time, on every install, and the package manager's prompt is never bypassed. `pip install` always goes into the project's per-host venv (`$VENV_DIR`); the system Python is never touched.

```bash
./setup.sh            # auto-detect, prompt before each install
./setup.sh --dry-run  # preview the plan, no installs (the only way to use it from CI)
```

If `./setup.sh` can't auto-detect (e.g. you're on a headless box, or the GPU is a niche vendor), it'll prompt you for the backend. Or fall back to the manual recipe below.

### Pick a backend

| Backend | Best for | Vendor coverage |
|---|---|---|
| **CUDA** (recommended) | NVIDIA GPUs | NVIDIA only |
| **Vulkan** | AMD, Intel, ARM Mali, NVIDIA | Any vendor |
| **CoreML** | macOS Apple Silicon | Apple only |
| **OpenVINO** | Intel CPUs / GPUs / NPUs | Intel only |
| HIP/ROCm | Older AMD discrete | AMD only |

Pick the first row that matches your hardware. CUDA is the fastest on NVIDIA, Vulkan is the universal fallback. (OpenVINO is listed for Intel hardware but is not auto-probed by this app yet — see [v2 follow-ups](#v2-follow-ups-not-in-this-release).)

### Install commands (Arch Linux)

For **NVIDIA / CUDA** (the recommended path for the user's RTX 4060 Ti):

```bash
# 1. Install the CUDA toolkit + driver
sudo pacman -S cuda                    # pulls in nvidia-utils + headers

# 2. Rebuild pywhispercpp with CUDA support. The GGML_CUDA=1 env
#    var tells the build system to link against cuBLAS. Use --no-cache
#    so pip doesn't reuse the CPU wheel's cached sdist.
GGML_CUDA=1 pip install --force-reinstall --no-cache \
    git+https://github.com/absadiki/pywhispercpp

# 3. Verify the next startup banner says
#    [transcribe] cuda (GPU 0, flash_attn=on)
```

For **Vulkan** (any vendor — works on AMD, Intel, ARM, NVIDIA):

```bash
# 1. Install the vendor driver + the vendor-neutral loader
sudo pacman -S vulkan-icd-loader
sudo pacman -S nvidia-utils             # NVIDIA; vulkan-radeon for AMD,
                                        # vulkan-intel for Intel, vulkan-mali for ARM

# 2. Rebuild pywhispercpp with Vulkan support
GGML_VULKAN=1 pip install --force-reinstall --no-cache \
    git+https://github.com/absadiki/pywhispercpp

# 3. Verify
vulkaninfo --summary                    # should list your GPU
# [transcribe] vulkan (GPU 0, flash_attn=on)
```

For **macOS / CoreML** (Apple Silicon Macs — fastest on M1/M2/M3):

```bash
# Run on the Mac itself; the wheel needs Apple's toolchain.
WHISPER_COREML=1 pip install --force-reinstall --no-cache \
    git+https://github.com/absadiki/pywhispercpp
```

### Per-session controls

Once the wheel is GPU-enabled, the app auto-detects and uses it. You can override per-session without editing `config.toml`:

```bash
./start.sh --gpu                # force GPU on (warns if wheel is CPU-only)
./start.sh --no-gpu             # force CPU (useful for power-saving on laptops)
./start.sh --gpu-device 1       # use the second GPU when multiple are present
./start.sh --threads 16         # use 16 CPU threads for the CPU path
```

Or in `config.toml`:

```toml
[transcribe]
use_gpu = "auto"    # true | false | "auto"  (default: auto = use GPU if wheel has it)
gpu_device = 0      # GPU index, default 0
n_threads = 0       # CPU threads for the CPU path (0 = auto)
```

### V2 follow-ups (not in this release)

- **OpenVINO detection.** Intel's vendor stack (works on Intel CPUs, GPUs, and NPUs) needs a different probe than CUDA/Vulkan because the OpenVINO backend is loaded as a separate `use_openvino=True` path in pywhispercpp. Tracked but not implemented; the README documents the install path for users who want it now.
- **Per-profile GPU device selection** (e.g. GPU 0 for English, GPU 1 for Chinese, on a multi-GPU workstation).
- **CoreML / Apple Neural Engine** is the macOS equivalent of GPU and would be 5-10x faster than the current CPU path on M-series Macs. The probe code already lists `metal` / `coreml` strings as a recognized backend; just needs a `WHISPER_COREML=1` install step documented.
- **An `install-gpu.sh` helper** that auto-detects the GPU vendor via `lspci` and runs the right command. Easy follow-up. *(Shipped as `./setup.sh` — this follow-up is done.)*

## Auto-stop on silence

By default Speak Input stops the recording automatically when **0.8 seconds of consecutive silence** passes while you're holding the key. You don't have to time the release — finish your sentence, pause, and the recorder stops by itself. Two things are wired together:

- **Trailing-silence trim** runs on every release, whether manual or auto. The audio buffer is walked backwards in 30ms hops and any trailing silent portion is dropped, so whisper doesn't see a long silent tail (a real source of hallucinated filler text).
- **Auto-stop watchdog** is a small background thread that polls the recorder's live RMS at ~20Hz. When the configured number of seconds of sub-threshold audio passes, it synthesizes a release for the active profile.

The two related config knobs are both under `[audio]`:

```toml
[audio]
silence_threshold = 0.005   # RMS floor: below this counts as "silence"
auto_stop_seconds = 0.8     # auto-stop after this many seconds of silence
                            # 0 disables; the old "release the key yourself"
                            # behavior comes back
```

Both flags have CLI overrides: `-S/--silence-threshold` and `-A/--auto-stop-seconds`. Lower `auto_stop_seconds` for snappier response; raise it if the watchdog is chopping mid-sentence pauses. Set it to `0` to disable entirely — manual release works exactly as before.

When auto-stop fires, the release path runs the same as a manual release: same buffer-trim step, same silence-gate check, same transcriber. The only difference is who flipped the bit.

### Multi-sentence sessions: chunked re-arm

Auto-stop doesn't end the session. If you keep holding the key, the captured sentence is trimmed, transcribed, and typed — and a fresh watchdog is armed for the **next** sentence. You can dictate a whole paragraph in one key-hold; each sentence lands at the cursor as soon as the silence gap tells the system you're done with it. When you finally release the key, whatever audio was buffered during the last transcribe gets drained and processed as a final chunk, then the recorder tears down.

This is the default behavior; there is no config flag to disable it (set `auto_stop_seconds = 0` if you want the old "release the key yourself" behavior).

## Two profiles, one key per language

Speak Input runs **one or two profiles**. A profile binds one hotkey to one STT setup (model + language + prompt). The typical setup is two profiles, so a single key speaks one language and the model never has to guess:

| Profile   | Default key                    | Default language | Default model |
| --------- | ------------------------------ | ---------------- | ------------- |
| primary   | `alt_r` (macOS) / `ctrl_r` (Linux/Windows) | `auto`  | `small`       |
| secondary | `cmd_r` (macOS) / `cmd_r` (Linux/Windows, pynput maps to Super) | `zh`    | `small`       |

Hold the primary key, speak English, release → English text. Hold the secondary key, speak Chinese, release → Chinese text. Whisper never runs the language-ID pass because the language is already pinned per key.

To use only one language, delete the entire `[profile.secondary]` block in `config.toml` — the right-alt key keeps working, the right-cmd key is not wired, and only the primary model is loaded into RAM.

The two profiles can share the same model file. The default uses `small` (multilingual) for both, and Speak Input loads it into RAM **once** — the `WhisperCppTranscriber` instance is shared between profiles. To speed up the English path, set `primary.model = "small.en"` (note: the secondary profile's language must then be `en` or `auto` to keep the model valid). With one shared `small` model the program uses ~466 MB resident; with `small.en` for English + `small` for Chinese it's ~932 MB.

## Languages

The language is set **per profile**:

```toml
[profile.primary]
language = "auto"   # or "en"

[profile.secondary]
language = "zh"
```

| Value   | What it does                                                | Pairs with                          |
| ------- | ----------------------------------------------------------- | ----------------------------------- |
| `auto`  | Detect the spoken language per utterance (primary default)  | multilingual models (`tiny`/`base`/`small`/`medium`) |
| `en`    | Force English (faster, more accurate than auto for English) | any model                           |
| `zh`    | Force Chinese (Mandarin, secondary default)                 | multilingual models only           |

The trade-off with `auto`: it runs the language identifier on the first 30s of every recording, which adds a small latency hit and is slightly less accurate than explicitly stating the language. With a two-profile setup, you almost never need `auto` — the secondary profile pins `zh` and the primary pins `en` (or `auto` if you want to mix English with occasional other languages). Explicit `language` per profile is the whole point of having two keys.

English-only models (`tiny.en`, `base.en`, `small.en`) are faster but cannot do Chinese. If you set `language = "zh"` on a profile whose model is English-only, Speak Input will refuse to start with a clear error. The same goes the other way: `base.en` paired with `language = "en"` is fine, but `base.en` paired with `language = "auto"` is also allowed and just runs the language ID with an English-only encoder.

## Initial prompt (vocabulary biasing)

`initial_prompt` is set **per profile** in `config.toml` (or `-P` / `--initial-prompt` on the command line, which overrides the primary profile's prompt). It primes whisper's decoder with a fixed text fragment at the start of every transcription. This is a **lexical prior** — it biases the model toward specific vocabulary, but it does not change whisper's behavior the way a chat-model system prompt would.

### Shipped default: embedded software engineer

Out of the box, the program uses this bias — covering languages, RTOSes, MCU concepts, peripherals, protocols, build tools, debug, and C/C++ idioms:

```text
C, C++, Rust, assembly, embedded, firmware, kernel, driver,
RTOS, FreeRTOS, Zephyr, syscall, callback,
microcontroller, ARM Cortex, RISC-V, register, peripheral,
clock, PLL, prescaler, watchdog,
ROM, RAM, flash, EEPROM, heap, stack, allocator,
interrupt, ISR, IRQ, exception, fault, hardfault,
scheduler, mutex, semaphore, spinlock, atomic, preemptive,
GPIO, UART, SPI, I2C, I2S, CAN, USB, Ethernet,
ADC, DAC, PWM, timer, DMA, FIFO,
ack, nack, CRC, checksum, parity, packet, frame, payload,
endian, alignment, bitfield,
block, sector, page, erase, program, mount,
gcc, clang, cmake, ninja, linker, cross-compile,
elf, hex, optimization,
GDB, OpenOCD, JTAG, SWD, trace, profiler,
core dump, stack trace, backtrace,
volatile, const, static, inline, extern, weak, packed,
typedef, struct, union, enum, macro, pragma,
void, NULL, nullptr, true, false,
printf, sprintf, malloc, free, memcpy, memset, strlen
```

Why this is the default: in day-to-day embedded dictation, the same words keep coming up — peripheral acronyms, RTOS primitives, build/toolchain terms, fault types. Whisper's raw small model mangles them ("D M A controller", "you art one", "hard fault"). Pre-seeding the decoder with the right tokens fixes almost all of these. If you say *"configure the DMA controller for UART TX"*, the output is now exactly that — no post-edit needed.

The list deliberately avoids product names (STM32, ESP32, …) and `_t` type suffixes (`uint32_t`, etc.) so it stays useful across vendors. See the [full config default](src/speakinput/config.py) if you want to add or remove entries.

If your work isn't embedded, see [Switching to a different domain](#switching-to-a-different-domain) below.

### What it can bias for you

- **Names** that whisper would otherwise misspell: `"Shaowu"`, `"Karpathy"`, product/team names.
- **Acronyms**: `"K8s, SRE, PR, kubectl"`.
- **Technical jargon**: `"kubectl apply -f deployment.yaml"`, `"semver: 1.2.3-rc.1"`.
- **Style hints**: `"Use British English."`, `"Use semicolons."`.

### What it can't do

- **Behavioral directives** ("always be concise"). Whisper doesn't follow instructions; it transcribes.
- **Long passages.** The prompt is tokenized at start; very long prompts hit whisper's 224-token limit and may confuse the decoder for unrelated speech. The default is ~120 tokens, leaving headroom.
- **Generic text.** A prompt like `"hello world"` doesn't help anything and may bias the decoder toward outputting "hello" regardless of what you actually said.

### Switching to a different domain

Set `initial_prompt` in `config.toml` to a comma-separated list of the words you actually say. Some examples:

```toml
# Web / DevOps
initial_prompt = "Kubernetes, K8s, kubectl, Docker, Dockerfile, Helm, Terraform, AWS, S3, EC2, Lambda, CI/CD, GitHub Actions, Grafana, Prometheus, nginx, Postgres, Redis, gRPC, REST, JSON, YAML, semver"

# Data science / ML
initial_prompt = "PyTorch, TensorFlow, NumPy, pandas, scikit-learn, Jupyter, GPU, CUDA, TPU, transformer, attention, embedding, fine-tune, LoRA, RAG, vector database, Weights and Biases"

# Embedded (the default — shown for reference; product names left out
# so the bias stays useful across vendors)
initial_prompt = "C, C++, Rust, RTOS, FreeRTOS, GPIO, UART, SPI, I2C, DMA, ISR, scheduler, mutex, GDB, OpenOCD, JTAG"
```

Or override per-run with `-P`:

```bash
./start.sh -P "kubectl apply -f deployment.yaml, Helm, Terraform, K8s"
```

### Disabling the bias entirely

Set `initial_prompt = ""` in `config.toml` (empty string). Useful if you dictate in mixed/unpredictable domains and don't want any prior pulling the decoder one way or another.

```toml
[profile.primary]
initial_prompt = ""
```

Note: passing `-P ""` on the command line is awkward (the shell swallows the empty argument). Prefer the config file for disabling.

### Verifying the bias is active

Run `./start.sh -d` and look for the startup banner line `profile 1 : ... prompt=set` — that confirms a non-empty `initial_prompt` is reaching the primary profile's transcriber. With `initial_prompt = ""` it reads `prompt=off`. Same for `profile 2` if you have a secondary profile.

## macOS permissions

macOS splits the permissions for what Speak Input does into **two separate gates** in System Settings. This trips up a lot of people because getting one right isn't enough — both are required, and they have to be granted to the same executable.

| What Speak Input does | Permission needed | System Settings path |
| --- | --- | --- |
| **Detect** your push-to-talk key | Input Monitoring | Privacy & Security → Input Monitoring |
| **Send** keystrokes to the focused app | Accessibility | Privacy & Security → Accessibility |

You will know **only** Input Monitoring is granted if the push-to-talk key works for detection but the transcript never appears in your target app (or no warning shows up). The startup log line `pynput.keyboard.Listener WARNING: This process is not trusted!` is the smoking gun for the Accessibility gate.

### Setup

1. Open **System Settings → Privacy & Security**. Scroll down to **Accessibility** and to **Input Monitoring** — there are two separate lists.
2. Click the lock icon and authenticate to make changes.
3. For each of those two lists, click `+` to add an entry. In the file picker, press `Cmd+Shift+.` to reveal hidden files, then navigate to the **python binary inside the venv**:

   ```
   /Volumes/workspace/projects/speakinput/.venv/bin/python3
   ```

   (or `python3.14`, `python3.12` — whichever your venv created). Select it and click **Open**.
4. Toggle the switch **ON** for that entry in each list. Some macOS versions require you to flip the switch, not just add the file.
5. **Log out and back in.** Apple's recent macOS versions often don't honor newly-added Accessibility entries until the next login. A reboot is more reliable.

If the file picker won't let you select the hidden `.venv` python, the fallback is to also grant access to **Terminal.app** (or iTerm, Warp, VS Code — whatever you launched `./start.sh` from). The python process sometimes inherits the parent's permissions. The path is `/Applications/Utilities/Terminal.app`.

### Verifying permissions

Run `./start.sh --debug` and look for the line that says either `This process is not trusted!` (a permission is missing) or nothing of the sort (both gates are open). If you only see the warning about Input Monitoring, you have Accessibility and the warning is benign; if you see the Accessibility-style warning, re-check step 4 above.

### "It worked, then stopped" — TCC resets

macOS sometimes resets these permissions after a macOS update, after major Python upgrades (e.g. 3.13 → 3.14), or after the `.venv` is rebuilt. The venv python path changes, and the old permission entry no longer matches. Re-run steps 3–5 and the issue goes away.

## Linux / Wayland

Speak Input auto-detects Wayland sessions (`XDG_SESSION_TYPE=wayland`) and uses a second hotkey listener that reads the Linux kernel input subsystem directly via `python-evdev`, bypassing pynput's X11-only backend. On X11 Linux (or any Linux session where pynput can reach a display) it uses pynput as before.

No configuration is required to enable this — it's a runtime decision based on the session type. The startup banner line `[startup] hotkey   : evdev (Linux Wayland session — pynput bypassed)` confirms the evdev backend is active.

### Permissions

The evdev backend reads `/dev/input/eventN` directly, so the user must be able to open those files. On most distros this is handled by membership in the `input` group:

```bash
sudo usermod -aG input $USER
# log out and back in (group membership is applied at login)
```

Verify with `ls -l /dev/input/event*` — your user should be able to `cat` one of them (Ctrl-C to stop). If `/dev/input` doesn't exist at all, you're in a sandboxed/container environment without kernel input devices exposed; this is the runtime's problem, not the program's.

### Text injection (the typing part)

On Wayland, the output side is also auto-detected. Speak Input picks the first available of:

1. **`wtype`** — uses the wlroots `virtual-keyboard-unstable-v1` protocol directly. Daemon-free, works on any wlroots-based compositor (Sway, swayfx, Hyprland, river). Not supported on GNOME or KDE. Install: `pacman -S wtype` (Arch), `apt install wtype` (Debian/Ubuntu).
2. **`ydotool`** — uinput-based, needs the `ydotoold` daemon. Install: `pacman -S ydotool` (Arch), `apt install ydotool` (Debian), then `systemctl --user enable --now ydotool` to start the daemon. Works on more compositors (including eventually GNOME via libei).
3. **`pynput`** — last-resort fallback. Only works through XWayland, so it does nothing useful on a pure-Wayland session; if you see this in the startup banner, install wtype or ydotool.

The startup banner shows the chosen backend. Override with `[inject].backend = "wtype" | "ydotool" | "pynput"` in `config.toml`.

For the Unicode (CJK, accented) path, speakinput also needs a working clipboard tool. On Wayland that's `wl-copy` / `wl-paste` from the `wl-clipboard` package (`pacman -S wl-clipboard`, `apt install wl-clipboard`). The user's prior clipboard contents are restored after each injection.

### Double-fires with other key-grabbers

The evdev backend does **not** use `EVIOCGRAB` — it observes events while the focused application also receives them, so the user can type normally in other apps while speakinput is running. The trade-off: if another key-grabber is also watching the same device (e.g. `sxhkd`, `kglobalacceld`, or a screen-reader), it will see the same hotkey press and may act on it too. If you see the hotkey fire twice (or other odd behavior), check your other key-grabbers and either disable them for the configured key or pause the other process while dictating.

The right-Super physical key (`cmd_r` on a default Linux profile) may not exist on every keyboard. If the configured `cmd_r` key doesn't trigger, edit `config.toml` and switch the secondary profile's key to something your keyboard has, e.g. `key = "ctrl_r"` (right-Ctrl) or `key = "f12"`.

### v2 follow-ups (not in this release)

- `[hotkey].device_path` config knob for multi-keyboard setups (laptop + external USB). The current auto-detect picks the first keyboard-shaped device it finds; users with multiple keyboards who want a specific one selected will need this.
- Optional `EVIOCGRAB` mode for users who want exclusive device grab (toggle via config). Documented as a non-default because the cost (you can't type anywhere else while speakinput is running) is heavy for a background push-to-talk app.

## Usage

```bash
speakinput                       # run with default config
speakinput -m base.en            # override the model for this run
speakinput -g zh                 # force Chinese transcription
speakinput -g en                 # force English (skips language ID)
speakinput -l                     # show available input devices
speakinput -D                     # record 2s, print audio stats, don't inject
speakinput -n                     # print transcribed text to stderr instead of typing
speakinput -T                     # don't append a space after each transcript
speakinput -c ./my.toml           # use a custom config file
speakinput -d                     # debug mode: log every key event and transcript
```

### All flags

| Short | Long                  | What it does                                    |
| ----- | --------------------- | ----------------------------------------------- |
| `-c`  | `--config PATH`       | Path to config.toml                             |
| `-m`  | `--model NAME`        | Override the **primary** profile's whisper model (`tiny`/`base`/`small`/`medium`/`.en` variants) |
| `-g`  | `--language CODE`     | Override the **primary** profile's language (`auto` / `en` / `zh`) |
| `-l`  | `--list-devices`      | List available input devices and exit           |
| `-L`  | `--list-models`       | List curated whisper models and exit            |
| `-D`  | `--diagnose`          | Record 2s and print audio stats                 |
| `-n`  | `--no-inject`         | Print transcript to stderr instead of typing it |
| `-d`  | `--debug`             | Log every key event and transcript to stderr    |
| `-t`  | `--trailing-space`    | Append a space after each transcript (default)  |
| `-T`  | `--no-trailing-space` | Don't append a space after each transcript      |
| `-S`  | `--silence-threshold FLOAT` | Skip transcribe when audio RMS is below this floor (0 disables; default 0.005) |
| `-A`  | `--auto-stop-seconds FLOAT` | Auto-stop after this many seconds of silence while the key is held (0 disables; default 0.8). Trailing silence is also trimmed from the buffer before transcribe. See [Auto-stop on silence](#auto-stop-on-silence) |
| `-P`  | `--initial-prompt TEXT` | Override the **primary** profile's `initial_prompt` for this run (default: embedded-software-engineer bias; see [Initial prompt](#initial-prompt-vocabulary-biasing)) |
| `-v`  | `--verbose`           | Enable debug logging from python logging        |

CLI flags apply to the **primary** profile only. The secondary profile is configured in `config.toml` — there's no `--secondary-*` flag by design, since the secondary's whole point is a stable per-machine config.

By default the push-to-talk key is **Right Option (Alt)** on macOS and **Right Ctrl** on Linux/Windows. The choice is platform-aware — Alt is heavily used for menu mnemonics on PC desktops, so the default flips to Right Ctrl there. Hold the key, speak, release. The recognized text is typed into whatever field has focus, **with a trailing space** so the next word doesn't run into the last one. Disable the trailing space with `--no-trailing-space` if you want pure dictation (e.g. when typing into a code editor).

To change the hotkey(s), edit the config:

```toml
[profile.primary]
key = "alt_r"        # alt_r | ctrl_r | cmd_r | shift_r | caps_lock | f12

[profile.secondary]
key = "cmd_r"        # only read if [profile.secondary] is present
```

To change the model and language, edit the per-profile sections:

```toml
[profile.primary]
model = "small"        # tiny.en | base.en | small.en | tiny | base | small | medium
language = "auto"      # auto | en | zh

[profile.secondary]
model = "small"
language = "zh"
```

| Model    | Size  | Speed (M1) | Languages                |
| -------- | ----- | ---------- | ------------------------ |
| tiny.en  | 75 MB | ~real-time | English (fastest)        |
| base.en  | 142 MB| ~real-time | English                  |
| small.en | 466 MB| ~2x slower | English (best `.en`)     |
| tiny     | 75 MB | ~real-time | multilingual             |
| base     | 142 MB| ~real-time | multilingual             |
| small    | 466 MB| ~2x slower | multilingual (default)   |
| medium   | 1.5 GB| ~5x slower | multilingual (best)      |

The first time you select a model, pywhispercpp downloads it to `~/.cache/pywhispercpp/`.

## Configuration reference

The shipped `config.example.toml` is the source of truth — every field shown there is also the program's default. Copy it to the user config dir (`./start.sh` does this on first run) and uncomment/edit the lines you care about; anything left out falls back to the default.

```toml
[profile.primary]
key = "alt_r"           # alt_r | ctrl_r | cmd_r | shift_r | caps_lock | f12
                        # Default is platform-aware: alt_r on macOS,
                        # ctrl_r on Linux/Windows.
model = "small"         # whisper.cpp model; see model table below
language = "auto"       # auto | en | zh
beam_size = 1           # 1 = greedy (fastest); up to 10 for higher accuracy
# initial_prompt = ""   # default is an embedded-software-engineer bias;
                        # set to "" to disable. See "Initial prompt" below.

# Optional. Delete the entire [profile.secondary] block to run with
# one key only. Default key is cmd_r; default language is zh.
[profile.secondary]
key = "cmd_r"
model = "small"
language = "zh"
beam_size = 1
# initial_prompt = ""   # per-profile; same bias as primary by default

[audio]
device = null              # null = system default mic; or an integer index from --list-devices
sample_rate = 16000        # whisper expects 16 kHz; do not change
silence_threshold = 0.005  # RMS floor used by the silence gate and the
                           # auto-stop watchdog. Audio below this counts
                           # as silence. 0 disables both.
auto_stop_seconds = 0.8    # auto-stop after this many seconds of
                           # continuous silence while the key is held.
                           # Trailing silence is also trimmed from the
                           # buffer before transcribe. 0 disables.

[inject]
restore_clipboard_ms = 50  # how long to wait before restoring the prior clipboard contents
                           # (only relevant when injecting Unicode text)
trailing_space = true      # append a single space after each transcript
                           # (set false for code editors where you don't want auto-spacing)
```

To customize manually:

```bash
cp config.example.toml ~/Library/Application\ Support/speakinput/config.toml
$EDITOR ~/Library/Application\ Support/speakinput/config.toml
```

## How it works

```
hotkey press   →  look up profile by key  →  AudioRecorder starts capturing
                →  silence watchdog starts polling live RMS
                →  if 0.8s of sub-threshold audio passes → auto-release
                →  if hotkey is released manually first → cancel watchdog
hotkey release →  AudioRecorder stops
              →  trim trailing silence from the buffer (30ms hops)
              →  silence gate: if whole buffer's RMS < threshold, skip
              →  profile.transcriber.transcribe(buffer)
              →  TypingInjector types the result (or pastes via clipboard for non-ASCII)
```

The App keeps a `key → profile` map. When a key fires, the matching profile's transcriber runs (with that profile's language + initial_prompt). When two profiles share a model file, they share a `WhisperCppTranscriber` instance — one model load, ~466 MB resident, two keys.

Three interfaces — `Recorder`, `Transcriber`, `Injector` — are stable seams. v2 will add:

- A `StreamingTranscriber` that consumes `AudioRecorder.chunk_generator()` for partial results while the key is still held (the "overlapped streaming" goal from the original design).
- A `CommandInjector` that interprets the transcription and dispatches to shell or an agent.
- A streaming partial-results UI as you hold the key.

## Troubleshooting

**Key detection works but transcript never appears in the focused app.** This is the Accessibility permission — see the [macOS permissions](#macos-permissions) section. Run with `--debug` to confirm: the `[debug] key press end` and `[debug] transcript: '...'` lines will appear, but the receiving app stays empty. Add the venv python to **Privacy & Security → Accessibility**, log out/in, and try again.

**Nothing happens when I hold the hotkey.** On macOS, check both macOS permissions above. On Linux, verify the user is in the `input` group (see [Linux / Wayland](#linux--wayland)), and that `/dev/input/event*` is readable. Run with `-v` to see debug logs.

**On Linux Wayland, the listener fails to start with "no keyboard device found in /dev/input".** Either the user isn't in the `input` group, or `/dev/input` is empty (sandboxed/container env). See [Linux / Wayland](#linux--wayland).

**On Linux, the hotkey fires twice or other apps also see it.** Another key-grabber is also watching the device. The evdev backend intentionally does not grab exclusively — see the [double-fires](#double-fires-with-other-key-grabbers) note for workarounds.

**Transcription is much slower than on macOS.** The shipped `pywhispercpp` wheel is CPU-only. On Linux x86 that runs whisper.cpp on the CPU cores only, which is much slower than the Apple Silicon M-series path. Rebuild pywhispercpp against CUDA (NVIDIA) or Vulkan (any vendor) — see the [GPU acceleration](#gpu-acceleration) section.

**The startup banner says `cpu (wheel is CPU-only)` even after I ran the install command.** Check the build log for `error: failed to find cuBLAS` (CUDA) or `error: Vulkan headers not found` (Vulkan). The most common cause on Arch is missing the `cuda` / `vulkan-headers` package; the second most common is pip reusing a cached sdist (`--no-cache` fixes that).

**On Linux Wayland, the transcript shows up in debug output but nothing is typed into the focused app.** pynput is X11-only — the output side needs a Wayland-native typing tool. Install `wtype` (preferred, daemon-free) or `ydotool` + start `ydotoold`. See the [Text injection](#text-injection-the-typing-part) section above.

**The right-Option hotkey (macOS default) triggers menu mnemonics.** v1 uses `suppress=False` so the key reaches other apps — useful for Alt-Tab, but it can arm menu shortcuts in some apps. v2 will add a `suppress=True` mode for that case. On Linux/Windows the default is Right Ctrl, which has fewer menu-mnemonic conflicts.

**`pywhispercpp` fails to load the model.** Make sure the model name matches exactly (case-sensitive, e.g. `base.en` not `Base.en`). Run `speakinput --diagnose` to surface the error directly.

**Transcription is empty / nonsense.** Try `small` (or `medium`) for noisy environments. Make sure `--list-devices` is showing the right mic; set `[audio].device` to its index.

**The same phrase appears multiple times in the focused field.** Two processes are listening to the hotkey. The app refuses to start a second instance — the new process exits with code 3 and a clear error message pointing at the lockfile. Check `ps aux | grep speakinput` and kill any leftover processes. A common cause is starting the app, getting distracted, then starting it again from another terminal — every instance registers a hotkey listener and your single key release fans out to all of them.

**A random short phrase appears when I didn't say anything / accidentally tapped the hotkey.** Whisper hallucinates on near-empty audio — the silence gate should catch this. If you still see phantom text, your environment may be noisy enough that the RMS exceeds the default `0.005` floor. Lower it: `speakinput -S 0.01` or set `[audio].silence_threshold = 0.01` in config.toml. Set to `0` to disable the gate entirely (whisper will see every recording, including silence).

**Output contains "DMA", "FreeRTOS", "OpenOCD" etc. that I didn't say.** The shipped `initial_prompt` biases whisper toward embedded-software vocabulary — see [Initial prompt](#initial-prompt-vocabulary-biasing). It's a one-shot token prior: whisper overweights those words because they appear in the seed. If your dictation isn't embedded (or you want mixed/general speech), either set `[profile.primary].initial_prompt = ""` (and the same for `secondary` if you have one) in `config.toml` to disable the bias, or pass a domain-appropriate comma-separated list of the words you actually use. The phantom outputs are strongest right after the prompt tokens, weakest mid-sentence.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Building a release binary

`release.sh` builds a single-file frozen binary with PyInstaller for the **current host** (macOS arm64/x86_64, Linux x86_64). Cross-compiling isn't supported — run it on the host you want to ship for.

```bash
./release.sh             # build into dist/speakinput (~15 MB macOS arm64)
CLEAN=1 ./release.sh     # wipe build/ + dist/ first
./dist/speakinput --list-models
```

The build is reproducible via `speakinput.spec` (the source of truth for hidden imports and excludes). The output `dist/speakinput` is a self-contained binary — no Python install needed on the target host. Whisper models are downloaded on first run into the user's `pywhispercpp` cache, exactly like `pip install` would.

**Distribute** by tarring `dist/speakinput` + `dist/config.example.toml` + `dist/README.md`. The user untars, runs `./speakinput`, and follows the same permission setup as the `pip install` flow. macOS Gatekeeper may reject the unsigned binary on first run — right-click → Open, or sign with `codesign --force --deep --sign - dist/speakinput`.

## License

MIT
