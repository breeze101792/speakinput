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

Speak Input then checks whether the configured model file is present. If not, it downloads it from Hugging Face **at startup**, before the push-to-talk listener starts — you will not be surprised by a multi-hundred-MB download in the middle of a recording session. Subsequent runs reuse the cached model and start in seconds.

## Model management

The model is downloaded into pywhispercpp's cache directory, typically `~/Library/Application Support/pywhispercpp/models/`. By default Speak Input uses the `small` model (~466 MB, multilingual) so it works for English *and* Chinese out of the box. To see the curated model list, run `speakinput -L`.

```bash
speakinput -L                     # show the curated list
speakinput -m base.en             # pick a different model; downloaded on first use
```

If the download fails (no internet, firewall, disk full), Speak Input exits with code 2 and a clear error message — the listener never starts. Fix the underlying problem and re-run.

## Languages

`stt.language` in `config.toml` controls what language the model transcribes:

| Value   | What it does                                                | Pairs with                          |
| ------- | ----------------------------------------------------------- | ----------------------------------- |
| `auto`  | Detect the spoken language per utterance (default)          | multilingual models (`tiny`/`base`/`small`/`medium`) |
| `en`    | Force English (faster, more accurate than auto for English) | any model                           |
| `zh`    | Force Chinese (Mandarin)                                    | multilingual models only           |

With the default `model = "small"` and `language = "auto"`, you can switch between English and Chinese mid-session without restarting. The trade-off: `auto` runs the language identifier on the first 30s of every recording, which adds a small latency hit and is slightly less accurate than explicitly stating the language. If you only ever dictate in one language, set `language` explicitly and you'll get better results.

English-only models (`tiny.en`, `base.en`, `small.en`) are faster but cannot do Chinese. If you set `language = "zh"` with an English-only model, Speak Input will refuse to start with a clear error.

## Initial prompt (vocabulary biasing)

`stt.initial_prompt` in `config.toml` (or `-P` / `--initial-prompt` on the command line) primes whisper's decoder with a fixed text fragment at the start of every transcription. This is a **lexical prior** — it biases the model toward specific vocabulary, but it does not change whisper's behavior the way a chat-model system prompt would.

The shipped default is an embedded-software-engineer bias — C/C++/Rust, MCU names (STM32, ESP32, ARM Cortex), RTOS terms (FreeRTOS, Zephyr, scheduler, mutex, semaphore), peripherals (GPIO, UART, SPI, I2C, DMA, ADC, PWM), debug tools (JTAG, SWD, OpenOCD, GDB), types (`uint32_t`, `size_t`, `bool`), and common idioms (`printf`, `malloc`, `0x`). This dramatically improves recognition for typical embedded dictation out of the box: "configure the DMA controller for UART TX" comes out without mangling the acronyms.

Use it for:

- **Names** that whisper would otherwise misspell: `"Shaowu"`, `"Karpathy"`, product/team names.
- **Acronyms**: `"K8s, SRE, PR, kubectl"`.
- **Technical jargon**: `"kubectl apply -f deployment.yaml"`, `"semver: 1.2.3-rc.1"`.
- **Style hints**: `"Use British English."`, `"Use semicolons."`.

Do not use it for:

- Behavioral directives ("always be concise"). Whisper doesn't follow instructions; it transcribes.
- Long passages. The prompt is tokenized at start; very long prompts hit whisper's 224-token limit and may confuse the decoder for unrelated speech.
- Generic text. A prompt like `"hello world"` doesn't help anything and may bias the decoder toward outputting "hello" regardless of what you actually said.

To **disable** the default bias, set `initial_prompt = ""` in `config.toml`, or override per-run with an empty value (you cannot easily pass an empty string via `-P`; prefer the config file for disabling). To use a different bias, replace the value in `config.toml` or pass it via `-P` on the command line:

```bash
./start.sh -P "K8s, SRE, kubectl, Dockerfile, semver"
```

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
| `-m`  | `--model NAME`        | Override the whisper model (`tiny`/`base`/`small`/`medium`/`.en` variants) |
| `-g`  | `--language CODE`     | Override stt.language (`auto` / `en` / `zh`)               |
| `-l`  | `--list-devices`      | List available input devices and exit           |
| `-L`  | `--list-models`       | List curated whisper models and exit            |
| `-D`  | `--diagnose`          | Record 2s and print audio stats                 |
| `-n`  | `--no-inject`         | Print transcript to stderr instead of typing it |
| `-d`  | `--debug`             | Log every key event and transcript to stderr    |
| `-t`  | `--trailing-space`    | Append a space after each transcript (default)  |
| `-T`  | `--no-trailing-space` | Don't append a space after each transcript      |
| `-S`  | `--silence-threshold FLOAT` | Skip transcribe when audio RMS is below this floor (0 disables; default 0.005) |
| `-P`  | `--initial-prompt TEXT` | Whisper initial_prompt — bias the decoder toward specific vocabulary |
| `-v`  | `--verbose`           | Enable debug logging from python logging        |

By default the push-to-talk key is **Right Option (Alt)**. Hold it, speak, release. The recognized text is typed into whatever field has focus, **with a trailing space** so the next word doesn't run into the last one. Disable the trailing space with `--no-trailing-space` if you want pure dictation (e.g. when typing into a code editor).

To change the hotkey, edit the config:

```toml
[hotkey]
key = "alt_r"        # alt_r | ctrl_r | cmd_r | shift_r | caps_lock | f12
```

To change the model:

```toml
[stt]
model = "small"        # tiny.en | base.en | small.en | tiny | base | small | medium
language = "auto"      # auto | en | zh
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
[stt]
model = "small"            # whisper.cpp model; see model table below
language = "auto"          # auto | en | zh
beam_size = 1              # 1 = greedy (fastest); up to 10 for higher accuracy
# initial_prompt = ""      # optional: bias the decoder toward specific vocabulary
                           # (see "Initial prompt" below)

[audio]
device = null              # null = system default mic; or an integer index from --list-devices
sample_rate = 16000        # whisper expects 16 kHz; do not change
silence_threshold = 0.005  # skip transcribe when audio RMS is below this floor
                           # (0 disables the gate; lower = more sensitive)

[hotkey]
key = "alt_r"              # see valid keys above

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
hotkey press   →  AudioRecorder starts capturing
hotkey release →  AudioRecorder stops
              →  silence gate: if RMS < threshold, skip (no hallucination)
              →  WhisperCppTranscriber transcribes the buffer
              →  TypingInjector types the result (or pastes via clipboard for non-ASCII)
```

Three interfaces — `Recorder`, `Transcriber`, `Injector` — are stable seams. v2 will add:

- A `StreamingTranscriber` that consumes `AudioRecorder.chunk_generator()` for partial results while the key is still held (the "overlapped streaming" goal from the original design).
- A `CommandInjector` that interprets the transcription and dispatches to shell or an agent.
- A streaming partial-results UI as you hold the key.

## Troubleshooting

**Key detection works but transcript never appears in the focused app.** This is the Accessibility permission — see the [macOS permissions](#macos-permissions) section. Run with `--debug` to confirm: the `[debug] key press end` and `[debug] transcript: '...'` lines will appear, but the receiving app stays empty. Add the venv python to **Privacy & Security → Accessibility**, log out/in, and try again.

**Nothing happens when I hold the hotkey.** Check both macOS permissions above. Run with `-v` to see debug logs.

**The right-Option hotkey triggers menu mnemonics.** v1 uses `suppress=False` so the key reaches other apps — useful for Alt-Tab, but it can arm menu shortcuts in some apps. v2 will add a `suppress=True` mode for that case.

**`pywhispercpp` fails to load the model.** Make sure the model name matches exactly (case-sensitive, e.g. `base.en` not `Base.en`). Run `speakinput --diagnose` to surface the error directly.

**Transcription is empty / nonsense.** Try `small` (or `medium`) for noisy environments. Make sure `--list-devices` is showing the right mic; set `[audio].device` to its index.

**The same phrase appears multiple times in the focused field.** Two processes are listening to the hotkey. The app refuses to start a second instance — the new process exits with code 3 and a clear error message pointing at the lockfile. Check `ps aux | grep speakinput` and kill any leftover processes. A common cause is starting the app, getting distracted, then starting it again from another terminal — every instance registers a hotkey listener and your single key release fans out to all of them.

**A random short phrase appears when I didn't say anything / accidentally tapped the hotkey.** Whisper hallucinates on near-empty audio — the silence gate should catch this. If you still see phantom text, your environment may be noisy enough that the RMS exceeds the default `0.005` floor. Lower it: `speakinput -S 0.01` or set `[audio].silence_threshold = 0.01` in config.toml. Set to `0` to disable the gate entirely (whisper will see every recording, including silence).

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT
