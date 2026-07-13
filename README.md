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

On first run Speak Input writes a default config to `~/Library/Application Support/speakinput/config.toml`.

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
speakinput -m small.en           # override the model for this run
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
| `-m`  | `--model NAME`        | Override the whisper model (`tiny.en`/`base.en`/`small.en`) |
| `-l`  | `--list-devices`      | List available input devices and exit           |
| `-D`  | `--diagnose`          | Record 2s and print audio stats                 |
| `-n`  | `--no-inject`         | Print transcript to stderr instead of typing it |
| `-d`  | `--debug`             | Log every key event and transcript to stderr    |
| `-t`  | `--trailing-space`    | Append a space after each transcript (default)  |
| `-T`  | `--no-trailing-space` | Don't append a space after each transcript      |
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
model = "base.en"    # tiny.en | base.en | small.en
```

| Model    | Size  | Speed (M1) | Accuracy |
| -------- | ----- | ---------- | -------- |
| tiny.en  | 75 MB | ~real-time | OK for short phrases |
| base.en  | 142 MB| ~real-time | Good default |
| small.en | 466 MB| ~2x slower | Best for noisy mics |

The first time you select a model, pywhispercpp downloads it to `~/.cache/pywhispercpp/`.

## Configuration reference

```toml
[stt]
model = "base.en"          # whisper.cpp model
language = "en"            # language code
beam_size = 1              # 1 = greedy (fastest); up to 10 for higher accuracy

[audio]
device = null              # null = system default mic; or an integer index from --list-devices
sample_rate = 16000        # whisper expects 16 kHz; do not change

[hotkey]
key = "alt_r"              # see valid keys above

[inject]
restore_clipboard_ms = 50  # how long to wait before restoring the prior clipboard contents
                           # (only relevant when injecting Unicode text)
trailing_space = true      # append a single space after each transcript
                           # (set false for code editors where you don't want auto-spacing)
```

## How it works

```
hotkey press   →  AudioRecorder starts capturing
hotkey release →  AudioRecorder stops
              →  WhisperCppTranscriber transcribes the buffer
              →  TypingInjector types the result (or pastes via clipboard for non-ASCII)
```

Three interfaces — `Recorder`, `Transcriber`, `Injector` — are stable seams. v2 will add:

- A `StreamingTranscriber` that consumes `AudioRecorder.chunk_generator()` for partial results while the key is still held (the "overlapped streaming" goal from the original design).
- A `CommandInjector` that interprets the transcription and dispatches to shell or an agent.
- A multilingual model path for Chinese (the config field exists; the UI is intentionally not wired up yet).

## Troubleshooting

**Key detection works but transcript never appears in the focused app.** This is the Accessibility permission — see the [macOS permissions](#macos-permissions) section. Run with `--debug` to confirm: the `[debug] key press end` and `[debug] transcript: '...'` lines will appear, but the receiving app stays empty. Add the venv python to **Privacy & Security → Accessibility**, log out/in, and try again.

**Nothing happens when I hold the hotkey.** Check both macOS permissions above. Run with `-v` to see debug logs.

**The right-Option hotkey triggers menu mnemonics.** v1 uses `suppress=False` so the key reaches other apps — useful for Alt-Tab, but it can arm menu shortcuts in some apps. v2 will add a `suppress=True` mode for that case.

**`pywhispercpp` fails to load the model.** Make sure the model name matches exactly (case-sensitive, e.g. `base.en` not `Base.en`). Run `speakinput --diagnose` to surface the error directly.

**Transcription is empty / nonsense.** Try `small.en` for noisy environments. Make sure `--list-devices` is showing the right mic; set `[audio].device` to its index.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT
