# Speak Input — Architecture & Recovery Design

This document describes the internal architecture, the error-recovery
system, and the platform differences between macOS and Linux. It is
the definitive reference for "what happens when X fails" and "how the
program heals itself without a restart".

## Table of Contents

1. [Component Overview](#component-overview)
2. [Request Lifecycle](#request-lifecycle)
3. [Error Recovery Matrix](#error-recovery-matrix)
4. [Pluggable STT Engines](#pluggable-stt-engines)
5. [Microphone Failover](#microphone-failover)
6. [Injector Fallback Chain](#injector-fallback-chain)
7. [Listener Self-Healing](#listener-self-healing)
8. [Platform Differences](#platform-differences)
9. [Test Organization](#test-organization)
10. [Configuration Reference](#configuration-reference)

---

## Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                        App (app.py)                      │
│  Orchestrates the lifecycle. Owns the busy lock, the    │
│  event worker queue, the liveness watcher, and the      │
│  recovery config.                                       │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Hotkey   │  │ Recorder │  │Transcrib.│  │Injector │ │
│  │Listener  │  │(audio.py)│  │(transcr.) │  │(inject.)│ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ │
│       │              │              │              │      │
│  ┌────▼─────┐  ┌─────▼──────┐  ┌───▼──────┐  ┌───▼────┐ │
│  │pynput    │  │PortAudio   │  │Engine    │  │wtype   │ │
│  │  OR      │  │(sounddevice│  │Registry  │  │ydotool │ │
│  │evdev     │  │)           │  │          │  │pynput  │ │
│  └──────────┘  └────────────┘  └──────────┘  └────────┘ │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │Liveness      │  │Silence       │  │Media          │  │
│  │Watcher       │  │Watchdog      │  │Controller     │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Core interfaces (stable seams)

| Interface    | Protocol          | Implementations                          |
|--------------|-------------------|------------------------------------------|
| `Recorder`   | `audio.Recorder`  | `AudioRecorder` (PortAudio/sounddevice)  |
| `Transcriber`| `transcriber.Transcriber` | `WhisperCppTranscriber`, `FasterWhisperTranscriber`, `AppleSpeechTranscriber` |
| `Injector`   | `injector.Injector` | `TypingInjector`, `WtypeInjector`, `YdotoolInjector`, `FallbackInjector` |
| `Feedback`   | `feedback.Feedback` | `NullFeedback`, `StderrFeedback`, `RumpsFeedback` |
| `HotkeyListener` | `hotkey.HotkeyListenerProtocol` | `HotkeyListener` (pynput), `EvdevHotkeyListener` |

Each interface is a `typing.Protocol` so any implementation with the
right method signatures can be substituted — no inheritance required.

---

## Request Lifecycle

A single push-to-talk cycle:

```
1. User holds hotkey
   → HotkeyListener callback (pynput CGEventTap / evdev read_loop)
   → _enqueue_event(on_hotkey_press, profile)  [returns immediately]
   → Event worker thread picks up the job
   → on_hotkey_press:
      a. Acquire _busy lock
      b. recorder.start()  [opens PortAudio stream]
      c. If auto_stop_seconds > 0: arm SilenceWatchdog
      d. MediaController.pause()
      e. feedback.set_state("listening")

2. User speaks (PortAudio callback feeds chunks to _chunks list)

3. User releases hotkey (OR watchdog fires on silence)
   → _enqueue_event(on_hotkey_release, profile)
   → on_hotkey_release / _on_watchdog_chunk:
      a. recorder.drain()  [returns accumulated audio]
      b. trim_trailing_silence()
      c. Silence gate: if RMS < threshold, skip
      d. _transcribe_with_recovery():
         - transcriber.transcribe(audio)
         - If raises: reload model, retry up to N times
      e. zh_conversion (if Chinese text)
      f. injector.inject(text)  [with _inject_lock]
      g. feedback.set_state("idle")
      h. Release _busy lock

4. If auto-stop fired mid-press:
   → Re-arm watchdog for next sentence
   → User can keep holding for multi-sentence dictation
```

---

## Error Recovery Matrix

Every failure domain has a recovery path. The `[recovery]` config
section controls each independently.

| Failure                     | Recovery                       | Config flag                        |
|-----------------------------|--------------------------------|------------------------------------|
| **Mic unplugged mid-session** | Re-enumerate devices, pick next available | `mic_failover_scan` |
| **Mic open fails**          | Retry with fallback device     | `mic_open_retries`                |
| **PortAudio stream wedges** | Bounded close (1.5s timeout), abandon | `_CLOSE_JOIN_TIMEOUT_S`    |
| **Transcriber crashes**     | Reload model, retry transcribe | `transcribe_retries`, `transcribe_backoff_s` |
| **Engine package missing**  | Fall back to whispercpp        | `transcribe.engine`               |
| **Injector fails at runtime** | FallbackInjector tries next backend | `injector_fallback`          |
| **Hotkey listener dies**    | Liveness watcher restarts it   | `listener_restart_min_interval_s` |
| **macOS CGEventTap disabled** | Self-heal monkey-patch re-enables tap | automatic              |
| **System sleep**            | Clock-skew detection, restart all listeners | automatic          |
| **Stuck press (release lost)** | `_abort_press` discards buffer, releases lock | automatic      |
| **Media resume hangs**      | Bounded thread join (3s)       | automatic                          |
| **Clipboard write fails**   | Skip Unicode inject, warn user | automatic                          |
| **Subprocess hangs**        | 5s timeout on all shell-outs   | `_SUBPROCESS_TIMEOUT_S`           |
| **Event worker handler raises** | Log and continue (worker never dies) | automatic                |
| **Single-instance lock held** | Exit(3) with clear message     | automatic                          |
| **Config validation fails** | Exit(1) with field name        | automatic                          |
| **Model download fails**    | Exit(2) before listener starts | automatic                          |

### The philosophy: "never require a restart"

The user's words from the design conversation:

> every error could be recovered by this program itself. And I don't
> need to trigger everything in order to make it work. Sometimes it
> just fails, I need to restart it again.

Every recovery path is automatic. The user does not need to press a
button, edit a config, or restart the app. If a component fails, the
app either heals itself (reload, retry, fallback) or degrades
gracefully (warn + continue with the next available backend).

The `[recovery]` config section lets the user tune or disable any
recovery path. Setting a flag to `0` or `false` restores the old
"fail and let the user restart" behavior — useful for debugging.

---

## Pluggable STT Engines

Three engines are available via `[transcribe].engine`:

| Engine           | Package                    | OS support              | Model setup         |
|------------------|----------------------------|-------------------------|---------------------|
| `whispercpp`     | `pywhispercpp`             | All (macOS, Linux, Win) | Downloads from HF on first use |
| `faster_whisper` | `faster-whisper`           | Linux (NVIDIA), macOS   | Downloads from HF on first use |
| `apple`          | `pyobjc-framework-Speech`  | macOS 13+ only          | No model download — uses OS built-in |

### Engine selection and fallback

```
config.engine = "faster_whisper"
  → create_transcriber("faster_whisper", ...)
  → try FasterWhisperTranscriber(...)
  → if ImportError or TranscriberError:
      warn("faster_whisper unavailable; falling back to whispercpp")
      → WhisperCppTranscriber(...)
```

The app **never** fails to start because an optional engine isn't
installed. It always falls back to `whispercpp` (the always-available
default).

### OS-aware model setup

- **whispercpp**: downloads `.bin` model files from Hugging Face via
  `pywhispercpp.utils.download_model()`. Cached in
  `~/.cache/pywhispercpp/models/`. Only downloads if the file is
  missing — subsequent runs reuse the cache.
- **faster_whisper**: downloads from Hugging Face via CTranslate2's
  model loader. Cached in `~/.cache/huggingface/`. Same models as
  `openai/whisper` (converted to CTranslate2 format).
- **apple**: no model download. Uses `SFSpeechRecognizer` which is
  built into macOS. Requires dictation enabled in System Settings.

The `ensure_model()` function in `models.py` handles the download for
whispercpp and faster_whisper. For apple, it's a no-op.

---

## Microphone Failover

When the configured audio device disappears (USB unplug, Bluetooth
disconnect, device index change), the recorder automatically falls
back:

```
recorder.start()
  → device = self.device  (e.g. 2)
  → _device_is_present(2)?  → False (USB mic unplugged)
  → if mic_failover_scan:
      _find_fallback_device()  → scans all input devices
      → picks the one with most input channels
      → device = scanned_index
  → else:
      device = None  (system default)
  → retry opening stream up to mic_open_retries times
  → if all retries fail: raise AudioError (user sees 'error' state)
```

The key insight: PortAudio re-resolves `device=None` on every
`InputStream()` call, so a newly-plugged mic or a Sound control panel
switch is picked up automatically. The failover scan is only needed
when the user has pinned a specific device index in config.toml.

---

## Injector Fallback Chain

On Linux Wayland, the output side has three backends. The
`FallbackInjector` wraps them in a chain:

```
inject("text")
  → try WtypeInjector.inject(text)
  → if raises: warn, advance to YdotoolInjector
  → try YdotoolInjector.inject(text)
  → if raises: warn, advance to TypingInjector (pynput)
  → try TypingInjector.inject(text)
  → if raises: propagate (last resort)
```

The active index advances permanently — a failed backend is
considered dead for the rest of the session. This avoids repeated
timeouts on a wedged backend.

On macOS and Windows, there's only one backend (pynput), so the
chain is a single element and `FallbackInjector` is transparent.

---

## Listener Self-Healing

Three layers of defense keep the hotkey alive:

### Layer 1: CGEventTap self-heal (macOS only)

`_mac_tap_heal.py` monkey-patches pynput's `ListenerMixin._run` to
check `CGEventTapIsEnabled()` on every run-loop iteration. If macOS
disabled the tap (sleep/wake, focus churn, permission transition),
the patched loop calls `CGEventTapEnable(tap, True)` to re-enable it.
Detection latency: ~1s (the run loop polls with a 1s timeout).

### Layer 2: Liveness watcher

A background thread (`_LivenessWatcher`) polls every 5s:
- `listener.is_running()` — is the thread alive?
- `listener.last_event_at()` — has any event arrived in the last 300s?

If the thread is dead OR no events have arrived for 300s (the tap is
wedged in a way the self-heal can't fix), `_on_listener_dead(key)` is
called:
1. Try to restart the listener (`_restart_listener`).
2. If the restart succeeds: log `[info] hotkey listener restarted`.
3. If the listener died again within `listener_restart_min_interval_s`
   (default 60s): warn the user instead of flapping.
4. Abort any stranded press (the release event died with the listener).

### Layer 3: Sleep detection

The watcher also detects system sleep via wall-clock vs monotonic-clock
skew. When the skew exceeds 5s, `_on_system_sleep()` restarts **every**
listener (macOS disables all CGEventTaps across sleep) and aborts any
active press.

---

## Platform Differences

### Hotkey detection

| Platform | Primary backend | Fallback          |
|----------|-----------------|-------------------|
| macOS    | pynput (CGEventTap + HIToolbox) | — |
| Linux    | evdev (reads `/dev/input/event*`) | pynput (X11 only) |
| Windows  | pynput (SendInput) | — |

On Linux, evdev is preferred because it works on **both** Wayland and
X11 (it reads the kernel input subsystem directly, no display server
needed). pynput's X11 backend can't reach a Wayland session without
XWayland.

### Text injection

| Platform | Primary | Fallback 1 | Fallback 2 |
|----------|---------|------------|------------|
| macOS    | pynput (HIToolbox) | — | — |
| Linux+X11| pynput (XTest) | — | — |
| Linux+Wayland | wtype (wlroots) | ydotool (uinput) | pynput (XWayland only) |
| Windows  | pynput (SendInput) | — | — |

### Media control

| Platform | Backend |
|----------|---------|
| macOS    | osascript (Spotify, Music.app) |
| Linux    | playerctl (MPRIS D-Bus) |
| Windows  | PowerShell (SMTC) |

### Default hotkeys

| Platform | Primary key | Secondary key |
|----------|-------------|---------------|
| macOS    | `alt_r` (Right Option) | `cmd_r` (Right Command) |
| Linux    | `ctrl_r` (Right Ctrl)  | `cmd_r` (Right Super) |
| Windows  | `ctrl_r` (Right Ctrl)  | `cmd_r` (Right Super) |

### Permissions

| Platform | What's needed |
|----------|---------------|
| macOS    | Accessibility + Input Monitoring (Privacy & Security) |
| Linux    | `input` group membership (for evdev `/dev/input` access) |
| Windows  | (no special permissions for pynput) |

---

## Test Organization

Tests are split by concern and platform:

| File               | What it tests                          | Platform |
|--------------------|----------------------------------------|----------|
| `test_app.py`      | App orchestrator, press/release, recovery | Cross-platform (mocked I/O) |
| `test_audio.py`    | AudioRecorder, PortAudio, mic failover | Cross-platform (mocked sounddevice) |
| `test_hotkey.py`   | Hotkey listeners, evdev, CGEventTap    | Cross-platform + macOS-only tests (skipped on Linux) |
| `test_injector.py` | TypingInjector, WtypeInjector, etc.    | Cross-platform (mocked subprocess) |
| `test_transcriber.py` | WhisperCppTranscriber, engine probe  | Cross-platform (mocked pywhispercpp) |
| `test_recovery.py` | Recovery config, crash recovery, fallback chains | Cross-platform (mocked I/O) |
| `test_config.py`   | Config loading, validation, overrides  | Cross-platform |
| `test_models.py`   | Model bootstrap, download, upgrade     | Cross-platform (mocked pywhispercpp) |
| `test_silence.py`  | Silence trim, auto-stop watchdog       | Cross-platform |
| `test_singleinstance.py` | Single-instance lockfile         | Cross-platform |
| `test_cli.py`      | CLI flags, -C/--edit-config            | Cross-platform |

### Platform-specific tests

macOS-only tests (CGEventTap self-heal) use:
```python
if sys.platform != "darwin":
    pytest.skip("CGEventTap self-heal is macOS-only")
```

Linux-specific tests (evdev) use the `fake_evdev_keyboard` fixture
which requires the `evdev` package (installed only on Linux). On
non-Linux platforms, these tests are skipped via the import guard in
`hotkey.py`.

### Running tests

```bash
pytest                    # all tests (3 macOS-only skipped on Linux)
pytest tests/test_recovery.py  # only recovery tests
pytest -k "transcriber"   # only transcriber-related tests
ruff check .              # lint
```

---

## Configuration Reference

### [recovery] section

```toml
[recovery]
transcribe_retries = 2              # 0 = no retry
transcribe_backoff_s = 0.5         # 0 = retry immediately
mic_failover_scan = true            # false = system-default fallback only
mic_open_retries = 2               # 0 = fail on first attempt
injector_fallback = true            # false = no backend fallback
listener_restart_min_interval_s = 60  # 0 = always restart (may flap)
```

### [transcribe] section

```toml
[transcribe]
engine = "whispercpp"   # whispercpp | faster_whisper | apple
use_gpu = "auto"         # auto | true | false
gpu_device = 0
n_threads = 0            # 0 = auto
```

See `config.example.toml` for the full reference with inline comments.