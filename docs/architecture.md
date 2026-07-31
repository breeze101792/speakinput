# Speakinput Architecture

Push-to-talk voice transcription for macOS and Linux. Hold a hotkey, speak, release — text appears in the focused field.

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Main Thread                         │
│  cli.main() → App(config) → app.run()                   │
│  Blocks on _shutdown.wait()                             │
└──────────────┬──────────────────────────────────────────┘
               │ spawns
    ┌──────────┼──────────────────────────────┐
    ▼          ▼                              ▼
Event Worker   Listener Threads (1-2)     Liveness Watcher
Thread         pynput (CGEventTap)        Polls every 5s
Serializes     or evdev (read_loop)       Detects dead threads
all press/     One per hotkey key         Detects system sleep
release work                           heartbeat stale events
    │
    ├── on_hotkey_press()
    │     ├── recorder.start()
    │     │     └── PortAudio callback thread (audio chunks)
    │     ├── SilenceWatchdog.start()
    │     │     └── Watchdog thread (20Hz RMS polling)
    │     └── media_controller.pause()
    │
    └── on_hotkey_release() / _finalize()
          ├── recorder.drain() / close()
          │     └── Audio teardown thread (bounded 1.5s)
          ├── _process_and_inject()
          │     ├── trim_trailing_silence()
          │     ├── transcriber.transcribe()  [CPU/GPU bound]
          │     └── injector.inject()
          │           └── Clipboard restore timer (daemon)
          └── media_controller.resume()
```

## Module Map

| Module | Role | Key Class |
|--------|------|-----------|
| `app.py` | Orchestrator — owns all components, press state machine | `App` |
| `audio.py` | PortAudio capture — start/stop/drain/close | `AudioRecorder` |
| `hotkey.py` | Global key detection — pynput or evdev backend | `HotkeyListener` / `EvdevHotkeyListener` |
| `silence.py` | Trailing silence trim + auto-stop watchdog | `SilenceWatchdog` |
| `config.py` | Frozen dataclass config, TOML loading | `Config` |
| `transcriber.py` | whisper.cpp wrapper with GPU auto-detect | `WhisperCppTranscriber` |
| `injector.py` | Text injection — pynput / wtype / ydotool | `TypingInjector` |
| `feedback.py` | Menu bar status indicator — rumps or stderr | `StderrFeedback` / `RumpsFeedback` |
| `media.py` | Pause/resume media on press/release | `MediaController` |
| `_mac_tap_heal.py` | Monkey-patch pynput to re-enable disabled CGEventTap | — |
| `models.py` | Download/load whisper model files | — |
| `singleinstance.py` | Process-level flock guard | — |

## Threading Model

All hotkey callbacks are **O(microseconds)** — they just enqueue to a `queue.Queue`. The event worker thread consumes the queue and runs the heavy work (record, transcribe, inject) serialized. This prevents macOS from disabling slow CGEventTaps.

**Threads:**

| Thread | Purpose | Lifetime |
|--------|---------|----------|
| Main | Blocks on `_shutdown.wait()` | Process |
| Event worker | Serializes all press/release bodies | Process |
| Listener (per key) | pynput/evdev key detection | Process |
| PortAudio callback | Delivers audio chunks (~30ms) | While recording |
| Liveness watcher | Polls listener health every 5s | Process |
| Silence watchdog | Polls RMS at 20Hz, fires on silence | Per press |
| Audio teardown | Bounded stop/close of PortAudio | Per close (1.5s max) |
| Media resume | Bounded media resume during shutdown | Shutdown only |
| Heartbeat | Prints "still alive" every 60s | Process (debug) |

## Lock Discipline

| Lock | Scope | Purpose |
|------|-------|---------|
| `_busy` | App | Only one active press at a time |
| `_body_lock` | App | Serialize chunked drain→re-arm window |
| `_inject_lock` | App | Serialize `injector.inject()` across threads |
| `_prompt_lock` | App | Guard continuity state (across-press hints) |
| `_stream_lock` | AudioRecorder | Serialize open/stop/close (CoreAudio HAL) |
| `_rms_lock` | AudioRecorder | Guard `_last_rms` between callback and watchdog |
| `_OPENCC_LOCK` | App (global) | Guard lazy OpenCC construction |

**Rule:** The PortAudio callback (`_on_audio`) never takes `_stream_lock` — it must never block.

## Press Lifecycle

```
1. PRESS: listener callback → enqueue on_hotkey_press
2. EVENT WORKER:
   a. Guard: _busy.locked() → ignore (re-entry)
   b. _busy.acquire()
   c. recorder.start() → PortAudio stream opens
   d. _arm_watchdog() → SilenceWatchdog starts (if auto-stop)
   e. media_controller.pause()
   f. feedback → "listening"

3. RECORDING (ongoing):
   PortAudio callback → chunks accumulate, RMS + timestamp updated
   Watchdog polls RMS at 20Hz:
     - If RMS < threshold for auto_stop_seconds → _on_watchdog_chunk()
       → drain → transcribe → inject → re-arm watchdog
     - If stream_healthy() == False for 4s → fire (dead stream)
   User can hold key through multiple auto-stop chunks

4. RELEASE: listener callback → enqueue on_hotkey_release
5. EVENT WORKER:
   a. _manual_release_pending = True (signals watchdog to bail)
   b. Stop watchdog
   c. media_controller.resume()
   d. feedback → "processing"
   e. _finalize():
      - drain() → get remaining audio
      - close() → tear down stream
      - _process_and_inject() → transcribe final chunk
      - _busy.release()
      - feedback → "idle"
```

## Layered Prompt Construction

The whisper `initial_prompt` is assembled from three sources (capped at 400 chars total):

1. **Static vocabulary bias** — configured per profile (e.g. "embedded software engineer")
2. **Across-press hint** — last transcribed text from the previous key press
3. **Within-press chunk** — last transcribed text from the current press (multi-sentence sessions)

## Model Sharing

Two profiles using the same model path share a single `WhisperCppTranscriber` instance (one load, ~466 MB resident). Dedup happens in `_build_transcribers()`.

## Platform Backends

| Component | macOS | Linux |
|-----------|-------|-------|
| Hotkey | pynput (CGEventTap) | evdev (preferred) → pynput |
| Injection | pynput (Cmd+V) | wtype → ydotool → pynput |
| Media | osascript (Spotify) | playerctl (MPRIS) |
| Menu bar | rumps (NSStatusBar) | stderr fallback |
| Audio | PortAudio/CoreAudio | PortAudio/ALSA |
