# Error Recovery

Speakinput has a layered recovery system that keeps the app running through macOS environment changes without user intervention.

## Recovery Layers

### 1. Listener Death (thread crash / event tap disabled)

**Detection:** `_LivenessWatcher` polls `is_running()` every 5s. Also checks `last_event_at()` against a 300s stale threshold (heartbeat backstop).

**Recovery:**
- `_on_listener_dead(key)` calls `_restart_listener(key)` — creates a fresh listener in place
- Flapping guard: if the listener died within 60s of the last restart, warn instead of restarting
- Any active press is aborted (`_abort_press()`) since the release event was lost

### 2. macOS CGEventTap Disable (sleep/wake, focus, permissions)

Three layers of defense:

| Layer | What | When |
|-------|------|------|
| `_mac_tap_heal.py` | Monkey-patches pynput to call `CGEventTapIsEnabled()` + `CGEventTapEnable()` every 1s | Every CFRunLoop timeout |
| Heartbeat backstop | `_LivenessWatcher` detects "thread alive but no events for 300s" | Background poll |
| Sleep restart | `_on_system_sleep()` restarts all listeners | Clock skew > 5s |

### 3. System Sleep/Wake

**Detection:** `_LivenessWatcher._check_sleep()` computes wall-clock vs monotonic-clock skew. Threshold: 5s.

**Recovery (`_on_system_sleep`):**
1. Restart all hotkey listeners
2. Close the audio recorder (so next `start()` opens fresh stream)
3. Abort any orphaned press

### 4. Dead Audio Stream (post-sleep PortAudio)

**Detection:** `AudioRecorder.stream_healthy(timeout_s=3.0)` checks if `_last_callback_at` is recent. After sleep/wake, CoreAudio HAL can silently drop the callback link — stream handle stays open, `is_recording()` returns True, but no audio arrives.

**Recovery:**
- `SilenceWatchdog` polls `stream_healthy()` each tick
- If unhealthy for >= 4s (`HEALTHY_TIMEOUT_S`), fires the trigger to finalize the press
- Recovers if the stream comes back healthy (resets timer)

### 5. Audio Device Gone / Permission Denied

**Detection:** `AudioRecorder.start()` catches exceptions from `sd.InputStream()` or `stream.start()`.

**Recovery:**
- Prints actionable message (check mic, check Microphone permission)
- Raises `AudioError` (user-facing, logged at WARNING not EXCEPTION)
- `App.on_hotkey_press()` catches it, sets feedback to "error", releases busy lock
- Next press retries normally

### 6. CoreAudio HAL Deadlock (stream close)

**Mitigation:** `AudioRecorder.close()` runs `stream.stop()` + `stream.close()` on a helper thread with 1.5s bounded join. If it hangs:
- The thread is abandoned (daemon, dies with process)
- `_stream` is nulled so next `start()` opens fresh
- Logged at WARNING

### 7. Subprocess Hang (pbcopy, wtype, osascript, playerctl)

All subprocess calls in `injector.py` and `media.py` are bounded by `_SUBPROCESS_TIMEOUT_S = 5.0`. Media resume during shutdown runs on a daemon thread with 3s bounded join.

### 8. Ctrl-C / Shutdown

| Signal | Action |
|--------|--------|
| First Ctrl-C | Sets `_shutdown` event, runs `shutdown()` teardown |
| Second Ctrl-C | Dumps all thread stacks, force-exits via `os._exit(2)` |
| SIGTERM/SIGHUP | Sets `_shutdown` (single-shot) |

### 9. Transcription / Injection Failure

`App._process_and_inject()` wraps `transcriber.transcribe()` and `injector.inject()` in try/except. Failures are logged but don't crash the app. The event worker thread catches all exceptions from event handlers — it MUST NOT die.

## Shutdown Order

```
1. Stop liveness watcher
2. Stop heartbeat (debug)
3. Stop media controller
4. Stop active watchdog
5. Stop all listeners
6. Stop event worker (drain queue)
7. Close recorder
8. Stop feedback
```

## Test Coverage

| Recovery Mechanism | Test File |
|-------------------|-----------|
| Stream health detection | `test_audio.py::test_stream_healthy_*` (6 tests) |
| Watchdog dead-stream detection | `test_silence.py::test_watchdog_*dead*` (4 tests) |
| Audio teardown timeout | `test_audio.py::test_recorder_close_does_not_block_when_portaudio_wedges` |
| Listener death/restart | `test_app.py` (liveness watcher tests) |
| Press abort | `test_app.py` (abort press tests) |
| Error feedback state | `test_audio.py::test_recorder_surfaces_stream_error_on_press` |
