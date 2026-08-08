"""Audio capture from the default input device."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# Upper bound for the PortAudio stop()+close() teardown. On macOS the
# C call can wedge indefinitely (CoreAudio HAL mutex contention with
# the audio IO thread, often after a sleep/wake or a USB/Bluetooth
# mic glitch). If we let it pin `_stream_lock` the main thread's
# shutdown-time `recorder.close()` deadlocks waiting for the same
# lock, and even SIGINT can't escape (the audio IO thread is blocked
# on a GIL the C extension never releases). 1.5s is long enough that
# a healthy teardown always finishes — `stream.stop()` on a quiet
# 16kHz mono stream is sub-10ms in practice — but short enough that
# a wedged C call costs us at most 1.5s of shutdown latency, not
# "force-quit the process".
_CLOSE_JOIN_TIMEOUT_S = 1.5

try:
    import sounddevice as sd
except ImportError as _exc:  # pragma: no cover - exercised only when missing
    sd = None  # type: ignore[assignment]
    _SD_IMPORT_ERROR = _exc
else:
    _SD_IMPORT_ERROR = None


class AudioError(RuntimeError):
    """Raised when audio capture cannot start.

    The message is meant for end users: it explains what went wrong in
    plain language and suggests a fix. `audio.py` prints a longer version
    of the same message to stderr with the original PortAudio code
    attached for debugging. Callers should log at WARNING (not
    EXCEPTION) — this is almost always a user-fixable environment
    problem (no mic, no permission, bad sample rate), not a bug.
    """


# Friendly explanations for the PortAudio error codes the user is most
# likely to hit. The codes are the `PaErrorCode` enum values from
# `pa/src/common/pa_errors.c` in the PortAudio source. Anything not in
# this map falls through to the generic "audio device error" hint,
# which covers the two cases macOS users actually see in the wild.
_PORTAUDIO_REASONS: dict[int, str] = {
    -10000: "operation timed out opening the audio device",  # paTimedOut
    -9999: "PortAudio not initialized",  # paNotInitialized
    -9998: "invalid audio device index",  # paInvalidDevice
    -9997: "device in use by another program",  # paDeviceBusy (was the most
    # useful real-world mapping even though the official name is
    # paInsufficientMemory — the user sees this when another app has
    # the mic exclusively open)
    -9996: "operation aborted (audio device was unplugged?)",  # paOperationAborted
    -9995: "audio host API reports compatibility error",  # paCompatibilityError
    -9994: "device busy (held exclusively by another program)",  # paDeviceBusy
    -9993: "Host API not initialized",  # paHostApiNotInitialized
    -9986: "internal audio engine error (CoreAudio/AUHAL rejected the stream)",  # paInternalError
    -9985: "device disconnected or unavailable",  # paDeviceUnavailable
}


def _describe_audio_error(exc: BaseException) -> str:
    """Return a one-line, human-friendly description of a PortAudio failure.

    Pulls the numeric error code out of `sounddevice.PortAudioError`
    (whose `args[1]` is the int code when present) and looks it up in
    `_PORTAUDIO_REASONS`. Falls back to the raw message for anything we
    don't recognize. The original `str(exc)` is intentionally NOT used
    as the final answer because it just says "Internal PortAudio
    error [PaErrorCode -9986]" which is meaningless to users.
    """
    code: int | None = None
    # sounddevice stores the int code in args[1] when raised via
    # PortAudioError(errormsg, err). Older versions only had the
    # message. Both shapes are handled.
    args = getattr(exc, "args", ())
    if len(args) >= 2 and isinstance(args[1], int):
        code = args[1]
    elif len(args) == 1 and isinstance(args[0], int):
        code = args[0]
    if code is not None and code in _PORTAUDIO_REASONS:
        return f"{_PORTAUDIO_REASONS[code]} (code {code})"
    if code is not None:
        return f"audio device error (code {code})"
    return "audio device error"


class Recorder(Protocol):
    def start(self) -> None: ...
    def stop(self) -> np.ndarray: ...
    def drain(self) -> np.ndarray: ...
    def close(self) -> None: ...
    def is_recording(self) -> bool: ...
    def current_rms(self) -> float: ...
    def stream_healthy(self, timeout_s: float = ...) -> bool: ...


@dataclass
class AudioRecorder:
    """Records mono float32 audio at `sample_rate` Hz from `device`.

    Uses a non-blocking queue fed by the PortAudio callback so the audio
    thread never waits on application code. `stop()` flushes the stream
    and returns the concatenated buffer.
    """

    sample_rate: int = 16000
    device: int | None = None
    channels: int = 1
    _stream: object | None = None
    _chunks: list[np.ndarray] | None = None
    _recording: bool = False
    # When True, a disappeared configured device triggers a scan of all
    # available input devices (not just system default). Set by the App
    # from [recovery].mic_failover_scan. When False, only falls back to
    # system default (the old behavior).
    mic_failover_scan: bool = True
    # How many times to retry opening a stream with the fallback device
    # before raising. Set by the App from [recovery].mic_open_retries.
    mic_open_retries: int = 2
    # Serializes stream open/stop/close. PortAudio/CoreAudio does NOT
    # tolerate two threads stopping the same stream concurrently: both
    # end up inside AudioOutputUnitStop contending on the HAL mutex
    # while the CoreAudio IO thread waits for the GIL — a three-way
    # deadlock observed in the wild (main thread in atexit's
    # Pa_Terminate vs. the hotkey thread in recorder.close()). Holding
    # this lock across the whole open/stop/close guarantees at most one
    # thread is inside CoreAudio per recorder; the second caller either
    # waits or no-ops on `_stream is None`.
    # The audio callback (`_on_audio`) never takes this lock — it must
    # never block.
    _stream_lock: threading.Lock = field(default_factory=threading.Lock)
    # The most recent chunk's RMS, updated from the audio callback and
    # read by the auto-stop watchdog. Guarded by `_rms_lock` because the
    # callback and the watchdog run on different threads.
    _last_rms: float = 0.0
    _rms_lock: threading.Lock = field(default_factory=threading.Lock)
    # Monotonic timestamp of the most recent audio callback. Updated
    # unconditionally in `_on_audio()` so `stream_healthy()` can detect
    # a PortAudio stream that is "open" but stopped delivering audio
    # (e.g. after macOS sleep/wake when the CoreAudio HAL reinitializes
    # but the existing stream handle goes silent). Shared read by
    # `stream_healthy()` on the watchdog thread — only one writer
    # (the callback) so no lock needed.
    _last_callback_at: float = 0.0

    def _require_sounddevice(self) -> None:
        if sd is None:
            raise AudioError(
                f"sounddevice is not installed: {_SD_IMPORT_ERROR}. "
                "Install with `pip install sounddevice`."
            )

    def is_recording(self) -> bool:
        return self._recording

    def current_rms(self) -> float:
        """Return the RMS of the most recently received audio chunk.

        Returns 0.0 when no audio has arrived yet. The value updates
        asynchronously as the PortAudio callback delivers chunks, so
        callers should sample it on a polling loop (the auto-stop
        watchdog does this at ~20 Hz).
        """
        with self._rms_lock:
            return self._last_rms

    def stream_healthy(self, timeout_s: float = 3.0) -> bool:
        """Return True if the stream is recording AND delivering audio.

        Detects the common post-sleep/wake failure where PortAudio's
        `InputStream` is open (`_recording` is True) but the CoreAudio
        HAL has stopped delivering callbacks — the stream is "dead" and
        no audio will ever arrive. After macOS sleep/wake, the HAL can
        silently drop the callback link without raising an error or
        closing the stream handle.

        `timeout_s` is the maximum seconds since the last callback
        before the stream is considered unhealthy. The default 3.0s
        is long enough to not false-positive during quiet speech (the
        callback fires every ~30ms at 16kHz/512 frames) but short
        enough to detect a dead stream within a few seconds.

        Returns False if the recorder was never started or has been
        closed (`_last_callback_at` remains 0.0 in both cases).
        """
        if not self._recording:
            return False
        if self._last_callback_at == 0.0:
            # Stream was just opened — give it a grace period.
            return True
        return (time.monotonic() - self._last_callback_at) < timeout_s

    def _device_is_present(self, device: int | None) -> bool:
        """Return True if the configured device can be opened.

        A `query_devices()` call is sub-millisecond — it just reads
        PortAudio's cached device table. We use it on the press path
        to detect the 'user unplugged their USB mic mid-session' case
        before we try to open the stream (which would raise a less
        helpful exception).

        For `device=None` (system default), the answer is always True:
        PortAudio re-resolves the default on every `InputStream()` call,
        so a newly-plugged mic or a macOS Sound control panel switch
        is picked up automatically. We don't try to second-guess it.
        """
        if device is None:
            return True
        try:
            sd.query_devices(device)
        except Exception:
            return False
        return True

    def _find_fallback_device(self) -> int | None:
        """Scan all input devices and return the first usable one.

        Used when the configured device disappeared and
        `mic_failover_scan` is True. Picks the device with the most
        input channels (usually a real microphone, not a virtual
        loopback). Returns None if no input device is found.
        """
        try:
            devices = sd.query_devices()
        except Exception:
            return None
        best_idx: int | None = None
        best_channels = 0
        for i, d in enumerate(devices):
            ch = int(d.get("max_input_channels", 0))
            if ch > best_channels:
                best_idx = i
                best_channels = ch
        return best_idx

    def start(self) -> None:
        if self._recording:
            return
        self._require_sounddevice()
        with self._stream_lock:
            self._chunks = []
            with self._rms_lock:
                self._last_rms = 0.0
            self._last_callback_at = 0.0
            # Fall back to system default if the pinned device disappeared
            # since the last press (USB unplug, Bluetooth headset
            # disconnected, etc.). The check is sub-ms; the stream-open
            # that follows is the real cost (~30-100ms on macOS). When
            # `device is None` the system default is used and re-resolved
            # by PortAudio on every call, so the fallback is unnecessary.
            device = self.device
            if device is not None and not self._device_is_present(device):
                fallback = None
                if self.mic_failover_scan:
                    scanned = self._find_fallback_device()
                    if scanned is not None:
                        print(
                            f"[warn] configured audio device {device} is not available; "
                            f"switching to device {scanned}",
                            file=sys.stderr,
                            flush=True,
                        )
                        fallback = scanned
                    else:
                        print(
                            f"[warn] configured audio device {device} is not available; "
                            f"no other input devices found; trying system default",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    print(
                        f"[warn] configured audio device {device} is not available; "
                        f"falling back to system default microphone",
                        file=sys.stderr,
                        flush=True,
                    )
                device = fallback
            # Try opening the stream, with retries if the first attempt
            # fails (e.g. the fallback device is also being unplugged
            # mid-open, or PortAudio needs a moment to settle after a
            # device change). Each retry scans for a fresh device.
            last_exc: Exception | None = None
            for attempt in range(max(self.mic_open_retries, 0) + 1):
                try:
                    self._stream = sd.InputStream(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype="float32",
                        device=device,
                        callback=self._on_audio,
                    )
                    self._stream.start()
                    self._recording = True
                    return
                except Exception as exc:
                    last_exc = exc
                    if attempt >= max(self.mic_open_retries, 0):
                        break
                    # The device we picked might also be gone. Re-scan.
                    if self.mic_failover_scan:
                        scanned = self._find_fallback_device()
                        if scanned is not None and scanned != device:
                            device = scanned
                    # Close any half-open stream from the failed attempt.
                    if self._stream is not None:
                        leaked = self._stream
                        self._stream = None
                        try:
                            leaked.close()
                        except Exception:
                            pass
            # All retries exhausted — close any half-open stream and
            # surface the error.
            if self._stream is not None:
                leaked = self._stream
                self._stream = None
                try:
                    leaked.close()
                except Exception:
                    pass
            reason = _describe_audio_error(last_exc) if last_exc else "unknown"
            print(
                f"[error] could not open audio input stream: {reason}. "
                f"Check that a microphone is connected and that speakinput "
                f"has Microphone permission in System Settings → Privacy "
                f"& Security → Microphone. If a USB/Bluetooth mic is "
                f"configured, try setting `device = null` in config.toml "
                f"to use the system default instead. (Original error: "
                f"{type(last_exc).__name__}: {last_exc})",
                file=sys.stderr,
                flush=True,
            )
            self._recording = False
            raise AudioError(
                f"audio stream open failed: {reason}"
            ) from last_exc

    def _on_audio(self, indata, frames, _pa_time, status) -> None:  # noqa: ANN001 (sounddevice API)
        # PortAudio's contract is that the callback MUST NOT raise:
        # any uncaught exception here takes the audio thread down and
        # either kills the process or stops further audio from ever
        # arriving. Wrap the body defensively and drop the offending
        # chunk. The rms update is best-effort; missing it for one
        # chunk only delays the auto-stop watchdog by 50ms.
        # status flags (overflow/underflow) are non-fatal; keep the
        # audio and let the caller surface the issue if needed.
        try:
            chunk = indata.copy().reshape(-1)
        except Exception:
            return
        # Record the callback timestamp unconditionally — this is the
        # heartbeat the `stream_healthy()` check relies on. Updated
        # before any early-return so even a failed RMS computation
        # still proves the HAL is alive. Use the `time` module
        # directly (the callback parameter `_pa_time` does not shadow
        # it) to avoid the name collision.
        self._last_callback_at = time.monotonic()
        if self._chunks is not None:
            self._chunks.append(chunk)
        # Track the most recent chunk's RMS for the auto-stop watchdog.
        # Computed once per callback so the watchdog's polling loop is a
        # cheap lock+read, not a full-buffer scan.
        try:
            if chunk.size:
                rms = float(np.sqrt(np.mean(chunk * chunk)))
            else:
                rms = 0.0
        except Exception:
            return
        with self._rms_lock:
            self._last_rms = rms

    def stop(self) -> np.ndarray:
        """Drain the buffer and close the PortAudio stream.

        Equivalent to `drain()` followed by `close()`. Returns whatever
        audio was recorded since the last `start()` or `drain()`.
        """
        audio = self.drain()
        self.close()
        return audio

    def drain(self) -> np.ndarray:
        """Return the recorded buffer since the last start/drain and
        clear the in-memory state. The PortAudio stream stays open so
        subsequent audio callbacks continue to accumulate into a fresh
        buffer.

        Used by the chunked auto-stop path: when silence triggers an
        auto-release mid-press, we drain the captured audio for
        transcription, then keep listening for the next sentence
        without paying the cost of tearing down and reopening the
        stream.

        If the recorder was never started (or has been closed), the
        chunks list is None and we return an empty buffer. This is
        the same shape as "recorded nothing" and lets callers
        (`_finalize`, `_on_watchdog_chunk`) treat the two cases
        identically.
        """
        chunks = self._chunks
        self._chunks = [] if self._stream is not None else None
        with self._rms_lock:
            self._last_rms = 0.0
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def close(self) -> None:
        """Stop and close the PortAudio stream. Idempotent.

        After `close()`, the recorder is no longer recording and must
        be `start()`-ed again to capture more audio. Any in-flight
        audio callbacks from PortAudio that arrive after `close()`
        are silently dropped (`_chunks` is set to `None` and the
        callback's guard skips the append).

        Serialized via `_stream_lock`: two threads must never be inside
        PortAudio's stop/close for the same stream at once (CoreAudio
        deadlocks on the HAL mutex otherwise).

        The actual `stream.stop()` + `stream.close()` run on a helper
        thread with a bounded join. On macOS the C call can wedge
        indefinitely (CoreAudio HAL mutex contention with the audio
        IO thread, sometimes triggered by a Bluetooth/USB mic glitch
        or a sleep/wake transition). If the helper doesn't return
        within the budget we abandon it, drop our reference, and
        let the process exit reclaim the native handle — `sounddevice`
        registers an atexit handler that would otherwise run inside
        `_Py_Finalize` (GIL held) while the IO thread is mid-callback,
        which is a known three-way deadlock. The `close()` call
        itself is what was hanging the process; bounding it is the
        whole point.
        """
        with self._stream_lock:
            stream = self._stream
            if stream is None:
                return
            # Detach the handle from the recorder BEFORE the bounded
            # teardown so any other thread that sees `_stream is None`
            # knows the recorder is closed. `_chunks` is cleared
            # outside the lock below, so the audio callback (which
            # only takes `_rms_lock`) is safe to run concurrently
            # with the teardown helper.
            self._stream = None
            self._recording = False

        def _teardown() -> None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        helper = threading.Thread(
            target=_teardown, name="speakinput-audio-teardown", daemon=True
        )
        helper.start()
        helper.join(timeout=_CLOSE_JOIN_TIMEOUT_S)
        # Drop any audio the callback accumulated after we stopped
        # the stream but before close() returned. The list object
        # itself is replaced (not cleared) so an in-flight callback
        # that already read `self._chunks` and is about to append
        # doesn't silently resurrect the dropped audio.
        self._chunks = None
        # Reset RMS so a stale chunk callback doesn't leak across close.
        with self._rms_lock:
            self._last_rms = 0.0
        if helper.is_alive():
            # The C-level stop/close is wedged. The thread is a
            # daemon, so it dies with the process; we've already
            # nulled `_stream` so subsequent `start()` opens a fresh
            # one. Logged at WARNING because it's a real, observed
            # symptom (not just a theoretical race) that the user
            # should be able to grep for in their log.
            log.warning(
                "PortAudio stream teardown did not finish within %.1fs; "
                "abandoning the close and continuing. This usually "
                "follows a sleep/wake or a USB/Bluetooth mic glitch.",
                _CLOSE_JOIN_TIMEOUT_S,
            )

    # --- v2 streaming seam: not consumed in v1, kept for the overlapped-stream upgrade.
    def chunk_generator(
        self, window_seconds: float = 1.0, hop_seconds: float = 0.5
    ) -> Iterator[np.ndarray]:
        """Yield overlapping windows of recorded audio while recording.

        v1 does not wire this up; the method exists so a future
        `StreamingTranscriber` can consume partial audio without changing the
        recorder's public surface. In v1 the returned generator yields nothing.
        """
        if window_seconds <= 0 or not 0 < hop_seconds <= window_seconds:
            raise ValueError("hop_seconds must be in (0, window_seconds]")
        return _empty_audio_stream()


def _empty_audio_stream() -> Iterator[np.ndarray]:
    """Generator stub for v1. v2 will replace with overlapping windows from
    the live recording buffer."""
    return
    yield np.zeros(0, dtype=np.float32)  # pragma: no cover - unreachable


def list_input_devices() -> list[dict]:
    """Return a list of input-capable devices, useful for the --list-devices CLI."""
    if sd is None:
        raise AudioError(f"sounddevice is not installed: {_SD_IMPORT_ERROR}")
    devices = sd.query_devices()
    return [
        {
            "index": i,
            "name": d["name"],
            "max_input_channels": int(d["max_input_channels"]),
            "default_samplerate": float(d["default_samplerate"]),
        }
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]
