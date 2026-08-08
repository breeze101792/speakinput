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


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear-interpolation resample from `from_rate` to `to_rate`.

    Used when the capture device doesn't support the target 16kHz
    natively — we open at its native rate (e.g. 44100) and resample
    here before returning the buffer to the transcriber. Linear
    interpolation is sufficient for speech (no need for a high-quality
    FIR filter — whisper is robust to minor interpolation artifacts).
    """
    if from_rate == to_rate or audio.size == 0:
        return audio
    ratio = from_rate / to_rate
    target_len = int(len(audio) / ratio)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


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
    # True after the system default (device=None) failed to open once.
    # Subsequent presses then scan for a real hardware device instead
    # of trying the broken default again.
    _default_blacklisted: bool = False
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
    # The actual sample rate the stream was opened at. May differ from
    # `sample_rate` (the target rate whisper needs) when the hardware
    # doesn't support 16kHz natively — we open at the device's native
    # rate and resample in drain()/stop() before returning the buffer.
    _capture_sample_rate: int = 16000
    # Devices that failed to open in a previous press. Persisted across
    # presses so we don't hammer the same broken ALSA device on every
    # key press. Cleared when a device successfully opens (the blacklist
    # is a per-session recovery state, not a permanent ban — a USB mic
    # that's re-plugged should work on the next press after the old
    # device index is gone from the system).
    _blacklisted_devices: set[int] = field(default_factory=set)
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

    def _find_fallback_device(self, exclude: set[int] | None = None) -> int | None:
        """Scan all input devices and return the best usable one.

        Used when the configured device disappeared and
        `mic_failover_scan` is True. Prefers real hardware devices
        (ALSA/HDA) over virtual bridges (pipewire, default) because
        virtual bridges can report many channels but still be broken.
        Among real devices, picks the one with the most input channels.

        Devices in `exclude` are skipped — this is how the
        blacklist prevents retrying broken devices across presses.

        Returns None if no usable input device is found.
        """
        try:
            devices = sd.query_devices()
        except Exception:
            return None
        excluded = exclude or set()
        # Score each device: (is_hardware, channels). Hardware devices
        # (names containing "hw:" or "HDA" or a USB vendor name) rank
        # higher than virtual ones ("pipewire", "default", "dmix").
        # This avoids the case where pipewire reports 128 channels but
        # is actually broken ("No such file or directory").
        best_idx: int | None = None
        best_score: tuple[bool, int] = (False, 0)
        for i, d in enumerate(devices):
            if i in excluded:
                continue
            ch = int(d.get("max_input_channels", 0))
            if ch <= 0:
                continue
            name = str(d.get("name", ""))
            # Heuristic: real hardware has "hw:" or "HDA" or "USB" in
            # the ALSA device name. Virtual bridges are named
            # "pipewire", "default", "dmix", "null", etc.
            is_hardware = any(
                tag in name for tag in ("hw:", "HDA", "USB", "Audio", "Codec")
            ) and not any(
                tag in name.lower() for tag in ("pipewire", "default", "dmix", "null")
            )
            score = (is_hardware, ch)
            if score > best_score:
                best_idx = i
                best_score = score
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
            # Pick the device to open. Start with the configured device.
            # If it's gone (query_devices raises), fall back to a
            # scanned device or system default. We pick ONE device and
            # try it ONCE — calling InputStream() multiple times in a
            # single press corrupts PortAudio state (the -9999 "not
            # initialized" crash). The blacklist ensures we don't try
            # the same broken device on the next press.
            device = self.device
            replaced_configured = False
            # If the configured device is blacklisted (failed in a
            # previous press) or gone (query_devices raises), fall
            # back to a scanned device or system default.
            if device is not None and (
                device in self._blacklisted_devices
                or not self._device_is_present(device)
            ):
                if device in self._blacklisted_devices:
                    print(
                        f"[warn] configured audio device {device} previously failed; "
                        f"trying alternative",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        f"[warn] configured audio device {device} is not available",
                        file=sys.stderr,
                        flush=True,
                    )
                device = None
                self._default_blacklisted = False
                replaced_configured = True
            if device is None and self.mic_failover_scan:
                # Fall back to scanning only when (a) a configured
                # device was pinned but failed/went missing, or (b)
                # the system default itself failed on a previous
                # press. When the user left device=null (unconfigured)
                # and the default hasn't failed before, use the system
                # default as-is — it may be a pipewire bridge that is
                # perfectly usable at its native rate.
                if replaced_configured or self._default_blacklisted:
                    scanned = self._find_fallback_device(
                        exclude=self._blacklisted_devices
                    )
                    if scanned is not None:
                        device = scanned
            # Determine the sample rate to use. Try the target rate
            # (16kHz) first, but if the device is known to not support
            # it, use the device's native rate and resample later.
            # We query the device's default sample rate upfront so we
            # only call InputStream() ONCE — calling it multiple times
            # in a single press corrupts PortAudio state (-9999).
            open_rate = self.sample_rate
            if device is not None:
                try:
                    dev_info = sd.query_devices(device)
                    native_rate = int(dev_info["default_samplerate"])
                    if native_rate != self.sample_rate:
                        # The device's default rate differs from our
                        # target. We'll try the native rate — most ALSA
                        # devices only support their default rate, not
                        # arbitrary rates. If 16kHz isn't in the
                        # supported list, using it will fail and
                        # corrupt PortAudio.
                        open_rate = native_rate
                except Exception:
                    pass
            elif device is None:
                # System default — query the default device's rate.
                try:
                    default_info = sd.query_devices(sd.default.device[0])
                    native_rate = int(default_info["default_samplerate"])
                    if native_rate != self.sample_rate:
                        open_rate = native_rate
                except Exception:
                    pass
            self._capture_sample_rate = open_rate
            # Try to open the stream — exactly ONE InputStream() call.
            try:
                self._stream = sd.InputStream(
                    samplerate=open_rate,
                    channels=self.channels,
                    dtype="float32",
                    device=device,
                    callback=self._on_audio,
                )
                self._stream.start()
                self._recording = True
                if open_rate != self.sample_rate:
                    print(
                        f"[info] recording at {open_rate}Hz "
                        f"(device doesn't support {self.sample_rate}Hz; "
                        f"will resample)",
                        file=sys.stderr,
                        flush=True,
                    )
                self._blacklisted_devices.clear()
                self._default_blacklisted = False
                return
            except Exception as exc:
                # Close any half-open stream.
                if self._stream is not None:
                    leaked = self._stream
                    self._stream = None
                    try:
                        leaked.close()
                    except Exception:
                        pass
                # Single failure — blacklist and report. No second
                # InputStream() attempt (would corrupt PortAudio).
                if device is not None:
                    self._blacklisted_devices.add(device)
                else:
                    # The system default failed — remember it so the
                    # next press scans for a real hardware device
                    # instead of trying the broken default again.
                    self._default_blacklisted = True
                reason = _describe_audio_error(exc)
                print(
                    f"[error] could not open audio input stream: {reason}. "
                    f"Check that a microphone is connected and that speakinput "
                    f"has Microphone permission in System Settings → Privacy "
                    f"& Security → Microphone. If a USB/Bluetooth mic is "
                    f"configured, try setting `device = null` in config.toml "
                    f"to use the system default instead. (Original error: "
                    f"{type(exc).__name__}: {exc})",
                    file=sys.stderr,
                    flush=True,
                )
                self._recording = False
                raise AudioError(
                    f"audio stream open failed: {reason}"
                ) from exc

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
        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        # Resample if the capture rate differs from the target rate
        # (whisper needs 16kHz). We open at the device's native rate
        # when it doesn't support 16kHz natively, then resample here.
        if self._capture_sample_rate != self.sample_rate and audio.size > 0:
            audio = _resample(audio, self._capture_sample_rate, self.sample_rate)
        return audio

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
