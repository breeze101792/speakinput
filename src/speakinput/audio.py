"""Audio capture from the default input device."""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

try:
    import sounddevice as sd
except ImportError as _exc:  # pragma: no cover - exercised only when missing
    sd = None  # type: ignore[assignment]
    _SD_IMPORT_ERROR = _exc
else:
    _SD_IMPORT_ERROR = None


class AudioError(RuntimeError):
    pass


class Recorder(Protocol):
    def start(self) -> None: ...
    def stop(self) -> np.ndarray: ...
    def drain(self) -> np.ndarray: ...
    def close(self) -> None: ...
    def is_recording(self) -> bool: ...
    def current_rms(self) -> float: ...


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
    # The most recent chunk's RMS, updated from the audio callback and
    # read by the auto-stop watchdog. Guarded by `_rms_lock` because the
    # callback and the watchdog run on different threads.
    _last_rms: float = 0.0
    _rms_lock: threading.Lock = field(default_factory=threading.Lock)

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

    def start(self) -> None:
        if self._recording:
            return
        self._require_sounddevice()
        self._chunks = []
        with self._rms_lock:
            self._last_rms = 0.0
        # Fall back to system default if the pinned device disappeared
        # since the last press (USB unplug, Bluetooth headset
        # disconnected, etc.). The check is sub-ms; the stream-open
        # that follows is the real cost (~30-100ms on macOS). When
        # `device is None` the system default is used and re-resolved
        # by PortAudio on every call, so the fallback is unnecessary.
        device = self.device
        if device is not None and not self._device_is_present(device):
            print(
                f"[warn] configured audio device {device} is not available; "
                f"falling back to system default microphone",
                file=sys.stderr,
                flush=True,
            )
            device = None
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=device,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            # No microphone at all (system default also gone), or the
            # OS denied us access. Surface a clear message instead of
            # letting the press fail silently. The user needs to know
            # WHY their key did nothing.
            print(
                f"[error] could not open audio input stream: {exc}. "
                f"Check that a microphone is connected and that speakinput "
                f"has Microphone permission in System Settings → Privacy "
                f"& Security → Microphone.",
                file=sys.stderr,
                flush=True,
            )
            self._stream = None
            self._recording = False
            raise AudioError(f"audio stream open failed: {exc}") from exc
        self._recording = True

    def _on_audio(self, indata, frames, time, status) -> None:  # noqa: ANN001 (sounddevice API)
        # status flags (overflow/underflow) are non-fatal; keep the audio and
        # let the caller surface the issue if needed.
        chunk = indata.copy().reshape(-1)
        if self._chunks is not None:
            self._chunks.append(chunk)
        # Track the most recent chunk's RMS for the auto-stop watchdog.
        # Computed once per callback so the watchdog's polling loop is a
        # cheap lock+read, not a full-buffer scan.
        if chunk.size:
            rms = float(np.sqrt(np.mean(chunk * chunk)))
        else:
            rms = 0.0
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
        """
        chunks = self._chunks or []
        self._chunks = []
        with self._rms_lock:
            self._last_rms = 0.0
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def close(self) -> None:
        """Stop and close the PortAudio stream. Idempotent.

        After `close()`, the recorder is no longer recording and must
        be `start()`-ed again to capture more audio. Any in-flight
        audio callbacks from PortAudio that arrive after close are
        silently dropped (the `_chunks` list is None).
        """
        if self._stream is None:
            return
        try:
            self._stream.stop()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass
        self._stream = None
        self._recording = False
        # Don't drop _chunks here — `drain()` may still want to read
        # what was buffered before close. _on_audio appends are still
        # safe because the list object isn't replaced.
        # Reset RMS so a stale chunk callback doesn't leak across close.
        with self._rms_lock:
            self._last_rms = 0.0

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
