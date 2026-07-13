"""Audio capture from the default input device."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
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
    def is_recording(self) -> bool: ...


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

    def _require_sounddevice(self) -> None:
        if sd is None:
            raise AudioError(
                f"sounddevice is not installed: {_SD_IMPORT_ERROR}. "
                "Install with `pip install sounddevice`."
            )

    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        if self._recording:
            return
        self._require_sounddevice()
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            callback=self._on_audio,
        )
        self._stream.start()
        self._recording = True

    def _on_audio(self, indata, frames, time, status) -> None:  # noqa: ANN001 (sounddevice API)
        # status flags (overflow/underflow) are non-fatal; keep the audio and
        # let the caller surface the issue if needed.
        if self._chunks is not None:
            # Copy because the buffer is reused by PortAudio.
            self._chunks.append(indata.copy().reshape(-1))

    def stop(self) -> np.ndarray:
        if not self._recording:
            return np.zeros(0, dtype=np.float32)
        assert self._stream is not None
        self._stream.stop()
        self._stream.close()
        self._recording = False
        chunks = self._chunks or []
        self._chunks = None
        self._stream = None
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

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
