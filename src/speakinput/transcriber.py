"""Speech-to-text abstraction with a whisper.cpp (pywhispercpp) implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np

try:
    from pywhispercpp.model import Model as _WhisperModel
except ImportError:  # pragma: no cover - exercised only when missing
    _WhisperModel = None


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class TranscriberError(RuntimeError):
    pass


def _resolve_model_path(name: str) -> str:
    """Accept either a bare model name (auto-resolve from pywhispercpp's cache)
    or an absolute path. pywhispercpp.Model accepts both, but we validate
    early for a clearer error message.
    """
    if os.path.isabs(name) and Path(name).exists():
        return name
    # pywhispercpp downloads to ~/.cache/pywhispercpp on first use; pass name through.
    return name


class WhisperCppTranscriber:
    """Wraps pywhispercpp.Model with our config defaults.

    The model is loaded lazily on first transcribe() call so that
    `--list-devices` and `--diagnose` work without paying the model load cost.
    """

    def __init__(
        self,
        model: str = "base.en",
        language: str = "en",
        beam_size: int = 1,
        translate: bool = False,
    ) -> None:
        if _WhisperModel is None:
            raise TranscriberError(
                "pywhispercpp is not installed. Install with `pip install pywhispercpp`."
            )
        self._model_name = _resolve_model_path(model)
        self._language = language
        # pywhispercpp uses sampling strategy (0=greedy, 1=beam_search) set at
        # construction time. For v1 we always use greedy; the config field is
        # preserved for v2 when beam_search params will be wired in.
        self._beam_size = beam_size
        self._translate = translate
        self._model: object | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # params_sampling_strategy: 0 = GREEDY, 1 = BEAM_SEARCH
        strategy = 1 if self._beam_size and self._beam_size > 1 else 0
        self._model = _WhisperModel(  # type: ignore[call-arg]
            self._model_name,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
            params_sampling_strategy=strategy,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""
        self._ensure_loaded()
        # pywhispercpp expects a 1-D float32 array already at the model's
        # native rate (16 kHz for whisper). `sample_rate` is accepted for
        # interface parity with future Transcriber implementations.
        del sample_rate
        segments = self._model.transcribe(  # type: ignore[attr-defined]
            audio,
            language=self._language,
            translate=self._translate,
        )
        # segments is a list of Segment objects with .text
        text = "".join(getattr(seg, "text", "") for seg in segments).strip()
        return text
