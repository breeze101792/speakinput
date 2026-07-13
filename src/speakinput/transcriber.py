"""Speech-to-text abstraction with a whisper.cpp (pywhispercpp) implementation."""

from __future__ import annotations

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


class WhisperCppTranscriber:
    """Wraps pywhispercpp.Model with our config defaults.

    The model is loaded eagerly in the constructor. The caller is expected to
    have already ensured the model file exists on disk (via
    `speakinput.models.ensure_model`) — passing a bare name still works for
    the v1 convenience path where pywhispercpp will auto-download on Model()
    init, but the recommended flow is the explicit bootstrap step.
    """

    def __init__(
        self,
        model: str | Path = "small",
        language: str = "auto",
        beam_size: int = 1,
        translate: bool = False,
        initial_prompt: str = "",
    ) -> None:
        if _WhisperModel is None:
            raise TranscriberError(
                "pywhispercpp is not installed. Install with `pip install pywhispercpp`."
            )
        self._model_path = str(model)
        # pywhispercpp uses `language=None` to enable per-utterance language
        # identification. The config "auto" maps to that here.
        self._language = None if language in (None, "", "auto") else language
        # pywhispercpp uses sampling strategy (0=greedy, 1=beam_search) set at
        # construction time. For v1 we always use greedy; the config field is
        # preserved for v2 when beam_search params will be wired in.
        self._beam_size = beam_size
        self._translate = translate
        # Whisper's initial_prompt is a lexical prior — tokenized once and
        # used to bias the decoder at the start of every transcription. Empty
        # string means "no prompt" (the whisper.cpp default behavior).
        self._initial_prompt = initial_prompt or None
        # params_sampling_strategy: 0 = GREEDY, 1 = BEAM_SEARCH
        strategy = 1 if self._beam_size and self._beam_size > 1 else 0
        self._model = _WhisperModel(  # type: ignore[call-arg]
            self._model_path,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
            params_sampling_strategy=strategy,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""
        # pywhispercpp expects a 1-D float32 array already at the model's
        # native rate (16 kHz for whisper). `sample_rate` is accepted for
        # interface parity with future Transcriber implementations.
        del sample_rate
        # When `_language is None` (auto), pywhispercpp runs the language
        # identifier on the first 30s of audio and prints the detected
        # language to stderr. We pass through whichever the caller set.
        segments = self._model.transcribe(  # type: ignore[attr-defined]
            audio,
            language=self._language,
            translate=self._translate,
            initial_prompt=self._initial_prompt,
        )
        # pywhispercpp can return the same segment multiple times when
        # whisper's temperature fallback chain re-samples the same low-
        # confidence region. Real speech segments are unique; collapsing
        # consecutive identical segments removes the duplicates without
        # affecting correct output. Whisper also emits a `[BLANK_AUDIO]`
        # marker for non-speech regions; drop it.
        _NO_SPEECH_MARKERS = ("[BLANK_AUDIO]",)
        seen: list[str] = []
        for seg in segments:
            text = getattr(seg, "text", "").strip()
            if not text or text in _NO_SPEECH_MARKERS:
                continue
            if seen and seen[-1] == text:
                continue
            seen.append(text)
        return " ".join(seen).strip()
