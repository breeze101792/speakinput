"""Tests for the transcriber. Mocks pywhispercpp so tests are CPU/disk-free."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def fake_pywhispercpp(monkeypatch):
    """Patch the module-level `_WhisperModel` reference inside the transcriber
    module — that's the symbol the constructor actually checks. Replacing only
    `sys.modules['pywhispercpp']` is not enough because the import was already
    captured at module load time.
    """
    from speakinput import transcriber as t_mod

    fake_cls = MagicMock()
    monkeypatch.setattr(t_mod, "_WhisperModel", fake_cls, raising=False)
    return fake_cls


def test_raises_when_pywhispercpp_missing(monkeypatch):
    from speakinput import transcriber as t_mod
    from speakinput.transcriber import TranscriberError, WhisperCppTranscriber

    monkeypatch.setattr(t_mod, "_WhisperModel", None, raising=False)
    with pytest.raises(TranscriberError, match="pywhispercpp"):
        WhisperCppTranscriber()


def test_constructor_loads_model_eagerly(fake_pywhispercpp):
    """The model is loaded in __init__, not deferred to first transcribe()."""
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    fake_pywhispercpp.return_value = model_instance

    WhisperCppTranscriber(model="/path/to/model.bin", language="en", beam_size=1)
    fake_pywhispercpp.assert_called_once()
    # The path we pass should be the one given to the Model constructor.
    assert fake_pywhispercpp.call_args.args[0] == "/path/to/model.bin"


def test_constructor_accepts_path_object(fake_pywhispercpp):
    from speakinput.transcriber import WhisperCppTranscriber

    WhisperCppTranscriber(model=Path("/some/model.bin"))
    assert fake_pywhispercpp.call_args.args[0] == "/some/model.bin"


def test_transcribe_empty_audio_returns_empty(fake_pywhispercpp):
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber()
    out = t.transcribe(np.zeros(0, dtype=np.float32), 16000)
    assert out == ""
    # Model was loaded (eagerly) but not invoked for empty audio.
    model_instance.transcribe.assert_not_called()


def test_transcribe_concatenates_multiple_segments(fake_pywhispercpp):
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    model_instance.transcribe.return_value = [
        MagicMock(text="foo "),
        MagicMock(text="bar "),
        MagicMock(text="baz"),
    ]
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber()
    out = t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert out == "foo bar baz"


def test_transcribe_skips_segments_without_text_attribute(fake_pywhispercpp):
    """Defensive: a model might yield plain dicts or other shapes; we tolerate it."""
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    model_instance.transcribe.return_value = [
        MagicMock(text="ok "),
        object(),  # no .text — getattr with default '' handles it
        MagicMock(text="done"),
    ]
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber()
    out = t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert out == "ok done"


def test_transcribe_passes_config_through(fake_pywhispercpp):
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    model_instance.transcribe.return_value = [MagicMock(text="x")]
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber(model="small.en", language="en", beam_size=3)
    t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    kwargs = model_instance.transcribe.call_args.kwargs
    assert kwargs["language"] == "en"
    assert kwargs["translate"] is False
    assert "beam_size" not in kwargs
    assert "sample_rate" not in kwargs  # pywhispercpp infers from audio
    # beam_size=3 should have selected beam_search (strategy=1) at construction.
    assert fake_pywhispercpp.call_args.kwargs["params_sampling_strategy"] == 1


def test_transcribe_with_greedy_strategy_for_beam_size_one(fake_pywhispercpp):
    from speakinput.transcriber import WhisperCppTranscriber

    fake_pywhispercpp.return_value = MagicMock()

    t = WhisperCppTranscriber(beam_size=1)
    t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert fake_pywhispercpp.call_args.kwargs["params_sampling_strategy"] == 0


def test_transcribe_with_translate_flag(fake_pywhispercpp):
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    model_instance.transcribe.return_value = [MagicMock(text="hola")]
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber(translate=True)
    t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert model_instance.transcribe.call_args.kwargs["translate"] is True


# --- language handling -----------------------------------------------------


def test_auto_language_is_passed_as_none(fake_pywhispercpp):
    """`language="auto"` must reach pywhispercpp as `None` so the model runs
    per-utterance language identification."""
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    model_instance.transcribe.return_value = [MagicMock(text="你好")]
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber(model="small", language="auto")
    t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert model_instance.transcribe.call_args.kwargs["language"] is None


def test_explicit_language_is_passed_through(fake_pywhispercpp):
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    model_instance.transcribe.return_value = [MagicMock(text="你好")]
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber(model="small", language="zh")
    t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert model_instance.transcribe.call_args.kwargs["language"] == "zh"


# --- duplicate-segment defense --------------------------------------------


def test_transcribe_collapses_consecutive_duplicate_segments(fake_pywhispercpp):
    """pywhispercpp can return the same segment multiple times when whisper's
    temperature fallback chain re-samples a low-confidence region. We must
    collapse consecutive duplicates so the user doesn't see the same phrase
    typed three times."""
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    model_instance.transcribe.return_value = [
        MagicMock(text="Hello, how are you? "),
        MagicMock(text="Hello, how are you? "),
        MagicMock(text="Hello, how are you?"),
    ]
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber()
    out = t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert out == "Hello, how are you?"


def test_transcribe_keeps_non_consecutive_duplicates(fake_pywhispercpp):
    """Only collapse when two identical segments are *adjacent* — legitimate
    repeated phrases (e.g. "yes yes" or the same word twice in a sentence)
    are still distinct segments with content between them."""
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    model_instance.transcribe.return_value = [
        MagicMock(text="yes "),
        MagicMock(text="no "),
        MagicMock(text="yes"),
    ]
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber()
    out = t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert out == "yes no yes"


def test_transcribe_drops_blank_audio_marker(fake_pywhispercpp):
    """The model's no-speech marker should never reach the user."""
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    model_instance.transcribe.return_value = [
        MagicMock(text="[BLANK_AUDIO]"),
        MagicMock(text=" "),
        MagicMock(text=""),
    ]
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber()
    out = t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert out == ""
