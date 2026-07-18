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


# --- initial_prompt --------------------------------------------------------


def test_transcribe_passes_initial_prompt(fake_pywhispercpp):
    """A non-empty initial_prompt must reach pywhispercpp so the decoder
    gets the lexical prior."""
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    model_instance.transcribe.return_value = [MagicMock(text="kubectl")]
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber(initial_prompt="kubectl apply -f deployment.yaml")
    t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert (
        model_instance.transcribe.call_args.kwargs["initial_prompt"]
        == "kubectl apply -f deployment.yaml"
    )


def test_transcribe_empty_initial_prompt_becomes_none(fake_pywhispercpp):
    """The whisper.cpp C library prefers None over "" for the absent-prompt
    case. We normalize here so the user's `initial_prompt = ""` in
    config.toml behaves like "no prompt at all"."""
    from speakinput.transcriber import WhisperCppTranscriber

    model_instance = MagicMock()
    fake_pywhispercpp.return_value = model_instance

    t = WhisperCppTranscriber(initial_prompt="")
    t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert model_instance.transcribe.call_args.kwargs["initial_prompt"] is None


def test_constructor_default_has_no_prompt(fake_pywhispercpp):
    """Default constructor must not bias the decoder (initial_prompt=None)."""
    from speakinput.transcriber import WhisperCppTranscriber

    fake_pywhispercpp.return_value = MagicMock()

    t = WhisperCppTranscriber()
    t.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert fake_pywhispercpp.return_value.transcribe.call_args.kwargs["initial_prompt"] is None


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


# --- GPU backend probe + context_params ----------------------------------
#
# The probe caches its result for the lifetime of the process. Tests that
# patch the lib path need to clear the cache explicitly.


@pytest.fixture(autouse=False)
def _clear_gpu_probe_cache():
    """Clear the lru_cache on _probe_gpu_backend between tests.

    A test may have replaced `_probe_gpu_backend` with a plain lambda
    (which has no `cache_clear`); we tolerate that and silently skip.
    """
    from speakinput import transcriber as t_mod

    _safe_cache_clear(t_mod._probe_gpu_backend)
    yield
    _safe_cache_clear(t_mod._probe_gpu_backend)


def _safe_cache_clear(fn):
    cache_clear = getattr(fn, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def _write_fake_lib(tmp_path, content: bytes) -> Path:
    """Write a fake libwhisper.so at the given path and return the path."""
    p = tmp_path / "libwhisper.so"
    p.write_bytes(content)
    return p


def test_probe_returns_none_for_cpu_only_wheel(tmp_path, monkeypatch, _clear_gpu_probe_cache):
    from speakinput import transcriber as t_mod

    fake = _write_fake_lib(tmp_path, b"this is a CPU-only whisper.cpp build")
    # Make _locate_libwhisper() return our fake lib without needing the
    # real _pywhispercpp module's `__file__` to point at it.
    monkeypatch.setattr(t_mod, "_locate_libwhisper", lambda: fake)
    assert t_mod._probe_gpu_backend() is None


def test_probe_detects_cuda_when_cuda_symbols_present(
    tmp_path, monkeypatch, _clear_gpu_probe_cache
):
    from speakinput import transcriber as t_mod

    fake = _write_fake_lib(
        tmp_path,
        b"ggml-cuda init successful; loaded ggml_cuda backend v12.0",
    )
    monkeypatch.setattr(t_mod, "_locate_libwhisper", lambda: fake)
    assert t_mod._probe_gpu_backend() == "cuda"


def test_probe_detects_vulkan_when_vulkan_symbols_present(
    tmp_path, monkeypatch, _clear_gpu_probe_cache
):
    from speakinput import transcriber as t_mod

    fake = _write_fake_lib(tmp_path, b"ggml-vulkan backend ready")
    monkeypatch.setattr(t_mod, "_locate_libwhisper", lambda: fake)
    assert t_mod._probe_gpu_backend() == "vulkan"


def test_probe_priority_cuda_beats_vulkan(
    tmp_path, monkeypatch, _clear_gpu_probe_cache
):
    """When a lib has BOTH CUDA and Vulkan strings, CUDA wins (priority order)."""
    from speakinput import transcriber as t_mod

    fake = _write_fake_lib(
        tmp_path,
        b"ggml-cuda v12; ggml-vulkan fallback; ggml-cublas init",
    )
    monkeypatch.setattr(t_mod, "_locate_libwhisper", lambda: fake)
    assert t_mod._probe_gpu_backend() == "cuda"


def test_probe_returns_none_when_lib_missing(
    monkeypatch, _clear_gpu_probe_cache
):
    from speakinput import transcriber as t_mod

    monkeypatch.setattr(t_mod, "_locate_libwhisper", lambda: None)
    assert t_mod._probe_gpu_backend() is None


def test_resolve_context_params_auto_picks_gpu_when_available(
    monkeypatch, _clear_gpu_probe_cache
):
    from speakinput import transcriber as t_mod

    monkeypatch.setattr(t_mod, "_probe_gpu_backend", lambda: "vulkan")
    params = t_mod._resolve_context_params(use_gpu=None, gpu_device=0)
    assert params == {"use_gpu": True, "gpu_device": 0, "flash_attn": True}


def test_resolve_context_params_auto_skips_when_cpu_only(
    monkeypatch, _clear_gpu_probe_cache
):
    from speakinput import transcriber as t_mod

    monkeypatch.setattr(t_mod, "_probe_gpu_backend", lambda: None)
    assert t_mod._resolve_context_params(use_gpu=None, gpu_device=0) == {}


def test_resolve_context_params_force_off_ignores_available_gpu(
    monkeypatch, _clear_gpu_probe_cache
):
    from speakinput import transcriber as t_mod

    monkeypatch.setattr(t_mod, "_probe_gpu_backend", lambda: "cuda")
    # Even though a GPU is available, an explicit `use_gpu=False` must
    # disable it (e.g. a user with a weak iGPU wants CPU for stability).
    assert t_mod._resolve_context_params(use_gpu=False, gpu_device=0) == {}


def test_resolve_context_params_force_on_with_cpu_lib_warns(
    monkeypatch, _clear_gpu_probe_cache, capsys
):
    from speakinput import transcriber as t_mod

    monkeypatch.setattr(t_mod, "_probe_gpu_backend", lambda: None)
    # No GPU present, but user explicitly forced it on. We must NOT
    # crash — log a warning to stderr and return {}.
    params = t_mod._resolve_context_params(use_gpu=True, gpu_device=0)
    assert params == {}
    captured = capsys.readouterr()
    assert "use_gpu=true" in captured.err
    assert "GPU acceleration" in captured.err


def test_resolve_context_params_passes_through_gpu_device(
    monkeypatch, _clear_gpu_probe_cache
):
    from speakinput import transcriber as t_mod

    monkeypatch.setattr(t_mod, "_probe_gpu_backend", lambda: "vulkan")
    params = t_mod._resolve_context_params(use_gpu=None, gpu_device=2)
    assert params["gpu_device"] == 2


def test_whispercpp_transcriber_passes_context_params(
    fake_pywhispercpp, _clear_gpu_probe_cache, monkeypatch
):
    """When use_gpu=True, the constructor must forward `context_params` to
    pywhispercpp.Model with use_gpu / gpu_device / flash_attn set."""
    from speakinput import transcriber as t_mod
    from speakinput.transcriber import WhisperCppTranscriber

    # Force the probe to see a GPU so the context_params dict is built.
    monkeypatch.setattr(t_mod, "_probe_gpu_backend", lambda: "vulkan")
    WhisperCppTranscriber(use_gpu=True, gpu_device=1)
    kwargs = fake_pywhispercpp.call_args.kwargs
    assert kwargs["context_params"] == {
        "use_gpu": True,
        "gpu_device": 1,
        "flash_attn": True,
    }


def test_whispercpp_transcriber_omits_context_params_when_cpu(
    fake_pywhispercpp, _clear_gpu_probe_cache
):
    """use_gpu=False → no context_params kwarg, so pywhispercpp uses its
    CPU defaults. Avoids passing an empty {} that could trigger a
    strictness check on the C side."""
    from speakinput.transcriber import WhisperCppTranscriber

    WhisperCppTranscriber(use_gpu=False)
    assert "context_params" not in fake_pywhispercpp.call_args.kwargs


def test_whispercpp_transcriber_passes_n_threads(fake_pywhispercpp):
    """n_threads>0 is forwarded as a kwarg; n_threads=0 (default) is omitted
    so pywhispercpp picks its own default (min(4, cores))."""
    from speakinput.transcriber import WhisperCppTranscriber

    WhisperCppTranscriber(n_threads=8)
    assert fake_pywhispercpp.call_args.kwargs["n_threads"] == 8

    # Reset and try n_threads=0 (the default).
    fake_pywhispercpp.reset_mock()
    WhisperCppTranscriber()
    assert "n_threads" not in fake_pywhispercpp.call_args.kwargs


def test_whispercpp_transcriber_rejects_negative_gpu_device(
    fake_pywhispercpp, _clear_gpu_probe_cache
):
    from speakinput.transcriber import TranscriberError, WhisperCppTranscriber

    with pytest.raises(TranscriberError, match="gpu_device"):
        WhisperCppTranscriber(use_gpu=True, gpu_device=-1)


def test_whispercpp_transcriber_rejects_negative_n_threads(fake_pywhispercpp):
    from speakinput.transcriber import TranscriberError, WhisperCppTranscriber

    with pytest.raises(TranscriberError, match="n_threads"):
        WhisperCppTranscriber(n_threads=-1)


def test_gpu_summary_includes_install_hint_for_cpu_only(
    monkeypatch, _clear_gpu_probe_cache, capsys
):
    """The summary line that the startup banner prints must point users
    at the README when the wheel is CPU-only — so they know to rebuild."""
    from speakinput import transcriber as t_mod

    monkeypatch.setattr(t_mod, "_probe_gpu_backend", lambda: None)
    summary = t_mod._gpu_summary(use_gpu=None, gpu_device=0)
    assert "cpu" in summary
    assert "GPU acceleration" in summary
    # And the force-on warning is logged too.
    t_mod._resolve_context_params(use_gpu=True, gpu_device=0)
    captured = capsys.readouterr()
    assert "use_gpu=true" in captured.err


def test_gpu_summary_names_the_backend_when_gpu_present(
    monkeypatch, _clear_gpu_probe_cache
):
    from speakinput import transcriber as t_mod

    monkeypatch.setattr(t_mod, "_probe_gpu_backend", lambda: "cuda")
    summary = t_mod._gpu_summary(use_gpu=None, gpu_device=0)
    assert "cuda" in summary
    assert "GPU 0" in summary


def test_gpu_summary_forced_off_explicit(monkeypatch, _clear_gpu_probe_cache):
    from speakinput import transcriber as t_mod

    monkeypatch.setattr(t_mod, "_probe_gpu_backend", lambda: "cuda")
    summary = t_mod._gpu_summary(use_gpu=False, gpu_device=0)
    assert "cpu" in summary
    assert "forced" in summary
