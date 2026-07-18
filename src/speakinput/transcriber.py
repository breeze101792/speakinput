"""Speech-to-text abstraction with a whisper.cpp (pywhispercpp) implementation.

GPU acceleration
----------------

Whisper can run on CPU or on a GPU-backed whisper.cpp build. The
*build* time choice decides which backend (CUDA / Vulkan / Metal /
OpenCL / HIP) is available — the *runtime* only has to opt in via
`context_params={"use_gpu": True, ...}`. There is no app-level
backend selection: the wheel is built against one backend and that
is the only one it can use.

Detection at startup: `_probe_gpu_backend()` reads the symbols
embedded in the loaded `libwhisper.so` and reports the first GPU
backend it finds (priority order: cuda > vulkan > metal > hip >
opencl). If none is present, the wheel is CPU-only and we log a
one-line hint to the README's "GPU acceleration" section.

To enable GPU on Linux:

    # NVIDIA (primary — fastest on the user's RTX 4060 Ti)
    sudo pacman -S cuda
    GGML_CUDA=1 pip install --force-reinstall --no-cache \\
        git+https://github.com/absadiki/pywhispercpp

    # Vulkan (any vendor — works on AMD, Intel, ARM, NVIDIA)
    sudo pacman -S vulkan-icd-loader nvidia-utils   # or vulkan-radeon / vulkan-intel
    GGML_VULKAN=1 pip install --force-reinstall --no-cache \\
        git+https://github.com/absadiki/pywhispercpp

The startup banner shows which backend is active, so you can tell
at a glance whether the rebuild succeeded.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    from pywhispercpp.model import Model as _WhisperModel
except ImportError:  # pragma: no cover - exercised only when missing
    _WhisperModel = None


# --- GPU backend probe ----------------------------------------------------
#
# whisper.cpp embeds the names of its enabled backends as strings in the
# shipped `libwhisper.{so,dylib,dll}`. We scan for those strings instead
# of asking pywhispercpp (which exposes `use_gpu=True` as a single flag
# without telling you *which* backend is live). The probe is cached at
# module load — the loaded lib never changes for the lifetime of the
# process.

# Priority order: CUDA first (the user is on NVIDIA today and CUDA is
# the fastest on NVIDIA), then Vulkan (universal), then Metal (macOS),
# then HIP (AMD ROCm — less common, listed for completeness), then
# OpenCL (legacy fallback for hardware without Vulkan).
_GPU_BACKEND_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cuda", ("ggml-cuda", "ggml_cuda", "cublas")),
    ("vulkan", ("ggml-vulkan", "ggml_vulkan")),
    ("metal", ("ggml-metal", "ggml_metal", "coreml")),
    ("hip", ("ggml-hip", "ggml_hip", "rocm")),
    ("opencl", ("ggml-opencl", "ggml_opencl")),
)

_LIB_NAMES = ("libwhisper.so", "libwhisper.dylib", "whisper.dll")


def _locate_libwhisper() -> Path | None:
    """Return the path of the loaded whisper.cpp shared library, if findable."""
    try:
        import _pywhispercpp  # noqa: F401  (must be importable for any of this to work)
    except ImportError:
        return None
    # The .so / .dylib / .dl sits next to the extension module.
    for name in _LIB_NAMES:
        candidate = Path(_pywhispercpp.__file__).parent / name
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def _probe_gpu_backend() -> str | None:
    """Return the first GPU backend present in the loaded lib, or None for CPU.

    Order: cuda > vulkan > metal > hip > opencl. The result is cached
    so the file scan happens at most once per process.
    """
    lib = _locate_libwhisper()
    if lib is None:
        return None
    try:
        # Don't load the whole file into a Python str (it can be tens of MB);
        # `errors='replace'` keeps us safe for any non-UTF-8 bytes.
        with lib.open("rb") as f:
            blob = f.read()
    except OSError:
        return None
    haystack = blob.decode("utf-8", errors="replace")
    for name, markers in _GPU_BACKEND_MARKERS:
        if any(marker in haystack for marker in markers):
            return name
    return None


def _resolve_context_params(
    use_gpu: bool | None,
    gpu_device: int,
) -> dict[str, Any]:
    """Pick the right `context_params` dict for the user's intent + the loaded lib.

    `use_gpu=None` means "auto": enable GPU if a backend is present,
    else stay on CPU. `use_gpu=True` is the same but with a warning
    if no backend is present (so the user knows the install was a
    no-op). `use_gpu=False` is explicit CPU.
    """
    if use_gpu is False:
        return {}

    backend = _probe_gpu_backend()
    if backend is None:
        if use_gpu is True:
            print(
                "[transcribe] warning: use_gpu=true but no GPU backend found "
                "in the loaded pywhispercpp wheel — falling back to CPU. "
                "See README → 'GPU acceleration' for the rebuild command.",
                file=sys.stderr,
                flush=True,
            )
        return {}

    # GPU is available. `flash_attn` is supported on CUDA and Vulkan
    # builds; harmless on others (whisper.cpp ignores it if the
    # backend doesn't have a flash-attn path).
    return {
        "use_gpu": True,
        "gpu_device": gpu_device,
        "flash_attn": True,
    }


def _gpu_summary(use_gpu: bool | None, gpu_device: int) -> str:
    """One-line description of the chosen backend, for the startup banner."""
    if use_gpu is False:
        return "cpu (forced off)"
    backend = _probe_gpu_backend()
    if backend is None:
        return "cpu (wheel is CPU-only — see README → 'GPU acceleration')"
    params = _resolve_context_params(use_gpu, gpu_device)
    extras = []
    if params.get("flash_attn"):
        extras.append("flash_attn=on")
    extras_str = ", ".join(extras)
    if extras_str:
        return f"{backend} (GPU {gpu_device}, {extras_str})"
    return f"{backend} (GPU {gpu_device})"


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

    GPU use is controlled by the `use_gpu` kwarg:
    - `None` (default) — auto-detect: enable GPU if the wheel has a
      GPU backend, else stay on CPU.
    - `True` — force GPU on. If the wheel is CPU-only, log a warning
      and fall back to CPU (don't crash; the user will see the
      warning in stderr).
    - `False` — force CPU.
    """

    def __init__(
        self,
        model: str | Path = "small",
        language: str = "auto",
        beam_size: int = 1,
        translate: bool = False,
        initial_prompt: str = "",
        *,
        use_gpu: bool | None = None,
        gpu_device: int = 0,
        n_threads: int = 0,
    ) -> None:
        if _WhisperModel is None:
            raise TranscriberError(
                "pywhispercpp is not installed. Install with `pip install pywhispercpp`."
            )
        if gpu_device < 0:
            raise TranscriberError(
                f"gpu_device must be >= 0, got {gpu_device}"
            )
        if n_threads < 0:
            raise TranscriberError(
                f"n_threads must be >= 0 (0 = auto), got {n_threads}"
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
        context_params = _resolve_context_params(use_gpu, gpu_device)
        # n_threads=0 → use pywhispercpp's default (min(4, cores)).
        # n_threads>0 → forward the user's choice.
        model_kwargs: dict[str, Any] = dict(
            params_sampling_strategy=strategy,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
        )
        if context_params:
            model_kwargs["context_params"] = context_params
        if n_threads > 0:
            model_kwargs["n_threads"] = n_threads
        self._model = _WhisperModel(  # type: ignore[call-arg]
            self._model_path,
            **model_kwargs,
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


# Allow `from speakinput.transcriber import _GPU_BACKEND_MARKERS` for tests
# that want to assert against the table without re-defining it.
__all__ = [
    "Transcriber",
    "TranscriberError",
    "WhisperCppTranscriber",
    "_GPU_BACKEND_MARKERS",
    "_LIB_NAMES",
    "_locate_libwhisper",
    "_probe_gpu_backend",
    "_resolve_context_params",
    "_gpu_summary",
]
