"""Model bootstrap: ensure the configured whisper model is downloaded.

Separated from the transcriber so the rest of the app doesn't need to think
about file paths or downloads. The model file is downloaded once into pywhispercpp's
cache directory and reused on subsequent runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from pywhispercpp import utils as _pw_utils
except ImportError:  # pragma: no cover - pywhispercpp is a hard dep
    _pw_utils = None  # type: ignore[assignment]


# English-only models and the multilingual model with the same size tier.
# Used by `resolve_for_language` to auto-upgrade an `.en` model to its
# multilingual counterpart when the user requests a non-English language.
_EN_TO_MULTILINGUAL = {
    "tiny.en": "tiny",
    "base.en": "base",
    "small.en": "small",
}

# Languages that require a multilingual model. `auto` also requires it
# because the model needs to be able to recognize non-English speech;
# an English-only model with `auto` would always detect "en" and be slower
# than just setting `language = "en"`.
_NEEDS_MULTILINGUAL = {"zh", "auto"}


class ModelNotFoundError(RuntimeError):
    """Raised when the configured model name is not in the curated allowlist."""


class ModelDownloadError(RuntimeError):
    """Raised when pywhispercpp fails to download or locate the model."""


def _is_path_like(name: str) -> bool:
    return "/" in name or name.endswith(".bin")


def resolve_for_language(model: str, language: str) -> tuple[str, str | None]:
    """Return the (possibly upgraded) model name and an info message.

    If the configured model is English-only (`*.en`) and the language
    requires multilingual support, swap it for the same-tier multilingual
    model. Returns the new model name; the info message is non-None only
    when a swap actually happened. The user is told about the swap so it's
    not a silent surprise.

    Absolute paths (e.g. `/path/to/custom.bin`) are returned unchanged.
    """
    if _is_path_like(model):
        return model, None
    if language in _NEEDS_MULTILINGUAL and model in _EN_TO_MULTILINGUAL:
        upgraded = _EN_TO_MULTILINGUAL[model]
        msg = (
            f"model {model!r} is English-only; upgrading to {upgraded!r} "
            f"so language={language!r} works"
        )
        return upgraded, msg
    return model, None


def ensure_model(name: str) -> Path:
    """Ensure `name` is available on disk; download if not.

    Returns the absolute path to the model file. For our curated model names
    (`tiny.en`, `base.en`, `small.en`), the file is downloaded into
    pywhispercpp's cache dir on first use. For absolute paths, just verifies
    existence and returns the path.

    Raises:
        ModelNotFoundError: the name is not a known model and not a path.
        ModelDownloadError: the download failed.
        FileNotFoundError: a path was given but the file does not exist.
    """
    if _pw_utils is None:
        raise ModelDownloadError(
            "pywhispercpp is not installed. Install with `pip install pywhispercpp`."
        )

    # User-supplied absolute path: just verify.
    if _is_path_like(name):
        p = Path(name).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"model file not found: {p}")
        return p.resolve()

    # Curated name: defer to pywhispercpp's downloader.
    if name not in _pw_utils.AVAILABLE_MODELS:
        raise ModelNotFoundError(
            f"unknown whisper model: {name!r}. "
            f"Use --list-models to see available options, or pass an absolute path."
        )

    print(f"checking model {name}...", file=sys.stderr, flush=True)
    try:
        path = _pw_utils.download_model(name)
    except Exception as exc:
        raise ModelDownloadError(f"failed to download model {name!r}: {exc}") from exc
    if not path:
        raise ModelDownloadError(f"download_model({name!r}) returned no path")
    resolved = Path(path).resolve()
    print(f"model ready: {resolved}", file=sys.stderr, flush=True)
    return resolved
