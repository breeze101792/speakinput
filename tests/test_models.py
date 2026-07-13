"""Tests for the model bootstrap module."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_pw_utils(monkeypatch):
    """Replace the pywhispercpp.utils module the bootstrap uses."""
    from speakinput import models as m

    fake = MagicMock()
    fake.AVAILABLE_MODELS = ["tiny.en", "base.en", "small.en", "large-v3"]
    fake.download_model = MagicMock(return_value="/cache/base.en.bin")
    monkeypatch.setattr(m, "_pw_utils", fake, raising=False)
    return fake


# --- known model name (download path) ---------------------------------------


def test_ensure_known_model_calls_downloader(fake_pw_utils, capsys):
    from speakinput.models import ensure_model

    path = ensure_model("base.en")
    assert str(path) == "/cache/base.en.bin"
    fake_pw_utils.download_model.assert_called_once_with("base.en")
    captured = capsys.readouterr()
    assert "checking model" in captured.err
    assert "model ready" in captured.err


def test_ensure_unknown_known_model_raises(fake_pw_utils):
    from speakinput.models import ModelNotFoundError, ensure_model

    with pytest.raises(ModelNotFoundError, match="unknown whisper model"):
        ensure_model("not-a-model")


def test_ensure_downloader_failure_raises(fake_pw_utils):
    from speakinput.models import ModelDownloadError, ensure_model

    fake_pw_utils.download_model.side_effect = RuntimeError("network down")
    with pytest.raises(ModelDownloadError, match="failed to download"):
        ensure_model("base.en")


def test_ensure_downloader_returns_empty_raises(fake_pw_utils):
    from speakinput.models import ModelDownloadError, ensure_model

    fake_pw_utils.download_model.return_value = None
    with pytest.raises(ModelDownloadError, match="no path"):
        ensure_model("base.en")


# --- path input (no download) -----------------------------------------------


def test_ensure_path_verifies_existence(tmp_path, monkeypatch):
    """An absolute .bin path is accepted as-is, no download attempted."""
    from speakinput import models as m

    fake = MagicMock()
    fake.AVAILABLE_MODELS = []
    monkeypatch.setattr(m, "_pw_utils", fake, raising=False)

    p = tmp_path / "custom.bin"
    p.write_bytes(b"\x00" * 10)
    out = m.ensure_model(str(p))
    assert out == p.resolve()
    fake.download_model.assert_not_called()


def test_ensure_missing_path_raises(tmp_path, monkeypatch):
    from speakinput import models as m

    fake = MagicMock()
    monkeypatch.setattr(m, "_pw_utils", fake, raising=False)
    with pytest.raises(FileNotFoundError, match="model file not found"):
        m.ensure_model(str(tmp_path / "missing.bin"))


# --- pywhispercpp missing ---------------------------------------------------


def test_ensure_without_pywhispercpp_raises(monkeypatch):
    from speakinput import models as m
    from speakinput.models import ModelDownloadError

    monkeypatch.setattr(m, "_pw_utils", None, raising=False)
    with pytest.raises(ModelDownloadError, match="pywhispercpp is not installed"):
        m.ensure_model("base.en")
