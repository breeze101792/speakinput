from pathlib import Path

import pytest

from speakinput.config import (
    AudioConfig,
    Config,
    HotkeyConfig,
    InjectConfig,
    STTConfig,
    load_config,
    write_default_config,
)


def test_default_construction():
    cfg = Config()
    assert cfg.stt == STTConfig()
    assert cfg.audio == AudioConfig()
    assert cfg.hotkey == HotkeyConfig()
    assert cfg.inject == InjectConfig()


def test_default_inject_has_trailing_space_on():
    from speakinput.config import InjectConfig

    assert InjectConfig().trailing_space is True
    assert Config().inject.trailing_space is True


def test_from_dict_reads_trailing_space():
    cfg = Config.from_dict({"inject": {"trailing_space": False}})
    assert cfg.inject.trailing_space is False


def test_with_overrides_trailing_space():
    cfg = Config()
    new = cfg.with_overrides(trailing_space=False)
    assert new.inject.trailing_space is False
    assert cfg.inject.trailing_space is True  # original untouched


def test_validation_rejects_unknown_model():
    with pytest.raises(ValueError, match="stt.model"):
        Config(stt=STTConfig(model="bogus")).validate()


def test_validation_rejects_unknown_hotkey():
    with pytest.raises(ValueError, match="hotkey.key"):
        Config(hotkey=HotkeyConfig(key="super")).validate()


def test_validation_rejects_zero_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        Config(audio=AudioConfig(sample_rate=0)).validate()


def test_validation_rejects_out_of_range_beam_size():
    with pytest.raises(ValueError, match="beam_size"):
        Config(stt=STTConfig(beam_size=0)).validate()
    with pytest.raises(ValueError, match="beam_size"):
        Config(stt=STTConfig(beam_size=11)).validate()


def test_from_dict_overrides_sections():
    cfg = Config.from_dict(
        {
            "stt": {"model": "small.en", "language": "en", "beam_size": 5},
            "audio": {"device": 2, "sample_rate": 16000},
            "hotkey": {"key": "f12"},
            "inject": {"restore_clipboard_ms": 100},
        }
    )
    assert cfg.stt.model == "small.en"
    assert cfg.stt.beam_size == 5
    assert cfg.audio.device == 2
    assert cfg.hotkey.key == "f12"
    assert cfg.inject.restore_clipboard_ms == 100


def test_from_dict_omitted_device_defaults_to_none():
    cfg = Config.from_dict({"audio": {}})
    assert cfg.audio.device is None


def test_from_dict_explicit_none_device_is_none():
    cfg = Config.from_dict({"audio": {"device": None}})
    assert cfg.audio.device is None


def test_with_overrides_does_not_mutate_original():
    cfg = Config()
    new = cfg.with_overrides(model="small.en", key="f12")
    assert new.stt.model == "small.en"
    assert new.hotkey.key == "f12"
    # Original is frozen dataclass: untouched.
    assert cfg.stt.model == "base.en"
    assert cfg.hotkey.key == "alt_r"


def test_load_config_writes_default_when_missing(tmp_path: Path):
    path = tmp_path / "config.toml"
    assert not path.exists()
    cfg = load_config(path)
    assert path.exists()
    assert cfg.stt.model == "base.en"
    assert cfg.hotkey.key == "alt_r"


def test_load_config_reads_existing(tmp_path: Path):
    path = tmp_path / "config.toml"
    write_default_config(path)
    cfg = load_config(path)
    assert cfg.stt.model == "base.en"


def test_write_default_config_does_not_overwrite(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text('[stt]\nmodel = "tiny.en"\n', encoding="utf-8")
    write_default_config(path)
    assert 'model = "tiny.en"' in path.read_text(encoding="utf-8")
