from pathlib import Path

import pytest

from speakinput.config import (
    AudioConfig,
    Config,
    InjectConfig,
    Profile,
    load_config,
    primary_profile,
    secondary_profile,
)


# --- default construction --------------------------------------------------


def test_default_construction():
    cfg = Config()
    assert cfg.primary == primary_profile()
    assert cfg.secondary is None
    assert cfg.audio == AudioConfig()
    assert cfg.inject == InjectConfig()


def test_primary_profile_default_key_is_platform_aware():
    """primary_profile() picks the platform default at call time."""
    import sys as _sys

    from speakinput.config import _default_primary_key

    assert primary_profile().key == _default_primary_key()


def test_secondary_profile_default_key_is_cmd_r():
    """secondary_profile() defaults to cmd_r on every platform."""
    assert secondary_profile().key == "cmd_r"


def test_secondary_profile_default_language_is_zh():
    """Secondary is for the non-primary language by default (zh)."""
    assert secondary_profile().language == "zh"


# --- from_dict: required primary ------------------------------------------


def test_from_dict_reads_trailing_space():
    cfg = Config.from_dict({"inject": {"trailing_space": False}})
    assert cfg.inject.trailing_space is False


def test_with_overrides_trailing_space():
    cfg = Config()
    new = cfg.with_overrides(trailing_space=False)
    assert new.inject.trailing_space is False
    assert cfg.inject.trailing_space is True  # original untouched


def test_from_dict_overrides_sections():
    cfg = Config.from_dict(
        {
            "profile": {
                "primary": {
                    "key": "f12",
                    "model": "small.en",
                    "language": "en",
                    "beam_size": 5,
                }
            },
            "audio": {"device": 2, "sample_rate": 16000},
            "inject": {"restore_clipboard_ms": 100},
        }
    )
    assert cfg.primary.key == "f12"
    assert cfg.primary.model == "small.en"
    assert cfg.primary.beam_size == 5
    assert cfg.audio.device == 2
    assert cfg.inject.restore_clipboard_ms == 100


def test_from_dict_omitted_device_defaults_to_none():
    cfg = Config.from_dict({"audio": {}})
    assert cfg.audio.device is None


def test_from_dict_explicit_none_device_is_none():
    cfg = Config.from_dict({"audio": {"device": None}})
    assert cfg.audio.device is None


def test_from_dict_partial_primary_uses_defaults_for_missing_fields():
    """When the user only sets `key = "f12"` in [profile.primary], the
    other fields still take their default values."""
    cfg = Config.from_dict({"profile": {"primary": {"key": "f12"}}})
    assert cfg.primary.key == "f12"
    assert cfg.primary.model == "small"
    assert cfg.primary.language == "auto"
    assert cfg.primary.beam_size == 1
    assert cfg.primary.initial_prompt  # non-empty default


def test_from_dict_partial_secondary_uses_defaults_for_missing_fields():
    """Same for [profile.secondary]: missing fields default sensibly."""
    cfg = Config.from_dict({"profile": {"secondary": {"language": "en"}}})
    assert cfg.secondary is not None
    assert cfg.secondary.key == "cmd_r"
    assert cfg.secondary.model == "small"
    assert cfg.secondary.language == "en"
    assert cfg.secondary.beam_size == 1


def test_from_dict_missing_secondary_section_disables_secondary():
    """To disable the secondary profile, the user omits the
    `[profile.secondary]` block entirely. With no `secondary` key in
    the `[profile]` table, `cfg.secondary` is None and only the
    primary key is wired."""
    cfg = Config.from_dict({"profile": {"primary": {}}})
    assert cfg.secondary is None


def test_from_dict_explicit_empty_secondary_section_uses_defaults():
    """An empty `[profile.secondary]` table (no fields inside) means
    'use all the default secondary values', NOT 'disable the
    secondary profile'. To disable, omit the section entirely. This
    is the same convention as [audio] / [inject]: an empty section
    is a request for defaults, not a no-op."""
    cfg = Config.from_dict({"profile": {"secondary": {}}})
    assert cfg.secondary is not None
    assert cfg.secondary.key == "cmd_r"
    assert cfg.secondary.language == "zh"


# --- with_overrides --------------------------------------------------------


def test_with_overrides_applies_to_primary():
    """CLI flags like -m / -g / -P flow to the primary profile."""
    cfg = Config()
    new = cfg.with_overrides(model="base.en", language="en")
    assert new.primary.model == "base.en"
    assert new.primary.language == "en"
    # Original is frozen: untouched.
    assert cfg.primary.model == "small"
    assert cfg.primary.language == "auto"


def test_with_overrides_does_not_mutate_original():
    cfg = Config()
    new = cfg.with_overrides(model="base.en", key="f12")
    assert new.primary.model == "base.en"
    assert new.primary.key == "f12"
    # The default hotkey is platform-aware.
    from speakinput.config import _default_primary_key
    assert cfg.primary.key == _default_primary_key()


def test_with_overrides_initial_prompt():
    """-P / --initial-prompt must reach the primary profile."""
    cfg = Config()
    new = cfg.with_overrides(initial_prompt="kubectl apply -f deployment.yaml")
    assert new.primary.initial_prompt == "kubectl apply -f deployment.yaml"
    assert cfg.primary.initial_prompt != new.primary.initial_prompt


def test_with_overrides_initial_prompt_clear_to_empty():
    cfg = Config()
    new = cfg.with_overrides(initial_prompt="")
    assert new.primary.initial_prompt == ""


def test_with_overrides_silence_threshold():
    cfg = Config()
    new = cfg.with_overrides(silence_threshold=0.02)
    assert new.audio.silence_threshold == 0.02
    assert cfg.audio.silence_threshold == 0.005


# --- load_config ------------------------------------------------------------


def test_load_config_returns_defaults_when_missing(tmp_path: Path):
    """No file at the path? Return (Config(), None). The default
    secondary is None — single-profile mode is the no-config fallback."""
    path = tmp_path / "config.toml"
    assert not path.exists()
    cfg, source = load_config(path)
    assert cfg.primary.model == "small"
    assert cfg.primary.language == "auto"
    assert cfg.secondary is None
    from speakinput.config import _default_primary_key
    assert cfg.primary.key == _default_primary_key()
    assert source is None
    # load_config never touches the filesystem.
    assert not path.exists()


def test_load_config_reads_primary(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[profile.primary]\nmodel = "tiny.en"\nlanguage = "en"\n',
        encoding="utf-8",
    )
    cfg, source = load_config(path)
    assert cfg.primary.model == "tiny.en"
    assert cfg.primary.language == "en"
    assert cfg.secondary is None
    assert source == path


def test_load_config_reads_secondary(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[profile.primary]\nkey = "alt_r"\n'
        '[profile.secondary]\nkey = "cmd_r"\nlanguage = "zh"\n',
        encoding="utf-8",
    )
    cfg, source = load_config(path)
    assert cfg.primary.key == "alt_r"
    assert cfg.secondary is not None
    assert cfg.secondary.key == "cmd_r"
    assert cfg.secondary.language == "zh"
    assert source == path


def test_load_config_explicit_none_path_uses_default(tmp_path: Path, monkeypatch):
    target = tmp_path / "explicit_default.toml"
    target.write_text('[profile.primary]\nmodel = "medium"\n', encoding="utf-8")
    monkeypatch.setattr(
        "speakinput.config.default_config_path", lambda: target
    )
    cfg, source = load_config()
    assert cfg.primary.model == "medium"
    assert source == target


# --- platform-aware primary hotkey ----------------------------------------


def test_default_primary_key_on_macos_is_alt_r(monkeypatch):
    """On Darwin, Right Option (`alt_r`) is the canonical push-to-talk key."""
    import sys as _sys

    from speakinput.config import _default_primary_key

    monkeypatch.setattr(_sys, "platform", "darwin")
    assert _default_primary_key() == "alt_r"


def test_default_primary_key_on_linux_is_ctrl_r(monkeypatch):
    import sys as _sys

    from speakinput.config import _default_primary_key

    monkeypatch.setattr(_sys, "platform", "linux")
    assert _default_primary_key() == "ctrl_r"
    monkeypatch.setattr(_sys, "platform", "win32")
    assert _default_primary_key() == "ctrl_r"


def test_primary_default_hotkey_follows_platform(monkeypatch):
    import sys as _sys

    monkeypatch.setattr(_sys, "platform", "darwin")
    assert primary_profile().key == "alt_r"
    assert Config().primary.key == "alt_r"

    monkeypatch.setattr(_sys, "platform", "linux")
    assert primary_profile().key == "ctrl_r"
    assert Config().primary.key == "ctrl_r"


def test_explicit_primary_hotkey_in_config_overrides_platform_default(tmp_path: Path):
    """A user who pinned `key = "alt_r"` in their config.toml on Linux
    should still get alt_r — their explicit choice wins over the
    platform default."""
    path = tmp_path / "config.toml"
    path.write_text('[profile.primary]\nkey = "alt_r"\n', encoding="utf-8")
    cfg, _ = load_config(path)
    assert cfg.primary.key == "alt_r"


# --- multilingual / Chinese support ---------------------------------------


def test_default_model_is_multilingual():
    """Shipped default is `small` (multilingual) so first-run Chinese
    works out of the box without any config editing."""
    assert Config().primary.model == "small"
    assert Config().primary.language == "auto"


def test_validation_accepts_multilingual_models():
    for m in ("tiny", "base", "small", "medium"):
        Config(primary=Profile(model=m, language="zh")).validate()


def test_validation_accepts_auto_language():
    Config(primary=Profile(model="base", language="auto")).validate()
    Config(primary=Profile(model="small.en", language="auto")).validate()


def test_validation_rejects_unknown_model():
    with pytest.raises(ValueError, match="primary.model"):
        Config(primary=Profile(model="bogus")).validate()


def test_validation_rejects_unknown_hotkey():
    with pytest.raises(ValueError, match="primary.key"):
        Config(primary=Profile(key="super")).validate()


def test_validation_rejects_zero_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        Config(audio=AudioConfig(sample_rate=0)).validate()


def test_validation_rejects_out_of_range_beam_size():
    with pytest.raises(ValueError, match="beam_size"):
        Config(primary=Profile(beam_size=0)).validate()
    with pytest.raises(ValueError, match="beam_size"):
        Config(primary=Profile(beam_size=11)).validate()


def test_validation_rejects_unknown_language():
    with pytest.raises(ValueError, match="primary.language"):
        Config(primary=Profile(model="base", language="fr")).validate()


def test_validation_rejects_english_only_model_with_chinese():
    with pytest.raises(ValueError, match="English-only"):
        Config(primary=Profile(model="base.en", language="zh")).validate()


def test_validation_allows_english_only_model_with_en_or_auto():
    Config(primary=Profile(model="base.en", language="en")).validate()
    Config(primary=Profile(model="base.en", language="auto")).validate()


def test_validation_runs_for_secondary_profile():
    """A bad model in the secondary profile must also be caught."""
    with pytest.raises(ValueError, match="secondary.model"):
        Config(
            primary=Profile(),
            secondary=Profile(model="bogus", language="zh"),
        ).validate()


def test_validation_rejects_same_key_in_both_profiles():
    """Two profiles sharing a key is a configuration error — there's no
    way to dispatch a press to the right one."""
    with pytest.raises(ValueError, match="same key"):
        Config(
            primary=Profile(key="alt_r"),
            secondary=Profile(key="alt_r"),
        ).validate()


# --- CLI arg surface -------------------------------------------------------


def test_cli_language_argument_is_parsed():
    from speakinput.cli import _build_parser

    args = _build_parser().parse_args(["--language", "zh"])
    assert args.language == "zh"


def test_cli_language_short_flag():
    from speakinput.cli import _build_parser

    args = _build_parser().parse_args(["-g", "en"])
    assert args.language == "en"


def test_cli_language_defaults_to_none():
    from speakinput.cli import _build_parser

    args = _build_parser().parse_args([])
    assert args.language is None


def test_cli_rejects_unknown_language():
    from speakinput.cli import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--language", "fr"])


def test_cli_initial_prompt_short_flag():
    from speakinput.cli import _build_parser

    args = _build_parser().parse_args(["-P", "kubectl"])
    assert args.initial_prompt == "kubectl"


def test_cli_initial_prompt_long_flag():
    from speakinput.cli import _build_parser

    args = _build_parser().parse_args(["--initial-prompt", "K8s, SRE"])
    assert args.initial_prompt == "K8s, SRE"


def test_cli_initial_prompt_defaults_to_none():
    from speakinput.cli import _build_parser

    args = _build_parser().parse_args([])
    assert args.initial_prompt is None


def test_cli_initial_prompt_with_spaces():
    from speakinput.cli import _build_parser

    args = _build_parser().parse_args(["-P", "kubectl apply -f deployment.yaml"])
    assert args.initial_prompt == "kubectl apply -f deployment.yaml"


def test_cli_silence_threshold_short_flag():
    from speakinput.cli import _build_parser

    args = _build_parser().parse_args(["-S", "0.02"])
    assert args.silence_threshold == 0.02


def test_cli_silence_threshold_long_flag():
    from speakinput.cli import _build_parser

    args = _build_parser().parse_args(["--silence-threshold", "0"])
    assert args.silence_threshold == 0


def test_cli_silence_threshold_defaults_to_none():
    from speakinput.cli import _build_parser

    args = _build_parser().parse_args([])
    assert args.silence_threshold is None


# --- silence threshold ----------------------------------------------------


def test_default_silence_threshold():
    assert Config().audio.silence_threshold == 0.005


# --- initial_prompt default ------------------------------------------------


def test_default_initial_prompt_biases_toward_embedded_vocabulary():
    """The shipped default biases whisper toward embedded software engineer
    vocabulary. Disable by setting initial_prompt = "" in config.toml."""
    prompt = Config().primary.initial_prompt
    assert prompt  # non-empty
    for token in ("FreeRTOS", "GPIO", "ISR", "OpenOCD", "DMA", "RTOS", "PWM", "mutex"):
        assert token in prompt


def test_secondary_profile_has_same_default_prompt():
    """Both profiles start with the embedded-vocab bias; users override
    per profile as needed."""
    assert secondary_profile().initial_prompt == primary_profile().initial_prompt
