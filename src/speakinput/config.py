"""Configuration loading and first-run default write."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

APP_NAME = "speakinput"

# Curated models exposed by default. The `.en` variants are English-only and
# faster; the multilingual variants (`tiny`, `base`, `small`, `medium`) are
# slower per call but support Chinese and auto-detection. Any absolute path
# to a .bin file is also accepted at the CLI/ensure_model layer.
VALID_MODELS = (
    "tiny.en",
    "base.en",
    "small.en",
    "tiny",
    "base",
    "small",
    "medium",
)
# Languages that are always supported. `auto` translates to pywhispercpp's
# `language=None`, which triggers per-utterance language identification.
VALID_LANGUAGES = ("auto", "en", "zh")
VALID_HOTKEYS = ("alt_r", "ctrl_r", "cmd_r", "shift_r", "caps_lock", "f12")


# Model names that are English-only. Pairing one with a non-English language
# would be a misconfiguration; we surface it at validate() time.
_ENGLISH_ONLY_MODELS = frozenset({"tiny.en", "base.en", "small.en"})


@dataclass(frozen=True)
class STTConfig:
    model: str = "small"
    language: str = "auto"
    beam_size: int = 1


@dataclass(frozen=True)
class AudioConfig:
    device: int | None = None
    sample_rate: int = 16000
    # Audio whose RMS is below this floor is treated as silence and never
    # sent to the model. Stops whisper from hallucinating on near-empty
    # recordings (e.g. user pressed the hotkey by accident). 0 disables
    # the gate. Default 0.005 — quiet enough to swallow room noise,
    # loud enough to admit any actual speech.
    silence_threshold: float = 0.005


@dataclass(frozen=True)
class HotkeyConfig:
    key: str = "alt_r"


@dataclass(frozen=True)
class InjectConfig:
    restore_clipboard_ms: int = 50
    trailing_space: bool = True


@dataclass(frozen=True)
class Config:
    stt: STTConfig = field(default_factory=STTConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    inject: InjectConfig = field(default_factory=InjectConfig)

    @classmethod
    def from_toml(cls, path: Path) -> "Config":
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        stt = STTConfig(**data.get("stt", {}))
        audio_raw = dict(data.get("audio", {}))
        # `device` is intentionally optional in the TOML file (it has no null
        # literal). If absent, the dataclass default of None wins.
        if "device" in audio_raw and audio_raw["device"] is None:
            audio_raw["device"] = None
        audio = AudioConfig(**audio_raw)
        hotkey = HotkeyConfig(**data.get("hotkey", {}))
        inject = InjectConfig(**data.get("inject", {}))
        return cls(stt=stt, audio=audio, hotkey=hotkey, inject=inject)

    def validate(self) -> None:
        if self.stt.model not in VALID_MODELS:
            raise ValueError(f"stt.model must be one of {VALID_MODELS}, got {self.stt.model!r}")
        if self.stt.language not in VALID_LANGUAGES:
            raise ValueError(
                f"stt.language must be one of {VALID_LANGUAGES}, got {self.stt.language!r}"
            )
        if (
            self.stt.model in _ENGLISH_ONLY_MODELS
            and self.stt.language not in ("auto", "en")
        ):
            raise ValueError(
                f"model {self.stt.model!r} is English-only; "
                f"set stt.language to 'en' or 'auto', or pick a multilingual model."
            )
        if self.hotkey.key not in VALID_HOTKEYS:
            raise ValueError(f"hotkey.key must be one of {VALID_HOTKEYS}, got {self.hotkey.key!r}")
        if self.audio.sample_rate <= 0:
            raise ValueError("audio.sample_rate must be positive")
        if self.audio.silence_threshold < 0:
            raise ValueError("audio.silence_threshold must be >= 0 (0 disables)")
        if not 1 <= self.stt.beam_size <= 10:
            raise ValueError("stt.beam_size must be in [1, 10]")

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with select fields overridden (used by CLI flags)."""
        stt = replace(
            self.stt, **{k: v for k, v in overrides.items() if k in STTConfig.__annotations__}
        )
        audio = replace(
            self.audio, **{k: v for k, v in overrides.items() if k in AudioConfig.__annotations__}
        )
        hotkey = replace(
            self.hotkey, **{k: v for k, v in overrides.items() if k in HotkeyConfig.__annotations__}
        )
        inject = replace(
            self.inject, **{k: v for k, v in overrides.items() if k in InjectConfig.__annotations__}
        )
        return Config(stt=stt, audio=audio, hotkey=hotkey, inject=inject)


def default_config_path() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "config.toml"


def write_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    # Note: TOML has no null literal. Audio device is intentionally omitted;
    # `Config.from_dict` defaults it to None when missing.
    path.write_text(
        """[stt]
model = "small"
language = "auto"
beam_size = 1

[audio]
sample_rate = 16000
silence_threshold = 0.005

[hotkey]
key = "alt_r"

[inject]
restore_clipboard_ms = 50
trailing_space = true
""",
        encoding="utf-8",
    )


def load_config(path: Path | None = None, *, write_default: bool = True) -> Config:
    """Load config from `path`, falling back to the user config dir.

    If `write_default` is True and no file exists at the resolved path, a
    default config.toml is written so the user has a discoverable starting point.
    """
    resolved = path or default_config_path()
    if not resolved.exists():
        if write_default:
            write_default_config(resolved)
        else:
            raise FileNotFoundError(f"config not found: {resolved}")
    cfg = Config.from_toml(resolved)
    cfg.validate()
    return cfg
