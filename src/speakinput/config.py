"""Configuration loading and first-run default write."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

APP_NAME = "speakinput"

VALID_MODELS = ("tiny.en", "base.en", "small.en")
VALID_HOTKEYS = ("alt_r", "ctrl_r", "cmd_r", "shift_r", "caps_lock", "f12")


@dataclass(frozen=True)
class STTConfig:
    model: str = "base.en"
    language: str = "en"
    beam_size: int = 1


@dataclass(frozen=True)
class AudioConfig:
    device: int | None = None
    sample_rate: int = 16000


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
        if self.hotkey.key not in VALID_HOTKEYS:
            raise ValueError(f"hotkey.key must be one of {VALID_HOTKEYS}, got {self.hotkey.key!r}")
        if self.audio.sample_rate <= 0:
            raise ValueError("audio.sample_rate must be positive")
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
model = "base.en"
language = "en"
beam_size = 1

[audio]
sample_rate = 16000

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
