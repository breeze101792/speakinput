"""Configuration loading.

The program ships with `config.example.toml` in the project root as
documentation. The user copies that to their user config dir on first
run (or `start.sh` does it for them) and edits as needed. If the file
is missing, the program uses hard-coded defaults — every value in
`config.example.toml` is the same as the dataclass default.
"""

from __future__ import annotations

import sys
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


def _default_hotkey() -> str:
    """Pick a hotkey that fits the platform's keyboard conventions.

    On macOS, Right Option (`alt_r`) is the canonical push-to-talk key —
    most users have it free, it sits under the right thumb, and it
    doesn't conflict with the common Cmd-based shortcuts.

    On Linux/Windows, Alt is heavily used for menu mnemonics in many
    desktop apps, so we default to Right Ctrl (`ctrl_r`) instead. PC
    keyboards also tend to have a larger Right Ctrl than Mac keyboards
    have Right Option, making it easier to find by feel.

    Override in config.toml with [hotkey].key = "...".
    """
    if sys.platform == "darwin":
        return "alt_r"
    return "ctrl_r"


@dataclass(frozen=True)
class STTConfig:
    model: str = "small"
    language: str = "auto"
    beam_size: int = 1
    # Whisper's initial_prompt is a lexical prior — tokenized once and used
    # to bias the decoder at the start of every transcription. Useful for
    # names, technical jargon, or acronyms the base model would misspell
    # (e.g. "kubectl apply -f deployment.yaml" biases the decoder toward
    # those tokens). The default biases toward embedded software engineer
    # vocabulary: C/C++, RTOS terms, MCU peripherals, debug tools, and
    # common acronyms. Can be overridden per-run via `-P`/`--initial-prompt`
    # on the CLI, or set to "" in config.toml to disable the bias.
    initial_prompt: str = (
        # Languages + RTOSes
        "C, C++, Rust, assembly, embedded, firmware, kernel, driver, "
        "RTOS, FreeRTOS, Zephyr, syscall, callback, "
        # MCU / SoC (no product names)
        "microcontroller, ARM Cortex, RISC-V, register, peripheral, "
        "clock, PLL, prescaler, watchdog, "
        # Memory
        "ROM, RAM, flash, EEPROM, heap, stack, allocator, "
        # Interrupts / concurrency
        "interrupt, ISR, IRQ, exception, fault, hardfault, "
        "scheduler, mutex, semaphore, spinlock, atomic, preemptive, "
        # Communication peripherals (no product names)
        "GPIO, UART, SPI, I2C, I2S, CAN, USB, Ethernet, "
        "ADC, DAC, PWM, timer, DMA, FIFO, "
        # Protocols / data concepts
        "ack, nack, CRC, checksum, parity, packet, frame, payload, "
        "endian, alignment, bitfield, "
        # Storage
        "block, sector, page, erase, program, mount, "
        # Build / toolchain
        "gcc, clang, cmake, ninja, linker, cross-compile, "
        "elf, hex, optimization, "
        # Debug / analysis
        "GDB, OpenOCD, JTAG, SWD, trace, profiler, "
        "core dump, stack trace, backtrace, "
        # Language idioms
        "volatile, const, static, inline, extern, weak, packed, "
        "typedef, struct, union, enum, macro, pragma, "
        "void, NULL, nullptr, true, false, "
        "printf, sprintf, malloc, free, memcpy, memset, strlen"
    )


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
    # Default is platform-aware (see _default_hotkey). Use a
    # default_factory rather than a literal so the value is resolved
    # at instantiation time, not at class-definition time.
    key: str = field(default_factory=_default_hotkey)


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


def load_config(path: Path | None = None) -> tuple["Config", Path | None]:
    """Load config from `path`, falling back to the user config dir.

    Returns ``(config, source_path)``. If the resolved file is missing,
    returns ``(Config(), None)`` — the dataclass defaults. The program
    does not auto-write a config file; the user runs `./start.sh` (which
    copies the example on first run) or
    `cp config.example.toml ~/.config/speakinput/config.toml`.

    The returned `source_path` is non-None when a user-edited file was
    read, None when the program is running on its baked-in defaults.
    """
    resolved = path or default_config_path()
    if not resolved.exists():
        return Config(), None
    cfg = Config.from_toml(resolved)
    cfg.validate()
    return cfg, resolved

