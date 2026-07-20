"""Configuration loading.

The program ships with `config.example.toml` in the project root as
documentation. The user copies that to their user config dir on first
run (or `start.sh` does it for them) and edits as needed. If the file
is missing, the program uses hard-coded defaults — every value in
`config.example.toml` is the same as the dataclass default.

A `Config` holds two profiles (primary and optional secondary). Each
profile binds one hotkey to one STT setup (model + language + prompt).
A typical setup is alt_r for English and cmd_r for Chinese, but both
keys and both languages are user-configurable. The secondary profile
is optional; when omitted only the primary key is wired.
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


# --- embedded-software-engineer lexical bias -------------------------------
# Whisper's initial_prompt is tokenized once and used to bias the decoder at
# the start of every transcription. The default below biases toward common
# embedded firmware vocabulary: C/C++, RTOS terms, MCU peripherals, debug
# tools, and common acronyms. No product names (STM32, ESP32) and no `_t`
# type suffixes — those would pollute the BPE tokenizer with tokens the user
# is unlikely to actually say. Override per-profile or set to "" to disable.
_EMBEDDED_PROMPT = (
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


def _default_primary_key() -> str:
    """Pick a hotkey that fits the platform's keyboard conventions.

    On macOS, Right Option (`alt_r`) is the canonical push-to-talk key —
    most users have it free, it sits under the right thumb, and it
    doesn't conflict with the common Cmd-based shortcuts.

    On Linux/Windows, Alt is heavily used for menu mnemonics in many
    desktop apps, so we default to Right Ctrl (`ctrl_r`) instead. PC
    keyboards also tend to have a larger Right Ctrl than Mac keyboards
    have Right Option, making it easier to find by feel.

    Override in config.toml with [profile.primary] key = "...".
    """
    if sys.platform == "darwin":
        return "alt_r"
    return "ctrl_r"


def _default_secondary_key() -> str:
    """Default key for the second language profile.

    The user asked for Right Cmd (`cmd_r`) on macOS. On Linux/Windows
    there is no Right Cmd key on most keyboards, so we still default to
    `cmd_r` — `pynput` maps it to the Super key on those platforms, which
    is a reasonable "second modifier under the thumb" position. Override
    in config.toml with [profile.secondary] key = "...".
    """
    return "cmd_r"


@dataclass(frozen=True)
class Profile:
    """One hotkey + one STT setup.

    A profile pairs a key with everything needed to transcribe speech
    captured while that key is held: the model, the language hint, beam
    size, and an initial_prompt. Two profiles can share the same model
    file (the App caches transcribers by model path) — the typical
    English/Chinese setup uses `small` for both.

    Construct via the `primary_profile()` and `secondary_profile()`
    factory functions below — those set the right default key. The
    dataclass defaults are placeholders that are overridden in
    `Config.from_dict` and the factory functions.
    """

    key: str = "alt_r"
    model: str = "small"
    language: str = "auto"
    beam_size: int = 1
    initial_prompt: str = _EMBEDDED_PROMPT


def primary_profile() -> Profile:
    """Default primary profile: platform-aware hotkey, multilingual small,
    language=auto, embedded-vocab prompt."""
    return Profile(key=_default_primary_key())


def secondary_profile() -> Profile:
    """Default secondary profile: cmd_r (or Super on Linux/Windows),
    multilingual small, language=zh, embedded-vocab prompt."""
    return Profile(
        key=_default_secondary_key(),
        language="zh",
    )


@dataclass(frozen=True)
class AudioConfig:
    device: int | None = None
    sample_rate: int = 16000
    # Audio whose RMS is below this floor is treated as silence. Two
    # related things use it:
    #   1. The pre-transcribe gate: if the whole buffer's RMS is below
    #      this, skip whisper entirely. Stops whisper from hallucinating
    #      on near-empty recordings (e.g. user pressed the hotkey by
    #      accident). 0 disables.
    #   2. The auto-stop watchdog: while the key is held, if `auto_stop_seconds`
    #      of sub-threshold audio passes in a row, the watchdog
    #      synthesizes a release so the user doesn't have to time it
    #      themselves. Trailing silence is also trimmed from the
    #      buffer before transcribe.
    silence_threshold: float = 0.005
    # Seconds of consecutive sub-threshold audio that triggers an
    # auto-stop while the key is held. 0 disables the watchdog entirely
    # (the old "release the key yourself" behavior). Default 0 — by
    # default the user releases the key themselves; raise it to a
    # typical end-of-sentence pause (e.g. 0.8) to enable the chunked
    # auto-stop behavior.
    auto_stop_seconds: float = 0
    # Window in seconds during which the most recent transcription is
    # reused as a continuity hint for the NEXT press. When a new press
    # starts and the previous clip's text is younger than this window,
    # the program appends it to `initial_prompt` so whisper doesn't
    # "reset" the topic between sentences.
    #
    # Two layers of context feed the prompt at transcribe time:
    #   1. The per-profile `initial_prompt` (lexical bias — vocabulary
    #      names, jargon, style hints). Static; never changes.
    #   2. The most recent transcript from a previous press, when its
    #      age is under this window. Across-press continuity only.
    #   3. Within a single press, the previous auto-stopped chunk's
    #      text is ALWAYS used — that is not gated by this window.
    #
    # The order in the final prompt is (1) + (2) + (3), so the static
    # vocabulary bias wins, then the across-press topic, then the
    # within-press sentence-level context. 0 disables the across-press
    # hint entirely (within-press still flows).
    prev_clip_window_seconds: float = 60.0
    # Pause any playing media (music, video) when the hotkey is pressed
    # and resume it when the key is released. On Linux requires
    # `playerctl` (MPRIS); on macOS uses osascript for Spotify/Music;
    # on Windows uses PowerShell SMTC. No-op when no backend is found.
    pause_media: bool = True


@dataclass(frozen=True)
class InjectConfig:
    restore_clipboard_ms: int = 50
    trailing_space: bool = True
    # Output backend. One of:
    #   "auto"    — platform-driven selection (default)
    #   "pynput"  — pynput (macOS / Windows / X11 Linux)
    #   "wtype"   — wtype (wlroots Wayland, daemon-free)
    #   "ydotool" — ydotool (Wayland fallback, needs ydotoold running)
    # `auto` resolves at App construction time: macOS/Windows→pynput;
    # Linux+Wayland→wtype→ydotool→pynput; Linux+X11→pynput.
    backend: str = "auto"


VALID_INJECT_BACKENDS = ("auto", "pynput", "wtype", "ydotool")


@dataclass(frozen=True)
class TranscribeConfig:
    # GPU use. `None` means "auto-detect from the loaded pywhispercpp
    # wheel" — use GPU if the wheel was built with a GPU backend
    # (CUDA/Vulkan/Metal), else stay on CPU. `True` forces GPU on
    # (with a stderr warning if the wheel is CPU-only). `False`
    # forces CPU. The TOML "auto" string is normalized to `None`
    # at load time.
    use_gpu: bool | None = None
    gpu_device: int = 0
    # Number of CPU threads for the CPU path. 0 = pywhispercpp's
    # default (min(4, hardware_concurrency())). Bump this on big
    # x86 boxes (16+ cores) where the default leaves cores idle.
    n_threads: int = 0


@dataclass(frozen=True)
class Config:
    primary: Profile = field(default_factory=primary_profile)
    secondary: Profile | None = None
    audio: AudioConfig = field(default_factory=AudioConfig)
    inject: InjectConfig = field(default_factory=InjectConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)

    @classmethod
    def from_toml(cls, path: Path) -> "Config":
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        profiles_raw = data.get("profile", {})
        primary_raw = profiles_raw.get("primary", {})
        primary = Profile(
            key=primary_raw.get("key", _default_primary_key()),
            model=primary_raw.get("model", "small"),
            language=primary_raw.get("language", "auto"),
            beam_size=primary_raw.get("beam_size", 1),
            initial_prompt=primary_raw.get("initial_prompt", _EMBEDDED_PROMPT),
        )
        secondary_raw = profiles_raw.get("secondary")
        if secondary_raw is None:
            secondary = None
        else:
            secondary = Profile(
                key=secondary_raw.get("key", _default_secondary_key()),
                model=secondary_raw.get("model", "small"),
                language=secondary_raw.get("language", "zh"),
                beam_size=secondary_raw.get("beam_size", 1),
                initial_prompt=secondary_raw.get("initial_prompt", _EMBEDDED_PROMPT),
            )
        audio_raw = dict(data.get("audio", {}))
        # `device` is intentionally optional in the TOML file (it has no null
        # literal). If absent, the dataclass default of None wins.
        if "device" in audio_raw and audio_raw["device"] is None:
            audio_raw["device"] = None
        audio = AudioConfig(**audio_raw)
        inject = InjectConfig(**data.get("inject", {}))
        transcribe_raw = dict(data.get("transcribe", {}))
        # TOML has no `null` literal in this config; the string "auto"
        # is the user-facing way to say "let the app decide" and maps
        # to the Python `None` auto-detect sentinel.
        if isinstance(transcribe_raw.get("use_gpu"), str):
            ug = transcribe_raw["use_gpu"].strip().lower()
            if ug in ("auto", ""):
                transcribe_raw["use_gpu"] = None
            elif ug in ("true", "yes", "1"):
                transcribe_raw["use_gpu"] = True
            elif ug in ("false", "no", "0"):
                transcribe_raw["use_gpu"] = False
            # else: leave the bool as-is; validate() will catch garbage
        transcribe = TranscribeConfig(**transcribe_raw)
        return cls(
            primary=primary,
            secondary=secondary,
            audio=audio,
            inject=inject,
            transcribe=transcribe,
        )

    def validate(self) -> None:
        for label, profile in (("primary", self.primary), ("secondary", self.secondary)):
            if profile is None:
                continue
            if profile.model not in VALID_MODELS:
                raise ValueError(
                    f"profile.{label}.model must be one of {VALID_MODELS}, "
                    f"got {profile.model!r}"
                )
            if profile.language not in VALID_LANGUAGES:
                raise ValueError(
                    f"profile.{label}.language must be one of {VALID_LANGUAGES}, "
                    f"got {profile.language!r}"
                )
            if (
                profile.model in _ENGLISH_ONLY_MODELS
                and profile.language not in ("auto", "en")
            ):
                raise ValueError(
                    f"profile.{label}: model {profile.model!r} is English-only; "
                    f"set language to 'en' or 'auto', or pick a multilingual model."
                )
            if profile.key not in VALID_HOTKEYS:
                raise ValueError(
                    f"profile.{label}.key must be one of {VALID_HOTKEYS}, "
                    f"got {profile.key!r}"
                )
            if not 1 <= profile.beam_size <= 10:
                raise ValueError(
                    f"profile.{label}.beam_size must be in [1, 10], "
                    f"got {profile.beam_size}"
                )
        if (
            self.secondary is not None
            and self.primary.key == self.secondary.key
        ):
            raise ValueError(
                f"primary and secondary profiles share the same key "
                f"({self.primary.key!r}); pick distinct keys"
            )
        if self.audio.sample_rate <= 0:
            raise ValueError("audio.sample_rate must be positive")
        if self.audio.silence_threshold < 0:
            raise ValueError("audio.silence_threshold must be >= 0 (0 disables)")
        if self.audio.auto_stop_seconds < 0:
            raise ValueError("audio.auto_stop_seconds must be >= 0 (0 disables)")
        if self.audio.prev_clip_window_seconds < 0:
            raise ValueError(
                "audio.prev_clip_window_seconds must be >= 0 (0 disables across-press hint)"
            )
        if self.inject.backend not in VALID_INJECT_BACKENDS:
            raise ValueError(
                f"inject.backend must be one of {VALID_INJECT_BACKENDS}, "
                f"got {self.inject.backend!r}"
            )
        if not isinstance(self.transcribe.use_gpu, (bool, type(None))):
            raise ValueError(
                f"transcribe.use_gpu must be a bool or 'auto', "
                f"got {self.transcribe.use_gpu!r}"
            )
        if self.transcribe.gpu_device < 0:
            raise ValueError(
                f"transcribe.gpu_device must be >= 0, "
                f"got {self.transcribe.gpu_device}"
            )
        if self.transcribe.n_threads < 0:
            raise ValueError(
                f"transcribe.n_threads must be >= 0 (0 = auto), "
                f"got {self.transcribe.n_threads}"
            )

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with select fields overridden (used by CLI flags).

        All overrides apply to the primary profile only. The secondary
        profile is meant to be configured via config.toml; the CLI does
        not expose per-profile flags. Pass `secondary=...` to replace
        the entire secondary profile object, or `secondary=None` to
        disable it.
        """
        primary_keys = set(Profile.__annotations__)
        if any(k in primary_keys for k in overrides):
            primary = replace(
                self.primary,
                **{k: v for k, v in overrides.items() if k in primary_keys},
            )
            overrides = {k: v for k, v in overrides.items() if k not in primary_keys}
        else:
            primary = self.primary
        audio_keys = set(AudioConfig.__annotations__)
        if any(k in audio_keys for k in overrides):
            audio = replace(
                self.audio,
                **{k: v for k, v in overrides.items() if k in audio_keys},
            )
            overrides = {k: v for k, v in overrides.items() if k not in audio_keys}
        else:
            audio = self.audio
        inject_keys = set(InjectConfig.__annotations__)
        if any(k in inject_keys for k in overrides):
            inject = replace(
                self.inject,
                **{k: v for k, v in overrides.items() if k in inject_keys},
            )
            overrides = {k: v for k, v in overrides.items() if k not in inject_keys}
        else:
            inject = self.inject
        transcribe_keys = set(TranscribeConfig.__annotations__)
        if any(k in transcribe_keys for k in overrides):
            transcribe = replace(
                self.transcribe,
                **{k: v for k, v in overrides.items() if k in transcribe_keys},
            )
            overrides = {k: v for k, v in overrides.items() if k not in transcribe_keys}
        else:
            transcribe = self.transcribe
        secondary = overrides.pop("secondary", self.secondary)
        return Config(
            primary=primary,
            secondary=secondary,
            audio=audio,
            inject=inject,
            transcribe=transcribe,
        )


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
