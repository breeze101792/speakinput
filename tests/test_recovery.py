"""Tests for the auto-recovery system: transcriber crash recovery,
microphone failover, injector fallback chain, and configurable
recovery settings.

These tests are platform-agnostic — they mock the hardware/engine
layers and verify the recovery logic itself. Platform-specific
behavior (macOS CGEventTap, Linux evdev) is tested in test_hotkey.py
and test_app.py with platform markers.

Markers: pytest.mark.recovery
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from speakinput.config import AudioConfig, Config, RecoveryConfig

pytestmark = pytest.mark.recovery


@pytest.fixture
def fake_sd(monkeypatch):
    """Install a fake sounddevice for the mic failover tests."""
    from speakinput import audio as audio_mod

    fake = MagicMock()
    fake.query_devices.return_value = []
    fake.InputStream = MagicMock()
    monkeypatch.setattr(audio_mod, "sd", fake, raising=False)
    return fake


# --- RecoveryConfig defaults -----------------------------------------------


def test_recovery_config_defaults():
    """The default RecoveryConfig should enable all recovery paths."""
    rc = RecoveryConfig()
    assert rc.transcribe_retries == 2
    assert rc.transcribe_backoff_s == 0.5
    assert rc.mic_failover_scan is True
    assert rc.mic_open_retries == 2
    assert rc.injector_fallback is True
    assert rc.listener_restart_min_interval_s == 60.0


def test_config_has_recovery_section():
    """Config() must include a recovery section by default."""
    cfg = Config()
    assert isinstance(cfg.recovery, RecoveryConfig)


def test_recovery_config_from_toml():
    """The [recovery] section in TOML must be parsed correctly."""
    cfg = Config.from_dict({
        "recovery": {
            "transcribe_retries": 5,
            "transcribe_backoff_s": 1.0,
            "mic_failover_scan": False,
            "mic_open_retries": 0,
            "injector_fallback": False,
            "listener_restart_min_interval_s": 30.0,
        }
    })
    assert cfg.recovery.transcribe_retries == 5
    assert cfg.recovery.transcribe_backoff_s == 1.0
    assert cfg.recovery.mic_failover_scan is False
    assert cfg.recovery.mic_open_retries == 0
    assert cfg.recovery.injector_fallback is False
    assert cfg.recovery.listener_restart_min_interval_s == 30.0


def test_recovery_config_validation_rejects_negative_retries():
    with pytest.raises(ValueError, match="transcribe_retries"):
        Config(recovery=RecoveryConfig(transcribe_retries=-1)).validate()


def test_recovery_config_validation_rejects_negative_backoff():
    with pytest.raises(ValueError, match="transcribe_backoff_s"):
        Config(recovery=RecoveryConfig(transcribe_backoff_s=-0.1)).validate()


def test_recovery_config_validation_rejects_negative_mic_retries():
    with pytest.raises(ValueError, match="mic_open_retries"):
        Config(recovery=RecoveryConfig(mic_open_retries=-1)).validate()


def test_recovery_config_validation_rejects_negative_listener_interval():
    with pytest.raises(ValueError, match="listener_restart_min_interval"):
        Config(
            recovery=RecoveryConfig(listener_restart_min_interval_s=-1)
        ).validate()


def test_with_overrides_applies_to_recovery():
    cfg = Config()
    new = cfg.with_overrides(transcribe_retries=5)
    assert new.recovery.transcribe_retries == 5
    assert cfg.recovery.transcribe_retries == 2  # original untouched


# --- Transcriber crash recovery --------------------------------------------


def _build_app_for_recovery(debug=False, **recovery_overrides):
    """Build an App with mocked I/O and configurable recovery settings."""
    from speakinput.app import App

    recovery = RecoveryConfig(**recovery_overrides) if recovery_overrides else RecoveryConfig()
    config = Config(
        audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0),
        recovery=recovery,
    )
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(16000, 0.3, dtype=np.float32)

    transcriber = MagicMock()
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
        debug=debug,
    )
    return app, transcriber, recorder


def test_transcribe_crash_triggers_reload_and_retry(monkeypatch):
    """When transcribe() raises, the App should reload the model and
    retry. The second attempt should succeed."""
    app, transcriber, _ = _build_app_for_recovery()

    # First call raises, second succeeds.
    transcriber.transcribe.side_effect = [RuntimeError("model crashed"), "recovered text"]

    # Mock the reload to return a fresh mock that succeeds.
    new_transcriber = MagicMock()
    new_transcriber.transcribe.return_value = "recovered text"
    reload_mock = MagicMock(return_value=new_transcriber)
    monkeypatch.setattr(app, "_reload_transcriber", reload_mock)

    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)

    # The reload was called once.
    reload_mock.assert_called_once()
    # The new transcriber was used for the retry.
    new_transcriber.transcribe.assert_called_once()
    # The text was injected.
    app.injector.inject.assert_called_once_with("recovered text")


def test_transcribe_crash_exhausted_retries_logs_and_drops(monkeypatch, capsys):
    """When all retries are exhausted, the audio is dropped and the
    error is logged."""
    app, transcriber, _ = _build_app_for_recovery(transcribe_retries=1)

    transcriber.transcribe.side_effect = RuntimeError("persistent crash")
    monkeypatch.setattr(app, "_reload_transcriber", lambda profile: MagicMock(
        transcribe=MagicMock(side_effect=RuntimeError("still crashing"))
    ))

    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)

    # No text was injected.
    app.injector.inject.assert_not_called()
    # The lock was released so the next press is accepted.
    assert not app._busy.locked()


def test_transcribe_no_retry_when_retries_zero(monkeypatch):
    """When transcribe_retries=0, no reload/retry should happen."""
    app, transcriber, _ = _build_app_for_recovery(transcribe_retries=0)
    transcriber.transcribe.side_effect = RuntimeError("crash")

    reload_called = []
    monkeypatch.setattr(
        app, "_reload_transcriber",
        lambda profile: reload_called.append(1) or MagicMock(),
    )

    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)

    assert reload_called == []
    app.injector.inject.assert_not_called()


def test_transcribe_success_on_first_try_no_reload(monkeypatch):
    """A successful transcribe should NOT trigger a reload."""
    app, transcriber, _ = _build_app_for_recovery()
    transcriber.transcribe.return_value = "hello"

    monkeypatch.setattr(app, "_reload_transcriber", MagicMock())

    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)

    app._reload_transcriber.assert_not_called()  # type: ignore[attr-defined]
    app.injector.inject.assert_called_once_with("hello")


# --- Microphone failover ---------------------------------------------------


def test_mic_failover_scans_for_fallback_device(fake_sd, capsys):
    """When the configured device is gone and mic_failover_scan=True,
    the recorder should scan all input devices and pick a working one."""
    from speakinput.audio import AudioRecorder

    # query_devices returns a list with one input device at index 5.
    fake_sd.query_devices.return_value = [
        {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000.0},
        {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 16000.0},
    ]
    # The _device_is_present check for device 2 raises (device gone).
    # But _find_fallback_device calls query_devices() which returns the list.
    # The issue: _device_is_present also calls query_devices(device).
    # We need query_devices to work for the scan but fail for the pinned device.
    call_count = {"n": 0}

    def query_side_effect(*args):
        call_count["n"] += 1
        if args:
            # query_devices(2) — pinned device check
            raise Exception("device 2 not found")
        # query_devices() — full list for fallback scan
        return [
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000.0},
            {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 16000.0},
        ]

    fake_sd.query_devices.side_effect = query_side_effect
    r = AudioRecorder(device=2, mic_failover_scan=True)
    r.start()
    kwargs = fake_sd.InputStream.call_args.kwargs
    # The fallback device should be index 1 (the USB Mic, the only
    # device with input channels > 0).
    assert kwargs["device"] == 1
    captured = capsys.readouterr()
    assert "switching to device" in captured.err


def test_mic_failover_disabled_falls_back_to_default(fake_sd, capsys):
    """When mic_failover_scan=False, the recorder falls back to system
    default (device=None) without scanning."""
    from speakinput.audio import AudioRecorder

    fake_sd.query_devices.side_effect = Exception("device gone")
    r = AudioRecorder(device=2, mic_failover_scan=False, mic_open_retries=0)
    r.start()
    kwargs = fake_sd.InputStream.call_args.kwargs
    assert kwargs["device"] is None
    captured = capsys.readouterr()
    assert "system default" in captured.err


def test_mic_open_retries_tries_multiple_devices(fake_sd):
    """When mic_open_retries > 0, the recorder retries opening with
    different fallback devices."""
    from speakinput.audio import AudioRecorder

    # First InputStream raises, second succeeds.
    call_count = {"n": 0}

    def input_stream_side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("device busy")
        return MagicMock()

    fake_sd.InputStream.side_effect = input_stream_side_effect
    fake_sd.query_devices.return_value = [
        {"name": "Mic", "max_input_channels": 1, "default_samplerate": 16000.0},
    ]
    r = AudioRecorder(device=None, mic_open_retries=2)
    r.start()
    assert call_count["n"] >= 2
    assert r.is_recording()


# --- Engine registry -------------------------------------------------------


def test_engine_registry_has_whispercpp():
    from speakinput.transcriber import _ENGINE_REGISTRY
    assert "whispercpp" in _ENGINE_REGISTRY


def test_engine_registry_has_faster_whisper():
    from speakinput.transcriber import _ENGINE_REGISTRY
    assert "faster_whisper" in _ENGINE_REGISTRY


def test_engine_registry_has_apple():
    from speakinput.transcriber import _ENGINE_REGISTRY
    assert "apple" in _ENGINE_REGISTRY


def test_create_transcriber_falls_back_to_whispercpp(monkeypatch, capsys):
    """When the requested engine isn't installed, create_transcriber
    falls back to whispercpp with a warning."""
    from speakinput.transcriber import create_transcriber

    # faster_whisper will raise TranscriberError because it's not installed.
    t = create_transcriber(
        engine="faster_whisper",
        model="small",
    )
    # Should have fallen back to WhisperCppTranscriber (which IS installed
    # in the test venv) — or, if the model can't load, at least the
    # function didn't raise. The key contract: no crash.
    from speakinput.transcriber import WhisperCppTranscriber
    assert isinstance(t, WhisperCppTranscriber)
    captured = capsys.readouterr()
    assert "faster_whisper" in captured.err
    assert "falling back" in captured.err


def test_create_transcriber_unknown_engine_falls_back(monkeypatch, capsys):
    """An unknown engine name should fall back to whispercpp."""
    from speakinput.transcriber import create_transcriber, WhisperCppTranscriber

    t = create_transcriber(engine="nonexistent", model="small")
    assert isinstance(t, WhisperCppTranscriber)


def test_config_engine_validation():
    """Config must validate the engine name."""
    with pytest.raises(ValueError, match="transcribe.engine"):
        Config.from_dict({"transcribe": {"engine": "bogus"}}).validate()


def test_config_engine_from_toml():
    cfg = Config.from_dict({"transcribe": {"engine": "faster_whisper"}})
    assert cfg.transcribe.engine == "faster_whisper"


# --- Injector fallback chain -----------------------------------------------


def test_fallback_injector_tries_primary_first():
    """The FallbackInjector should try the primary injector first."""
    from speakinput.injector import FallbackInjector

    primary = MagicMock()
    secondary = MagicMock()
    fi = FallbackInjector([primary, secondary])
    fi.inject("hello")
    primary.inject.assert_called_once_with("hello")
    secondary.inject.assert_not_called()


def test_fallback_injector_falls_through_on_failure(capsys):
    """When the primary injector raises, the FallbackInjector should
    try the next one in the chain."""
    from speakinput.injector import FallbackInjector

    primary = MagicMock()
    primary.inject.side_effect = RuntimeError("wtype crashed")
    secondary = MagicMock()
    fi = FallbackInjector([primary, secondary])
    fi.inject("hello")
    primary.inject.assert_called_once_with("hello")
    secondary.inject.assert_called_once_with("hello")
    captured = capsys.readouterr()
    assert "falling back" in captured.err


def test_fallback_injector_last_resort_raises():
    """When all injectors fail, the last exception should propagate."""
    from speakinput.injector import FallbackInjector

    primary = MagicMock()
    primary.inject.side_effect = RuntimeError("primary dead")
    secondary = MagicMock()
    secondary.inject.side_effect = RuntimeError("secondary dead")
    fi = FallbackInjector([primary, secondary])
    with pytest.raises(RuntimeError, match="secondary dead"):
        fi.inject("hello")


def test_fallback_injector_empty_chain_raises():
    from speakinput.injector import FallbackInjector
    with pytest.raises(ValueError, match="at least one"):
        FallbackInjector([])


def test_fallback_injector_advances_active_idx(capsys):
    """After a failure, the active index advances so subsequent calls
    go directly to the working injector."""
    from speakinput.injector import FallbackInjector

    primary = MagicMock()
    primary.inject.side_effect = RuntimeError("dead")
    secondary = MagicMock()
    fi = FallbackInjector([primary, secondary])
    fi.inject("first")
    fi.inject("second")
    # Primary was only called once (the first time); the second call
    # went directly to secondary.
    primary.inject.assert_called_once_with("first")
    assert secondary.inject.call_count == 2


def test_select_injector_with_fallback_disabled_returns_primary(monkeypatch):
    """When enable_fallback=False, select_injector_with_fallback returns
    a single injector (no FallbackInjector wrapper)."""
    from speakinput.config import InjectConfig
    from speakinput.injector import select_injector_with_fallback, TypingInjector, FallbackInjector

    monkeypatch.setattr("speakinput.injector.sys.platform", "darwin")
    inj = select_injector_with_fallback(InjectConfig(), enable_fallback=False)
    assert isinstance(inj, TypingInjector)
    assert not isinstance(inj, FallbackInjector)


def test_select_injector_with_fallback_on_macos_returns_single(monkeypatch):
    """On macOS there's only one backend (pynput), so even with
    fallback enabled, the result is a plain TypingInjector (no
    FallbackInjector wrapper needed)."""
    from speakinput.config import InjectConfig
    from speakinput.injector import select_injector_with_fallback, TypingInjector, FallbackInjector

    monkeypatch.setattr("speakinput.injector.sys.platform", "darwin")
    inj = select_injector_with_fallback(InjectConfig(), enable_fallback=True)
    assert isinstance(inj, TypingInjector)
    assert not isinstance(inj, FallbackInjector)


def test_select_injector_with_fallback_on_wayland_returns_chain(monkeypatch):
    """On Wayland with wtype available, the result should be a
    FallbackInjector with wtype as primary and pynput as fallback."""
    from speakinput.config import InjectConfig
    from speakinput.injector import (
        select_injector_with_fallback,
        FallbackInjector,
        WtypeInjector,
    )

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setattr(
        "speakinput.injector.shutil.which",
        lambda b: "/usr/bin/wtype" if b == "wtype" else None,
    )
    inj = select_injector_with_fallback(InjectConfig(), enable_fallback=True)
    assert isinstance(inj, FallbackInjector)
    assert isinstance(inj._injectors[0], WtypeInjector)