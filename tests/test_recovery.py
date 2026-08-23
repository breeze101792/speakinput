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

import threading
import time
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
    assert rc.transcribe_timeout_s == 30.0
    assert rc.inject_timeout_s == 10.0
    assert rc.busy_lock_timeout_s == 120.0


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
            "transcribe_timeout_s": 60.0,
            "inject_timeout_s": 20.0,
            "busy_lock_timeout_s": 300.0,
        }
    })
    assert cfg.recovery.transcribe_retries == 5
    assert cfg.recovery.transcribe_backoff_s == 1.0
    assert cfg.recovery.mic_failover_scan is False
    assert cfg.recovery.mic_open_retries == 0
    assert cfg.recovery.injector_fallback is False
    assert cfg.recovery.listener_restart_min_interval_s == 30.0
    assert cfg.recovery.transcribe_timeout_s == 60.0
    assert cfg.recovery.inject_timeout_s == 20.0
    assert cfg.recovery.busy_lock_timeout_s == 300.0


def test_recovery_config_validation_rejects_negative_retries():
    with pytest.raises(ValueError, match="transcribe_retries"):
        Config(recovery=RecoveryConfig(transcribe_retries=-1)).validate()


def test_recovery_config_validation_rejects_negative_backoff():
    with pytest.raises(ValueError, match="transcribe_backoff_s"):
        Config(recovery=RecoveryConfig(transcribe_backoff_s=-0.1)).validate()


def test_recovery_config_validation_rejects_negative_transcribe_timeout():
    with pytest.raises(ValueError, match="transcribe_timeout_s"):
        Config(recovery=RecoveryConfig(transcribe_timeout_s=-1)).validate()


def test_recovery_config_validation_rejects_negative_inject_timeout():
    with pytest.raises(ValueError, match="inject_timeout_s"):
        Config(recovery=RecoveryConfig(inject_timeout_s=-1)).validate()


def test_recovery_config_validation_rejects_negative_busy_lock_timeout():
    with pytest.raises(ValueError, match="busy_lock_timeout_s"):
        Config(recovery=RecoveryConfig(busy_lock_timeout_s=-1)).validate()


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

    # query_devices returns a list with one input device at index 1.
    def query_side_effect(*args):
        if args:
            # query_devices(2) — pinned device check
            raise Exception("device 2 not found")
        # query_devices() — full list for fallback scan
        return [
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000.0},
            {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 16000.0},
        ]

    fake_sd.query_devices.side_effect = query_side_effect
    # Make InputStream fail for device=None (system default) so the
    # recorder falls through to the scanned device 1.
    def input_stream_side_effect(**kwargs):
        if kwargs.get("device") is None:
            raise OSError("default device unavailable")
        return MagicMock()
    fake_sd.InputStream.side_effect = input_stream_side_effect

    r = AudioRecorder(device=2, mic_failover_scan=True)
    r.start()
    kwargs = fake_sd.InputStream.call_args.kwargs
    # The fallback device should be index 1 (the USB Mic, the only
    # device with input channels > 0).
    assert kwargs["device"] == 1
    captured = capsys.readouterr()
    assert "device 2 is not available" in captured.err


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
    assert "device 2 is not available" in captured.err


def test_mic_blacklist_prevents_retrying_broken_device(fake_sd, capsys):
    """When a specific device fails to open, it's blacklisted so the
    next press skips it and tries a different one. Only ONE
    InputStream() call happens per press — multiple calls corrupt
    PortAudio state (-9999)."""
    from speakinput.audio import AudioError, AudioRecorder

    # Two input devices: 0 is broken, 1 works.
    device_list = [
        {"name": "Broken", "max_input_channels": 1, "default_samplerate": 16000.0},
        {"name": "Working", "max_input_channels": 1, "default_samplerate": 16000.0},
    ]
    fake_sd.query_devices.side_effect = lambda *args: (
        device_list[int(args[0])] if args else device_list
    )

    def input_stream_side_effect(**kwargs):
        if kwargs.get("device") == 0:
            raise OSError("broken")
        return MagicMock()

    fake_sd.InputStream.side_effect = input_stream_side_effect

    r = AudioRecorder(device=0, mic_failover_scan=True, mic_open_retries=0)
    # First press: device 0 is present (query succeeds) but InputStream
    # fails. Only one InputStream call — no multi-device retry. Raises.
    with pytest.raises(AudioError):
        r.start()
    # Device 0 is now blacklisted.
    assert 0 in r._blacklisted_devices
    # Only ONE InputStream call happened (no retry that corrupts PA).
    assert fake_sd.InputStream.call_count == 1

    # Second press: device 0 is blacklisted, scan finds device 1,
    # which works.
    r.start()
    last_call = fake_sd.InputStream.call_args_list[-1]
    assert last_call.kwargs["device"] == 1
    assert r.is_recording()
    r.close()


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


# --- Bounded transcribe: hang recovery -------------------------------------


def test_bounded_call_returns_value_when_fast_enough():
    """_bounded_call returns (True, result) when fn finishes within the
    timeout."""
    from speakinput.app import _bounded_call

    ok, result = _bounded_call(lambda x: x * 2, 21, timeout_s=5.0, name="test")
    assert ok is True
    assert result == 42


def test_bounded_call_returns_false_on_timeout(capsys):
    """_bounded_call returns (False, None) when fn exceeds the timeout
    and prints a warning."""
    import threading

    from speakinput.app import _bounded_call

    never = threading.Event()

    def _hang():
        never.wait()  # blocks forever

    ok, result = _bounded_call(_hang, timeout_s=0.1, name="test-hang")
    assert ok is False
    assert result is None
    captured = capsys.readouterr()
    assert "test-hang" in captured.err
    assert "0s" in captured.err  # the timeout value
    # Cleanup: let the daemon thread die with the test process.
    never.set()


def test_bounded_call_zero_timeout_bypasses_thread():
    """When timeout_s=0, _bounded_call runs fn directly (no helper
    thread). This is the 'disabled' config path."""
    from speakinput.app import _bounded_call

    ok, result = _bounded_call(lambda: "direct", timeout_s=0, name="test")
    assert ok is True
    assert result == "direct"


def test_bounded_call_propagates_exception():
    """When fn raises, _bounded_call re-raises the exception."""
    from speakinput.app import _bounded_call

    def _boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _bounded_call(_boom, timeout_s=5.0, name="test")


def test_transcribe_hang_triggers_timeout_and_retry(monkeypatch, capsys):
    """When transcribe() hangs longer than transcribe_timeout_s, the
    App should abandon it, treat it like a crash (reload + retry), and
    the retry should succeed."""
    import threading

    app, transcriber, _ = _build_app_for_recovery(transcribe_timeout_s=0.1)

    # First call hangs forever, second succeeds (on the reloaded
    # transcriber).
    never = threading.Event()

    def _hang(*args, **kwargs):
        never.wait()

    transcriber.transcribe.side_effect = _hang
    new_transcriber = MagicMock()
    new_transcriber.transcribe.return_value = "recovered after hang"
    monkeypatch.setattr(app, "_reload_transcriber", lambda profile: new_transcriber)

    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)

    # The reload was called (the timeout was treated like a crash).
    new_transcriber.transcribe.assert_called_once()
    # The text was injected.
    app.injector.inject.assert_called_once_with("recovered after hang")
    # The lock was released so the next press is accepted.
    assert not app._busy.locked()
    captured = capsys.readouterr()
    assert "transcribe" in captured.err.lower()
    assert "0s" in captured.err  # timeout value in the warning
    never.set()  # let the daemon thread die


def test_transcribe_hang_exhausted_retries_releases_lock(monkeypatch):
    """When all retries are exhausted on a hanging transcribe, the lock
    must be released so the next press works."""
    import threading

    app, transcriber, _ = _build_app_for_recovery(
        transcribe_retries=1, transcribe_timeout_s=0.1
    )

    never = threading.Event()

    def _hang(*args, **kwargs):
        never.wait()

    transcriber.transcribe.side_effect = _hang
    monkeypatch.setattr(
        app,
        "_reload_transcriber",
        lambda profile: MagicMock(transcribe=MagicMock(side_effect=_hang)),
    )

    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)

    # No text was injected.
    app.injector.inject.assert_not_called()
    # The lock was released.
    assert not app._busy.locked()
    never.set()


def test_transcribe_timeout_zero_disables_bounded_call(monkeypatch):
    """When transcribe_timeout_s=0, the bounded call is bypassed and
    transcribe() runs directly (old behavior). A fast transcribe works
    normally."""
    app, transcriber, _ = _build_app_for_recovery(transcribe_timeout_s=0)
    transcriber.transcribe.return_value = "hello"

    monkeypatch.setattr(app, "_reload_transcriber", MagicMock())

    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)

    app._reload_transcriber.assert_not_called()  # type: ignore[attr-defined]
    app.injector.inject.assert_called_once_with("hello")


# --- Bounded inject: hang recovery -----------------------------------------


def test_inject_hang_abandoned_and_lock_released(monkeypatch, capsys):
    """When injector.inject() hangs longer than inject_timeout_s, the
    App should abandon it, log a warning, and release the busy lock so
    the next press works."""
    import threading

    app, _, _ = _build_app_for_recovery(inject_timeout_s=0.1)
    # Make transcribe fast so we reach the inject path.
    app.transcribers[app.config.primary.key].transcribe.return_value = "hello"

    never = threading.Event()

    def _hang_inject(text):
        never.wait()

    app.injector.inject.side_effect = _hang_inject

    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)

    # The lock was released despite the inject hang.
    assert not app._busy.locked()
    captured = capsys.readouterr()
    assert "inject" in captured.err.lower()
    assert "0s" in captured.err  # timeout value
    never.set()


def test_inject_timeout_zero_disables_bounded_call():
    """When inject_timeout_s=0, the bounded call is bypassed and
    inject() runs directly (old behavior)."""
    app, _, _ = _build_app_for_recovery(inject_timeout_s=0)
    app.transcribers[app.config.primary.key].transcribe.return_value = "hello"

    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)

    app.injector.inject.assert_called_once_with("hello")


# --- Busy-lock watchdog -----------------------------------------------------


def test_busy_lock_watchdog_fires_on_stuck_press(capsys):
    """When the busy lock has been held longer than
    busy_lock_timeout_s, the watcher should fire on_busy_stuck."""
    from speakinput.app import _LivenessWatcher

    on_busy_stuck = MagicMock()
    lock = threading.Lock()
    lock.acquire()
    started = time.monotonic() - 200.0  # pretend the press started 200s ago

    watcher = _LivenessWatcher(
        listeners=[],
        interval_s=0.01,
        on_dead=MagicMock(),
        busy_lock=lock,
        press_started_at=lambda: started,
        busy_lock_timeout_s=100.0,
        on_busy_stuck=on_busy_stuck,
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
        lock.release()
    on_busy_stuck.assert_called_once()
    held_for = on_busy_stuck.call_args[0][0]
    assert held_for >= 100.0


def test_busy_lock_watchdog_does_not_fire_when_unheld():
    """When the busy lock is not held, the watcher must not fire."""
    from speakinput.app import _LivenessWatcher

    on_busy_stuck = MagicMock()
    lock = threading.Lock()  # unheld

    watcher = _LivenessWatcher(
        listeners=[],
        interval_s=0.01,
        on_dead=MagicMock(),
        busy_lock=lock,
        press_started_at=lambda: time.monotonic(),
        busy_lock_timeout_s=0.05,
        on_busy_stuck=on_busy_stuck,
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
    on_busy_stuck.assert_not_called()


def test_busy_lock_watchdog_does_not_fire_within_timeout():
    """When the lock has been held for less than the timeout, the
    watcher must not fire."""
    from speakinput.app import _LivenessWatcher

    on_busy_stuck = MagicMock()
    lock = threading.Lock()
    lock.acquire()
    started = time.monotonic()  # just now

    watcher = _LivenessWatcher(
        listeners=[],
        interval_s=0.01,
        on_dead=MagicMock(),
        busy_lock=lock,
        press_started_at=lambda: started,
        busy_lock_timeout_s=100.0,  # 100s timeout, held for <1s
        on_busy_stuck=on_busy_stuck,
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
        lock.release()
    on_busy_stuck.assert_not_called()


def test_busy_lock_watchdog_fires_only_once_per_episode(capsys):
    """The watchdog should fire on_busy_stuck at most once per stuck
    episode, then reset when the lock is released."""
    from speakinput.app import _LivenessWatcher

    on_busy_stuck = MagicMock()
    lock = threading.Lock()
    lock.acquire()
    started = time.monotonic() - 200.0

    watcher = _LivenessWatcher(
        listeners=[],
        interval_s=0.01,
        on_dead=MagicMock(),
        busy_lock=lock,
        press_started_at=lambda: started,
        busy_lock_timeout_s=100.0,
        on_busy_stuck=on_busy_stuck,
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
        lock.release()
    # Only one call despite multiple poll cycles.
    on_busy_stuck.assert_called_once()


def test_busy_lock_watchdog_disabled_when_timeout_zero():
    """When busy_lock_timeout_s=0, the watchdog is disabled."""
    from speakinput.app import _LivenessWatcher

    on_busy_stuck = MagicMock()
    lock = threading.Lock()
    lock.acquire()
    started = time.monotonic() - 9999.0

    watcher = _LivenessWatcher(
        listeners=[],
        interval_s=0.01,
        on_dead=MagicMock(),
        busy_lock=lock,
        press_started_at=lambda: started,
        busy_lock_timeout_s=0,
        on_busy_stuck=on_busy_stuck,
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
        lock.release()
    on_busy_stuck.assert_not_called()


def test_on_busy_stuck_enqueues_abort_press(capsys):
    """_on_busy_stuck should enqueue _abort_press on the event worker
    so it's serialized with any in-flight handler."""
    from speakinput.config import AudioConfig, Config
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: MagicMock()},
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    # Start the event worker so _on_busy_stuck enqueues instead of
    # running the abort inline.
    app._start_event_worker()
    # Simulate a stuck press: acquire the busy lock and set
    # _press_started_at to the past.
    app._busy.acquire()
    app._press_started_at = time.monotonic() - 200.0
    app._active_profile = config.primary

    app._on_busy_stuck(200.0)

    # The abort was enqueued (not run inline).
    fn, args = app._work_q.get_nowait()
    assert fn == app._abort_press
    assert "busy lock watchdog" in args[0]
    captured = capsys.readouterr()
    assert "busy lock held for 200s" in captured.err
    # Cleanup: drain the queue and release the lock.
    app._stop_event_worker()
    app._busy.release()


# --- Event worker liveness monitoring --------------------------------------


def test_liveness_watcher_detects_dead_worker():
    """When the event worker thread dies, the watcher should fire
    on_worker_dead once."""
    from speakinput.app import _LivenessWatcher

    on_worker_dead = MagicMock()
    worker = MagicMock()
    worker.is_alive.return_value = False

    watcher = _LivenessWatcher(
        listeners=[],
        interval_s=0.01,
        on_dead=MagicMock(),
        worker_thread=worker,
        on_worker_dead=on_worker_dead,
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
    on_worker_dead.assert_called_once()


def test_liveness_watcher_does_not_fire_when_worker_alive():
    """A healthy worker thread must not trigger on_worker_dead."""
    from speakinput.app import _LivenessWatcher

    on_worker_dead = MagicMock()
    worker = MagicMock()
    worker.is_alive.return_value = True

    watcher = _LivenessWatcher(
        listeners=[],
        interval_s=0.01,
        on_dead=MagicMock(),
        worker_thread=worker,
        on_worker_dead=on_worker_dead,
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
    on_worker_dead.assert_not_called()


def test_liveness_watcher_no_worker_monitoring_when_none():
    """When worker_thread is None, the watcher must not try to monitor
    it (no crash, no callback)."""
    from speakinput.app import _LivenessWatcher

    on_worker_dead = MagicMock()
    watcher = _LivenessWatcher(
        listeners=[],
        interval_s=0.01,
        on_dead=MagicMock(),
        worker_thread=None,
        on_worker_dead=on_worker_dead,
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
    on_worker_dead.assert_not_called()


def test_swap_worker_resets_tracking():
    """swap_worker replaces the worker thread and resets the alive-
    transition tracking so the new worker's future death fires
    on_worker_dead exactly once."""
    from speakinput.app import _LivenessWatcher

    on_worker_dead = MagicMock()
    old_worker = MagicMock()
    old_worker.is_alive.return_value = False  # dead
    new_worker = MagicMock()
    new_worker.is_alive.return_value = True  # alive

    watcher = _LivenessWatcher(
        listeners=[],
        interval_s=0.01,
        on_dead=MagicMock(),
        worker_thread=old_worker,
        on_worker_dead=on_worker_dead,
    )
    # The old worker is dead, but we swap before the watcher polls.
    watcher.swap_worker(new_worker)
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
    # The dead old worker must NOT fire on_worker_dead after the swap.
    on_worker_dead.assert_not_called()


def test_on_worker_dead_restarts_worker(capsys):
    """_on_worker_dead should restart the event worker thread."""
    from speakinput.config import AudioConfig, Config
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    app = App(
        config=config,
        recorder=MagicMock(),
        transcribers={config.primary.key: MagicMock()},
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    # Start the real worker so the restart has something to replace.
    app._start_event_worker()
    old_thread = app._worker_thread
    assert old_thread is not None

    app._on_worker_dead()
    # A new worker thread was created.
    assert app._worker_thread is not None
    assert app._worker_thread is not old_thread
    assert app._worker_thread.is_alive()
    captured = capsys.readouterr()
    assert "event worker died and was restarted" in captured.err
    # Cleanup: stop the worker.
    app._stop_event_worker()


def test_on_worker_dead_warns_when_restart_fails(monkeypatch, capsys):
    """If the worker restart fails, the user must get a warning."""
    from speakinput.config import AudioConfig, Config
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    app = App(
        config=config,
        recorder=MagicMock(),
        transcribers={config.primary.key: MagicMock()},
        injector=MagicMock(),
        feedback=MagicMock(),
    )

    def _fail_restart():
        return False

    monkeypatch.setattr(app, "_restart_event_worker", _fail_restart)
    app._on_worker_dead()
    captured = capsys.readouterr()
    assert "event worker" in captured.err
    assert "Restart speakinput" in captured.err


def test_restart_event_worker_warns_on_shutdown():
    """During shutdown, _restart_event_worker should be a no-op."""
    from speakinput.config import AudioConfig, Config
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    app = App(
        config=config,
        recorder=MagicMock(),
        transcribers={config.primary.key: MagicMock()},
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app._shutdown.set()
    result = app._restart_event_worker()
    assert result is False
    assert app._worker_thread is None