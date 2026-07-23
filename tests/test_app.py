"""Tests for the App orchestrator. Mocks audio + transcribers + injector."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import time

from speakinput.config import AudioConfig, Config, Profile


@pytest.fixture(autouse=True)
def _stub_hotkey_listener(monkeypatch):
    """Replace HotkeyListener with a no-op mock for the whole module.

    pynput's keyboard.Listener.init crashes on this test environment
    (macOS HIToolbox / headless Linux with no X11 display) — see
    earlier commits — so every test that would start a real listener
    uses this stub. The class mock returns a fresh listener instance
    per construction so per-profile assertions work as expected.

    `app.run()` also calls `resolve_key()` from `speakinput.hotkey`,
    which dereferences `keyboard.Key.<name>`. On a real macOS box
    pynput imports fine, but on headless Linux pynput's import-time
    X11 connect fails and `speakinput.hotkey.keyboard` ends up as
    None. We stub that module-level `keyboard` to a MagicMock with
    matching `Key.<name>` attributes so `resolve_key` works in
    either environment.

    The same goes for the new `EvdevHotkeyListener`: it would try to
    open `/dev/input/event*`, which fails on every test environment
    (no real keyboard, no permissions, or just no /dev/input in CI
    containers). We stub it to a no-op MagicMock for symmetry. Tests
    that specifically exercise evdev behavior live in test_hotkey.py
    with a properly-shaped fake.
    """
    fake_listener_cls = MagicMock()
    fake_listener_cls.side_effect = lambda *a, **kw: MagicMock()
    monkeypatch.setattr("speakinput.app.HotkeyListener", fake_listener_cls)
    fake_keyboard = MagicMock()
    fake_keyboard.Key.alt_r = "alt_r_key"
    fake_keyboard.Key.ctrl_r = "ctrl_r_key"
    fake_keyboard.Key.cmd_r = "cmd_r_key"
    fake_keyboard.Key.shift_r = "shift_r_key"
    fake_keyboard.Key.caps_lock = "caps_lock_key"
    fake_keyboard.Key.f12 = "f12_key"
    monkeypatch.setattr("speakinput.hotkey.keyboard", fake_keyboard, raising=False)
    fake_evdev_listener_cls = MagicMock()
    fake_evdev_listener_cls.side_effect = lambda *a, **kw: MagicMock()
    monkeypatch.setattr("speakinput.app.EvdevHotkeyListener", fake_evdev_listener_cls)
    # Also stub pynput's Controller inside the injector module so
    # `select_injector(...)` can construct a `TypingInjector` on
    # macOS / X11 Linux / "no wtype/ydotool" fallbacks without
    # actually starting a HIToolbox or X11 connection.
    fake_inj_keyboard = MagicMock()
    fake_inj_keyboard.Key = MagicMock()
    fake_inj_keyboard.Key.cmd = "cmd"
    fake_inj_keyboard.Key.ctrl = "ctrl"
    fake_inj_keyboard.Controller = MagicMock()
    monkeypatch.setattr(
        "speakinput.injector.Controller", fake_inj_keyboard.Controller, raising=False
    )
    monkeypatch.setattr(
        "speakinput.injector.Key", fake_inj_keyboard.Key, raising=False
    )
    return fake_listener_cls


def _build_app(debug: bool = False, dry_run: bool = False, config: Config | None = None,
               transcribers: dict | None = None):
    """Build an App with all I/O collaborators mocked out.

    `transcribers` defaults to a single mock keyed by the primary profile's
    hotkey; override it to simulate a multi-profile setup."""
    from speakinput.app import App

    config = config or Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.zeros(16000, dtype=np.float32)  # 1s of "silence"

    if transcribers is None:
        t = MagicMock()
        t.transcribe.return_value = "hello world"
        transcribers = {config.primary.key: t}

    injector = MagicMock()
    feedback = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=injector,
        feedback=feedback,
        dry_run=dry_run,
        debug=debug,
    )
    return app, recorder, transcribers, injector, feedback


# --- press/release: single-profile path ----------------------------------


def test_press_calls_recorder_start_and_marks_listening(capsys):
    app, recorder, _, _, feedback = _build_app()
    app.on_hotkey_press(app.config.primary)
    recorder.start.assert_called_once()
    feedback.set_state.assert_called_with("listening")
    captured = capsys.readouterr()
    assert "[debug]" not in captured.err


def test_press_ignored_when_already_busy(capsys):
    app, recorder, _, _, _ = _build_app()
    app.on_hotkey_press(app.config.primary)  # first press acquires the lock
    recorder.start.reset_mock()
    app.on_hotkey_press(app.config.primary)  # second press should be ignored
    recorder.start.assert_not_called()


def test_release_press_failed_means_nothing_to_do(capsys):
    app, recorder, transcribers, injector, _ = _build_app()
    recorder.is_recording.return_value = False
    app.on_hotkey_release(app.config.primary)
    transcribers[app.config.primary.key].transcribe.assert_not_called()
    injector.inject.assert_not_called()


def test_release_happy_path_calls_transcribe_and_inject(capsys):
    app, recorder, transcribers, injector, feedback = _build_app()
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    transcribers[app.config.primary.key].transcribe.assert_called_once()
    injector.inject.assert_called_once_with("hello world")
    feedback.set_state.assert_any_call("idle")


def test_release_empty_transcript_does_not_inject(capsys):
    app, _, transcribers, injector, _ = _build_app()
    transcribers[app.config.primary.key].transcribe.return_value = ""
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    injector.inject.assert_not_called()


def test_release_silence_skips_transcribe(capsys):
    """If audio RMS is below the configured silence_threshold, we must NOT
    call the transcriber. Whisper hallucinates on near-empty audio; better
    to skip the call entirely and log a debug line so the user can see
    why nothing was typed."""
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0.005, auto_stop_seconds=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.zeros(16000, dtype=np.float32)
    transcriber = MagicMock()
    injector = MagicMock()
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=injector,
        feedback=MagicMock(),
        debug=True,
    )
    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)

    transcriber.transcribe.assert_not_called()
    injector.inject.assert_not_called()
    captured = capsys.readouterr()
    assert "silence gate" in captured.err
    assert "skipping transcribe" in captured.err


def test_release_silence_threshold_zero_disables_gate(capsys):
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.zeros(16000, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = ""
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
        debug=True,
    )
    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)

    transcriber.transcribe.assert_called_once()


def test_release_loud_audio_passes_gate(capsys):
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0.005, auto_stop_seconds=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(16000, 0.5, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "hello"
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
        debug=True,
    )
    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)

    transcriber.transcribe.assert_called_once()


def test_dry_run_prints_text_to_stderr_instead_of_typing(capsys):
    app, _, transcribers, injector, _ = _build_app(dry_run=True)
    transcribers[app.config.primary.key].transcribe.return_value = "dry run output"
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    captured = capsys.readouterr()
    assert "dry run output" in captured.err
    injector.inject.assert_not_called()


# --- debug mode ------------------------------------------------------------


def test_debug_mode_logs_press_start_and_end(capsys):
    app, _, _, _, _ = _build_app(debug=True)
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    captured = capsys.readouterr()
    assert f"[debug] key press start ({app.config.primary.key})" in captured.err
    assert "[debug] key press end" in captured.err
    assert "held" in captured.err


def test_debug_mode_logs_audio_stats(capsys):
    app, recorder, _, _, _ = _build_app(debug=True)
    recorder.drain.return_value = np.zeros(16000, dtype=np.float32)
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    captured = capsys.readouterr()
    assert "[debug] audio: 16000 samples" in captured.err
    assert "rms=" in captured.err


def test_debug_mode_prints_transcript(capsys):
    app, _, transcribers, _, _ = _build_app(debug=True)
    transcribers[app.config.primary.key].transcribe.return_value = "the quick brown fox"
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    captured = capsys.readouterr()
    assert "[debug] transcript: 'the quick brown fox'" in captured.err


def test_debug_mode_prints_empty_transcript(capsys):
    app, _, transcribers, _, _ = _build_app(debug=True)
    transcribers[app.config.primary.key].transcribe.return_value = ""
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    captured = capsys.readouterr()
    assert "[debug] transcript: ''" in captured.err


def test_debug_mode_ignored_press_logs_message(capsys):
    app, _, _, _, _ = _build_app(debug=True)
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_press(app.config.primary)
    captured = capsys.readouterr()
    assert "press ignored: already busy" in captured.err


def test_no_debug_output_when_disabled(capsys):
    app, _, transcribers, _, _ = _build_app(debug=False)
    transcribers[app.config.primary.key].transcribe.return_value = "should not appear"
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    captured = capsys.readouterr()
    assert "[debug]" not in captured.err
    assert "should not appear" not in captured.err


def test_transcribe_error_releases_lock_and_returns_to_idle(capsys):
    app, _, transcribers, injector, feedback = _build_app(debug=True)
    transcribers[app.config.primary.key].transcribe.side_effect = RuntimeError("model exploded")
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    # Lock should be released — a fresh press should now succeed.
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    assert injector.inject.call_count == 0
    feedback.set_state.assert_any_call("idle")


def test_inject_error_does_not_crash(capsys):
    app, _, _, injector, _ = _build_app(debug=True)
    injector.inject.side_effect = RuntimeError("pbcopy failed")
    app.on_hotkey_press(app.config.primary)
    # Should NOT raise — the release path catches injection errors.
    app.on_hotkey_release(app.config.primary)


# --- Chinese conversion ---------------------------------------------------


def test_zh_conversion_converts_simplified_to_traditional():
    """When zh_conversion="traditional" and the transcript contains Chinese
    characters, the text must be converted to Traditional Chinese
    before injection."""
    from speakinput.app import App
    from speakinput.config import Profile

    config = Config(
        audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0),
        primary=Profile(key="alt_r", language="zh", zh_conversion="traditional"),
    )
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(16000, 0.3, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "简化字"
    transcribers = {config.primary.key: transcriber}
    injector = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=injector,
        feedback=MagicMock(),
    )
    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)
    transcriber.transcribe.assert_called_once()
    injected = injector.inject.call_args[0][0]
    assert injected == "簡化字"


def test_zh_conversion_converts_traditional_to_simplified():
    """When zh_conversion="simplified", Traditional Chinese text must be
    converted to Simplified Chinese before injection."""
    from speakinput.app import App
    from speakinput.config import Profile

    config = Config(
        audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0),
        primary=Profile(key="alt_r", language="zh", zh_conversion="simplified"),
    )
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(16000, 0.3, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "簡化字"
    transcribers = {config.primary.key: transcriber}
    injector = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=injector,
        feedback=MagicMock(),
    )
    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)
    transcriber.transcribe.assert_called_once()
    injected = injector.inject.call_args[0][0]
    assert injected == "简化字"


def test_zh_conversion_skipped_when_off():
    """When zh_conversion="off", Chinese text must be injected as-is
    without conversion."""
    from speakinput.app import App
    from speakinput.config import Profile

    config = Config(
        audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0),
        primary=Profile(key="alt_r", language="zh", zh_conversion="off"),
    )
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(16000, 0.3, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "简化字"
    transcribers = {config.primary.key: transcriber}
    injector = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=injector,
        feedback=MagicMock(),
    )
    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)
    transcriber.transcribe.assert_called_once()
    injected = injector.inject.call_args[0][0]
    assert injected == "简化字"  # unchanged


def test_zh_conversion_does_not_affect_english():
    """English text must pass through unchanged even when zh_conversion
    is enabled."""
    from speakinput.app import App

    config = Config(
        audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0),
        primary=Profile(zh_conversion="traditional"),
    )
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(16000, 0.3, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "hello world"
    transcribers = {config.primary.key: transcriber}
    injector = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=injector,
        feedback=MagicMock(),
    )
    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)
    injected = injector.inject.call_args[0][0]
    assert injected == "hello world"


def test_zh_conversion_default_is_traditional():
    """The default Profile should have zh_conversion="traditional"."""
    assert Profile().zh_conversion == "traditional"


def test_zh_conversion_shown_in_startup_banner(monkeypatch, capsys):
    """The startup banner must show zh_conversion value per profile."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(
        primary=Profile(key="alt_r", zh_conversion="traditional"),
        secondary=Profile(key="cmd_r", language="zh", zh_conversion="off"),
    )
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    assert "zh_conversion=traditional" in captured.err
    assert "zh_conversion=off" in captured.err


# --- model bootstrap tests ------------------------------------------------


def test_run_bootstraps_model_when_no_transcribers_injected(monkeypatch):
    """If no transcribers were passed in, run() must call _build_transcribers,
    which in turn calls ensure_model. Two profiles means two model loads
    (or one shared load if the model paths collide)."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(side_effect=lambda name: f"/resolved/{name}.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(
        primary=Profile(key="alt_r", model="small", language="auto"),
        secondary=Profile(key="cmd_r", model="small", language="zh"),
    )
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    assert app.transcribers == {}  # precondition

    app._shutdown.set()
    app.run()

    # Two profiles, both `small` -> one model file downloaded, one
    # WhisperCppTranscriber constructed (deduped by name+path).
    assert fake_ensure.call_count == 1
    assert fake_ensure.call_args.args == ("small",)
    assert fake_model_cls.call_count == 1
    # Two listeners started.
    assert set(app.listeners.keys()) == {"alt_r", "cmd_r"}


def test_run_bootstraps_two_distinct_models(monkeypatch):
    """Two profiles with different model files => two ensure_model calls
    and two WhisperCppTranscriber constructions."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(side_effect=lambda name: f"/resolved/{name}.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(
        primary=Profile(key="alt_r", model="base.en", language="en"),
        secondary=Profile(key="cmd_r", model="small", language="zh"),
    )
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    # base.en + en is allowed (English-only stays). base + zh is
    # multilingual, no upgrade. Two distinct model files, two
    # transcribers, two listeners.
    assert fake_ensure.call_count == 2
    assert fake_model_cls.call_count == 2
    assert set(app.listeners.keys()) == {"alt_r", "cmd_r"}


def test_run_skips_bootstrap_when_transcribers_injected():
    """If transcribers were passed in, run() must not call ensure_model —
    the test owns the transcribers."""
    app, _, transcribers, _, _ = _build_app()
    assert app.transcribers is transcribers  # precondition

    app._shutdown.set()
    app.run()  # should not raise
    assert app.transcribers is transcribers


def test_run_with_unknown_model_exits(monkeypatch, capsys):
    from speakinput.app import App
    from speakinput.config import Profile
    from speakinput.models import ModelNotFoundError

    fake_ensure = MagicMock(side_effect=ModelNotFoundError("unknown model"))
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)

    config = Config(primary=Profile(model="bogus"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())

    with pytest.raises(SystemExit) as exc_info:
        app.run()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "unknown model" in captured.err


def test_run_with_download_failure_exits(monkeypatch, capsys):
    from speakinput.app import App
    from speakinput.models import ModelDownloadError

    fake_ensure = MagicMock(side_effect=ModelDownloadError("network down"))
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)

    config = Config()
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())

    with pytest.raises(SystemExit) as exc_info:
        app.run()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "network down" in captured.err


# --- language / model auto-upgrade ----------------------------------------


def test_run_upgrades_english_only_model_for_zh(monkeypatch, capsys):
    """`base.en` + `language=zh` must auto-upgrade to `base` so Chinese
    actually works. The user is told about the swap in the log."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/base.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(primary=Profile(key="alt_r", model="base.en", language="zh"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    fake_ensure.assert_called_once_with("base")
    captured = capsys.readouterr()
    assert "upgrading" in captured.err
    assert "base.en" in captured.err
    assert "base" in captured.err


def test_run_upgrades_english_only_model_for_auto(monkeypatch, capsys):
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(primary=Profile(key="alt_r", model="small.en", language="auto"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    fake_ensure.assert_called_once_with("small")
    captured = capsys.readouterr()
    assert "upgrading" in captured.err


def test_run_does_not_upgrade_when_english_only_model_with_en(monkeypatch, capsys):
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/base.en.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(primary=Profile(key="alt_r", model="base.en", language="en"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    fake_ensure.assert_called_once_with("base.en")
    captured = capsys.readouterr()
    assert "upgrading" not in captured.err


def test_run_does_not_upgrade_multilingual_model(monkeypatch, capsys):
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(primary=Profile(key="alt_r", model="small", language="zh"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    fake_ensure.assert_called_once_with("small")
    captured = capsys.readouterr()
    assert "upgrading" not in captured.err


# --- multi-profile dispatch -------------------------------------------------


def test_secondary_profile_press_uses_secondary_transcriber(capsys):
    """A press on the secondary key must dispatch to the secondary
    profile's transcriber, not the primary's. This is the whole point
    of the two-profile design — different language, different model
    settings."""
    from speakinput.app import App

    config = Config(
        audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0),
        primary=Profile(key="alt_r", model="small", language="en"),
        secondary=Profile(key="cmd_r", model="small", language="zh"),
    )
    primary_t = MagicMock()
    primary_t.transcribe.return_value = "english text"
    secondary_t = MagicMock()
    secondary_t.transcribe.return_value = "中文文字"
    transcribers = {"alt_r": primary_t, "cmd_r": secondary_t}

    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.zeros(16000, dtype=np.float32)

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
    )

    # Press secondary, release secondary — the secondary transcriber
    # runs, the primary does not.
    app.on_hotkey_press(config.secondary)
    app.on_hotkey_release(config.secondary)
    primary_t.transcribe.assert_not_called()
    secondary_t.transcribe.assert_called_once()
    app.injector.inject.assert_called_once_with("中文文字")


def test_primary_profile_press_uses_primary_transcriber(capsys):
    """Symmetric to the above — primary key press dispatches to primary."""
    from speakinput.app import App

    config = Config(
        audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0),
        primary=Profile(key="alt_r", model="small", language="en"),
        secondary=Profile(key="cmd_r", model="small", language="zh"),
    )
    primary_t = MagicMock()
    primary_t.transcribe.return_value = "english text"
    secondary_t = MagicMock()
    transcribers = {"alt_r": primary_t, "cmd_r": secondary_t}

    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.zeros(16000, dtype=np.float32)

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
    )

    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)
    primary_t.transcribe.assert_called_once()
    secondary_t.transcribe.assert_not_called()
    app.injector.inject.assert_called_once_with("english text")


def test_two_listeners_started_for_two_profiles(monkeypatch):
    """run() must start a HotkeyListener for every profile."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(
        primary=Profile(key="alt_r"),
        secondary=Profile(key="cmd_r"),
    )
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()
    assert set(app.listeners.keys()) == {"alt_r", "cmd_r"}


def test_run_uses_evdev_listener_on_wayland(monkeypatch, capsys):
    """On a Wayland session (XDG_SESSION_TYPE=wayland), app.run() must
    construct EvdevHotkeyListener instead of the pynput-based
    HotkeyListener. The autouse fixture stubs both, so we just verify
    the right one was called."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")

    config = Config(primary=Profile(key="alt_r"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    # Banner reflects the chosen backend.
    assert "evdev" in captured.err
    # The autouse stub captured both class-level calls; check evdev won.
    import speakinput.app as appmod
    assert appmod.EvdevHotkeyListener.called
    assert not appmod.HotkeyListener.called


def test_run_falls_back_to_pynput_when_evdev_cant_find_keyboard(monkeypatch, capsys):
    """On Linux, evdev is preferred because it works on both Wayland
    and X11. If `probe_evdev_available()` reports no accessible keyboard
    in /dev/input (no permission, no /dev/input, no keyboard attached),
    the app falls back to pynput's X11 backend so the user still gets
    a working (or at least a clearly-failing) listener instead of
    silent no-ops."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    # Force the evdev probe to fail so we exercise the fallback path.
    monkeypatch.setattr("speakinput.app.probe_evdev_available", lambda: False)

    config = Config(primary=Profile(key="alt_r"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    # The banner explains WHY evdev was skipped so the user can debug
    # the fallback. Both the pynput line and the probe-failed diagnostic
    # mention evdev by name.
    assert "pynput" in captured.err
    assert "evdev probe failed" in captured.err
    import speakinput.app as appmod
    assert appmod.HotkeyListener.called
    assert not appmod.EvdevHotkeyListener.called


def test_run_prefers_evdev_on_linux_x11(monkeypatch, capsys):
    """On Linux, evdev is preferred over pynput because it works on both
    Wayland and X11 (it reads /dev/input directly, no X server needed).
    Previously the app gated evdev on `XDG_SESSION_TYPE=wayland`, which
    silently fell through to pynput on X11 sessions — and to a broken
    pynput on Wayland sessions where the env var wasn't propagated
    (tmux, SSH, containers). The new probe makes the decision based on
    actual evdev availability, not the session type."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)
    # Simulate an X11 Linux session — evdev is still the right choice.
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")

    config = Config(primary=Profile(key="alt_r"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    assert "evdev" in captured.err
    import speakinput.app as appmod
    assert appmod.EvdevHotkeyListener.called
    assert not appmod.HotkeyListener.called


def test_app_uses_wtype_injector_on_wayland(monkeypatch):
    """On a Wayland session, App() must construct a WtypeInjector when
    wtype is on PATH (it is on wlroots-based compositors). The default
    pynput TypingInjector only works on X11 Linux / XWayland."""
    from speakinput.app import App
    from speakinput.config import Profile

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    # Pretend wtype is on PATH; ydotool is not.
    monkeypatch.setattr(
        "speakinput.injector.shutil.which",
        lambda b: "/usr/bin/wtype" if b == "wtype" else None,
    )
    config = Config(primary=Profile(key="alt_r"))
    app = App(
        config=config,
        recorder=MagicMock(),
        transcribers={config.primary.key: MagicMock()},
        injector=None,  # force the auto-select path
        feedback=MagicMock(),
    )
    from speakinput.injector import WtypeInjector

    assert isinstance(app.injector, WtypeInjector)


def test_app_uses_pynput_injector_on_macos(monkeypatch):
    """On macOS, App() must always construct the pynput TypingInjector
    (HIToolbox) regardless of what Linux tools are installed."""
    from speakinput.app import App
    from speakinput.config import Profile

    monkeypatch.setattr("speakinput.injector.sys.platform", "darwin")
    config = Config(primary=Profile(key="alt_r"))
    app = App(
        config=config,
        recorder=MagicMock(),
        transcribers={config.primary.key: MagicMock()},
        injector=None,
        feedback=MagicMock(),
    )
    from speakinput.injector import TypingInjector

    assert isinstance(app.injector, TypingInjector)


def test_shutdown_stops_all_listeners(monkeypatch):
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(
        primary=Profile(key="alt_r"),
        secondary=Profile(key="cmd_r"),
    )
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()
    # Listeners remain in the dict after shutdown (so tests can
    # verify wiring) and each one had stop() called.
    assert set(app.listeners.keys()) == {"alt_r", "cmd_r"}
    for listener in app.listeners.values():
        listener.stop.assert_called_once()


def test_shared_model_loads_once(monkeypatch):
    """Two profiles using the same model file should load the model
    once and share the WhisperCppTranscriber instance. Same path ==
    same RAM footprint as one profile."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/shared/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(
        primary=Profile(key="alt_r", model="small"),
        secondary=Profile(key="cmd_r", model="small"),
    )
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    # One file download, one WhisperCppTranscriber, but two key -> transcriber
    # entries pointing to the same instance.
    assert fake_ensure.call_count == 1
    assert fake_model_cls.call_count == 1
    assert app.transcribers["alt_r"] is app.transcribers["cmd_r"]


# --- startup banner --------------------------------------------------------


def test_run_prints_startup_banner(monkeypatch, capsys):
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(
        primary=Profile(key="alt_r"),
        secondary=Profile(key="cmd_r", language="zh"),
    )
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    assert "[startup] config   : (defaults — no config.toml found)" in captured.err
    assert "[startup] profile 1 : key=alt_r model=small language=auto prompt=set zh_conversion=traditional" in captured.err
    assert "[startup] profile 2 : key=cmd_r model=small language=zh prompt=set zh_conversion=traditional" in captured.err
    # Both profiles share the same model file -> dedupe message shown.
    assert "[startup] models   : loaded 1 into memory (shared: 1 transcriber, 2 profiles)" in captured.err
    assert "[startup] sample   :" in captured.err
    assert "[startup] silence  :" in captured.err
    assert "[startup] inject   :" in captured.err


def test_run_banner_shows_config_source_when_file_loaded(monkeypatch, capsys, tmp_path):
    from speakinput.app import App

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config()
    config_path = Path("/Users/me/Library/Application Support/speakinput/config.toml")
    app = App(
        config=config,
        recorder=MagicMock(),
        injector=MagicMock(),
        feedback=MagicMock(),
        config_source=config_path,
    )
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    assert f"[startup] config   : {config_path}" in captured.err
    assert "(defaults" not in captured.err


def test_run_banner_single_profile_no_dedupe_message(monkeypatch, capsys):
    """With no secondary profile the dedupe line should NOT mention
    sharing — there's only one profile, sharing is impossible."""
    from speakinput.app import App

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config()  # no secondary
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    assert "[startup] profile 1" in captured.err
    assert "profile 2" not in captured.err
    assert "shared" not in captured.err


def test_run_banner_prompt_off_when_initial_prompt_empty(monkeypatch, capsys):
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(primary=Profile(key="alt_r", initial_prompt=""))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    assert "prompt=off" in captured.err


def test_run_passes_initial_prompt_to_transcriber(monkeypatch):
    """`profile.primary.initial_prompt` from config must reach the
    WhisperCppTranscriber constructor for that profile."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(primary=Profile(key="alt_r", initial_prompt="kubectl apply -f deployment.yaml"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    fake_model_cls.assert_called_once()
    assert (
        fake_model_cls.call_args.kwargs["initial_prompt"]
        == "kubectl apply -f deployment.yaml"
    )


# --- auto-stop watchdog --------------------------------------------------


def test_auto_stop_disabled_when_seconds_zero(capsys):
    """auto_stop_seconds=0 must NOT start a watchdog — preserves the
    old "release the key yourself" behavior."""
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0.005, auto_stop_seconds=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.zeros(16000, dtype=np.float32)
    transcriber = MagicMock()
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app.on_hotkey_press(config.primary)
    assert app._watchdog is None  # watchdog never started
    app.on_hotkey_release(config.primary)


def test_auto_stop_watchdog_started_when_enabled():
    """With auto_stop_seconds > 0, on_hotkey_press creates a watchdog."""
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0.005, auto_stop_seconds=0.2))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.5  # loud -> won't trigger
    recorder.drain.return_value = np.zeros(16000, dtype=np.float32)
    transcriber = MagicMock()
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app.on_hotkey_press(config.primary)
    try:
        assert app._watchdog is not None
    finally:
        app.on_hotkey_release(config.primary)


def test_auto_stop_watchdog_cleared_on_release():
    """The watchdog reference is cleared after release so a fresh press
    gets a fresh watchdog."""
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0.005, auto_stop_seconds=0.2))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.5
    recorder.drain.return_value = np.zeros(16000, dtype=np.float32)
    transcriber = MagicMock()
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)
    assert app._watchdog is None


def test_auto_stop_watchdog_fires_after_silence_window(capsys):
    """End-to-end: with auto_stop_seconds=0.1 and current_rms() returning
    silence, the watchdog should call on_hotkey_release within ~0.2s.
    The transcriber then runs as if the user had released the key."""
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0.005, auto_stop_seconds=0.1))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0  # pure silence from the start
    recorder.drain.return_value = np.full(16000, 0.5, dtype=np.float32)  # 1s of "loud" so gate passes
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "watchdog fired"
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    # Press but DO NOT release manually — the watchdog should.
    app.on_hotkey_press(config.primary)
    # Wait for the watchdog to fire (auto-stop=0.1s + a poll slack).
    deadline = time.monotonic() + 1.0
    while transcriber.transcribe.call_count == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    transcriber.transcribe.assert_called_once()
    # Injector was called with the transcriber's result.
    app.injector.inject.assert_called_once_with("watchdog fired")
    # Cleanup: tell the watchdog to stop (it should already be done, but
    # make the test self-contained).
    if app._watchdog is not None:
        app._watchdog.stop()


def test_auto_stop_disabled_when_silence_threshold_zero():
    """If silence_threshold=0 the watchdog is meaningless (no audio is
    ever 'silent'). Don't start one — saves the polling thread."""
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0.5))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.zeros(16000, dtype=np.float32)
    transcriber = MagicMock()
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app.on_hotkey_press(config.primary)
    try:
        assert app._watchdog is None
    finally:
        app.on_hotkey_release(config.primary)


# --- trailing-silence trim -----------------------------------------------


def test_release_trims_trailing_silence(capsys):
    """When the audio buffer has a silent tail, the release path must
    trim it down before computing the silence-gate RMS and transcribing."""
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0.005, auto_stop_seconds=0))
    sr = config.audio.sample_rate
    # 100ms of loud (RMS 0.5) followed by 200ms of silence.
    audio = np.concatenate(
        [np.full(sr // 10, 0.5, dtype=np.float32), np.zeros(sr // 5, dtype=np.float32)]
    )
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = audio
    transcriber = MagicMock()
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
        debug=True,
    )
    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)

    # Trim log line printed.
    captured = capsys.readouterr()
    assert "trimmed trailing silence" in captured.err
    # The transcriber received a buffer that is SHORTER than the original.
    call_audio = transcriber.transcribe.call_args.args[0]
    assert call_audio.size < audio.size


def test_release_does_not_trim_when_buffer_is_pure_speech(capsys):
    """A buffer of pure loud audio should reach whisper unchanged."""
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0.005, auto_stop_seconds=0))
    sr = config.audio.sample_rate
    audio = np.full(sr // 5, 0.5, dtype=np.float32)  # 200ms loud
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = audio
    transcriber = MagicMock()
    transcribers = {config.primary.key: transcriber}

    app = App(
        config=config,
        recorder=recorder,
        transcribers=transcribers,
        injector=MagicMock(),
        feedback=MagicMock(),
        debug=True,
    )
    app.on_hotkey_press(config.primary)
    app.on_hotkey_release(config.primary)

    captured = capsys.readouterr()
    assert "trimmed trailing silence" not in captured.err


# --- banner shows auto-stop --------------------------------------------


def test_run_banner_shows_auto_stop_seconds(monkeypatch, capsys):
    """The startup banner must show the auto_stop_seconds value so the
    user can verify the feature is configured as expected."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(primary=Profile(key="alt_r"))
    config = config.with_overrides(auto_stop_seconds=1.2)
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    assert "auto-stop after 1.2s" in captured.err


def test_run_banner_shows_auto_stop_off(monkeypatch, capsys):
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(primary=Profile(key="alt_r"))
    config = config.with_overrides(auto_stop_seconds=0)
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    assert "auto-stop after off" in captured.err


def test_run_banner_shows_continuity_window(monkeypatch, capsys):
    """The startup banner must show the prev_clip_window_seconds value
    so the user can verify the across-press hint is on and at what
    duration. 0 must show as 'off'."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)

    config = Config(primary=Profile(key="alt_r"))
    config = config.with_overrides(prev_clip_window_seconds=30)
    app = App(
        config=config,
        recorder=MagicMock(),
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app._shutdown.set()
    app.run()
    captured = capsys.readouterr()
    assert "across-press hint within 30s" in captured.err
    assert "within-press always on" in captured.err

    # 0 must read as 'off' so the user can tell at a glance that the
    # across-press hint is disabled.
    config2 = Config(primary=Profile(key="alt_r"))
    config2 = config2.with_overrides(prev_clip_window_seconds=0)
    app2 = App(
        config=config2,
        recorder=MagicMock(),
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app2._shutdown.set()
    app2.run()
    captured2 = capsys.readouterr()
    assert "across-press hint within off" in captured2.err


# --- chunked auto-stop: re-arm between chunks -----------------------------


def test_watchdog_chunk_drains_and_injects_then_rearms():
    """When the watchdog fires mid-press (user still holding the key),
    the captured audio is drained+transcribed+injected, and a fresh
    watchdog is armed for the next sentence. The recorder is NOT torn
    down — only its buffer is cleared."""
    from speakinput.app import App
    from speakinput.config import AudioConfig

    config = Config(audio=AudioConfig(auto_stop_seconds=0.1, silence_threshold=0.005))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    chunk_audio = np.full(16000, 0.3, dtype=np.float32)  # 1s of "speech"
    recorder.drain.return_value = chunk_audio
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "first sentence"
    injector = MagicMock()
    feedback = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: transcriber},
        injector=injector,
        feedback=feedback,
    )
    # Simulate a press so the busy lock is held.
    app._busy.acquire()
    # Direct call to the watchdog's on_trigger: this is what
    # SilenceWatchdog would do after seeing the auto-stop window of
    # silence.
    app._on_watchdog_chunk(app.config.primary)

    # The chunk was drained (NOT stopped) and injected.
    recorder.drain.assert_called_once()
    recorder.stop.assert_not_called()
    recorder.close.assert_not_called()
    transcriber.transcribe.assert_called_once()
    injector.inject.assert_called_once_with("first sentence")
    # A fresh watchdog was armed for the next sentence.
    assert app._watchdog is not None
    # The busy lock is still held — we're in a multi-chunk press.
    assert app._busy.locked()


def test_watchdog_chunk_rearms_using_configured_threshold():
    """The re-armed watchdog must use the configured silence_threshold
    and auto_stop_seconds, not stale values."""
    from speakinput.app import App
    from speakinput.config import AudioConfig

    config = Config(audio=AudioConfig(auto_stop_seconds=0.3, silence_threshold=0.02))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.zeros(0, dtype=np.float32)
    transcriber = MagicMock()
    injector = MagicMock()
    feedback = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: transcriber},
        injector=injector,
        feedback=feedback,
    )
    app._busy.acquire()
    app._on_watchdog_chunk(app.config.primary)
    wd = app._watchdog
    assert wd is not None
    assert wd._threshold == 0.02
    assert wd._auto_stop_seconds == 0.3


def test_watchdog_chunk_empty_audio_does_not_inject_but_rearms():
    """If the auto-stop fires during a buffer that was actually
    silence, the silence gate skips the transcribe+inject. We still
    re-arm so the next sentence gets caught."""
    from speakinput.app import App
    from speakinput.config import AudioConfig

    config = Config(audio=AudioConfig(auto_stop_seconds=0.1, silence_threshold=0.005))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.zeros(0, dtype=np.float32)
    transcriber = MagicMock()
    injector = MagicMock()
    feedback = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: transcriber},
        injector=injector,
        feedback=feedback,
    )
    app._busy.acquire()
    app._on_watchdog_chunk(app.config.primary)

    transcriber.transcribe.assert_not_called()
    injector.inject.assert_not_called()
    # But the watchdog was re-armed.
    assert app._watchdog is not None


def test_manual_release_during_chunked_session_finalizes():
    """If the user releases the hotkey during a chunked session, the
    manual release path finalizes — no fresh watchdog, busy lock
    released, recorder closed. No chunk body is running concurrently
    in this test; the test exercises the manual-release path on its
    own. The race-with-chunk-body case is covered by
    `test_watchdog_chunk_bails_out_when_manual_release_already_pending`."""
    from speakinput.app import App
    from speakinput.config import AudioConfig

    config = Config(audio=AudioConfig(auto_stop_seconds=0.1, silence_threshold=0.005))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(16000, 0.3, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "goodbye"
    injector = MagicMock()
    feedback = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: transcriber},
        injector=injector,
        feedback=feedback,
    )
    # Simulate an in-flight press.
    app._busy.acquire()
    app._active_profile = app.config.primary
    app._press_started_at = 0.0

    app.on_hotkey_release(app.config.primary)

    # The recorder is closed (final path), the busy lock is released.
    recorder.close.assert_called_once()
    recorder.drain.assert_called_once()
    assert not app._busy.locked()
    assert app._watchdog is None
    assert app._active_profile is None
    feedback.set_state.assert_any_call("idle")
    # The drained audio was transcribed+injected.
    transcriber.transcribe.assert_called_once()
    injector.inject.assert_called_once_with("goodbye")


def test_watchdog_chunk_bails_out_when_manual_release_already_pending():
    """If the user has already released (manual flag set) by the time
    the chunked body runs, the chunk body should NOT re-arm a new
    watchdog and NOT re-inject. The finalize is the manual release's
    job."""
    from speakinput.app import App
    from speakinput.config import AudioConfig

    config = Config(audio=AudioConfig(auto_stop_seconds=0.1, silence_threshold=0.005))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(8000, 0.3, dtype=np.float32)
    transcriber = MagicMock()
    injector = MagicMock()
    feedback = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: transcriber},
        injector=injector,
        feedback=feedback,
    )
    app._busy.acquire()
    # Simulate manual release having already been called: flag is set,
    # but the chunk body is still running (e.g. delayed watchdog).
    app._manual_release_pending = True
    app._on_watchdog_chunk(app.config.primary)

    # The chunk body bailed: no fresh watchdog was armed.
    assert app._watchdog is None
    # No transcribe was called by the chunk path (the manual release
    # path's _finalize will handle it).
    transcriber.transcribe.assert_not_called()
    injector.inject.assert_not_called()


def test_watchdog_chunk_does_not_rearm_if_recorder_closed():
    """If the recorder is closed out from under the chunk body
    (e.g. shutdown), the chunk body must not try to arm a watchdog
    on a closed recorder."""
    from speakinput.app import App
    from speakinput.config import AudioConfig

    config = Config(audio=AudioConfig(auto_stop_seconds=0.1, silence_threshold=0.005))
    recorder = MagicMock()
    # The chunk body itself calls is_recording exactly once, at the
    # re-arm gate. We want that one call to return False. Earlier
    # incidental calls (none in the chunk body itself, but the
    # watchdog's poll loop may run) should also be fine returning
    # False — a closed recorder is closed.
    recorder.is_recording.return_value = False
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(8000, 0.3, dtype=np.float32)
    transcriber = MagicMock()
    injector = MagicMock()
    feedback = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: transcriber},
        injector=injector,
        feedback=feedback,
    )
    app._busy.acquire()

    app._on_watchdog_chunk(app.config.primary)

    # No re-arm because is_recording was False.
    assert app._watchdog is None


def test_auto_stop_off_single_release_still_works(capsys):
    """Sanity: when auto_stop_seconds = 0, behavior is unchanged from
    before the chunked feature — one drain+process per press."""
    from speakinput.app import App
    from speakinput.config import AudioConfig

    config = Config(audio=AudioConfig(auto_stop_seconds=0, silence_threshold=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(16000, 0.3, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "hello"
    injector = MagicMock()
    feedback = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: transcriber},
        injector=injector,
        feedback=feedback,
    )

    app.on_hotkey_press(app.config.primary)
    assert app._watchdog is None  # feature off
    app.on_hotkey_release(app.config.primary)
    # Single drain+process path. The close+release happened.
    recorder.drain.assert_called_once()
    recorder.close.assert_called_once()
    assert not app._busy.locked()
    injector.inject.assert_called_once_with("hello")


# --- continuity hint (previous clip as initial_prompt) -------------------


def _build_app_with_loud_audio(debug: bool = False, **audio_overrides):
    """Helper: App whose recorder returns loud audio on drain.

    Used by the continuity tests so the silence gate doesn't short-
    circuit the transcribe path. Any kwargs are forwarded to
    AudioConfig (e.g. `prev_clip_window_seconds=0` to disable the
    across-press hint while keeping the within-press one)."""
    from speakinput.app import App

    config = Config(
        audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0, **audio_overrides)
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
    return app, recorder, transcriber


def test_continuity_no_hint_on_first_press():
    """The very first press has no previous clip — the transcriber
    receives only the configured lexical bias (the profile's
    `initial_prompt`) as the per-call prompt. No across-press text,
    no within-press chunk text."""
    app, _, transcriber = _build_app_with_loud_audio()
    transcriber.transcribe.return_value = "first sentence"
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    kwargs = transcriber.transcribe.call_args.kwargs
    prompt = kwargs.get("initial_prompt") or ""
    # The configured lexical bias is in the prompt (it's the default
    # embedded-software-engineer vocabulary). The last 200 chars of
    # the prompt end with "...printf, sprintf, malloc, free, memcpy,
    # memset, strlen" — assert on that tail to avoid the
    # per-component truncation interacting with the test.
    assert "printf" in prompt
    assert "strlen" in prompt
    # But no transcript-y content has leaked in from a previous clip.
    assert "first sentence" not in prompt


def test_continuity_within_press_chunks_use_previous_text():
    """Two auto-stopped chunks in the same press: the second chunk
    must include the first chunk's text in the initial_prompt."""
    from speakinput.app import App

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0.05))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(16000, 0.3, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.side_effect = ["first sentence", "second sentence"]
    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: transcriber},
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app.on_hotkey_press(app.config.primary)
    # Simulate the watchdog firing once (mid-press auto-stop) without
    # going through the real SilenceWatchdog timing.
    app._on_watchdog_chunk(config.primary)
    # Then the user releases the key.
    app.on_hotkey_release(app.config.primary)
    assert transcriber.transcribe.call_count == 2
    # First call: configured lexical bias is in the prompt, but no
    # previous chunk text yet (within-press reset on press).
    first_prompt = transcriber.transcribe.call_args_list[0].kwargs.get("initial_prompt") or ""
    assert "first sentence" not in first_prompt
    # Second call: should include "first sentence" as the within-press
    # hint.
    second_prompt = transcriber.transcribe.call_args_list[1].kwargs.get("initial_prompt") or ""
    assert "first sentence" in second_prompt


def test_continuity_across_press_uses_previous_clip_within_window():
    """Press 1 finishes. Press 2 starts within `prev_clip_window_seconds`.
    Press 2's first transcribe must include press 1's text as a hint."""
    app, _, transcriber = _build_app_with_loud_audio(
        prev_clip_window_seconds=60
    )
    transcriber.transcribe.side_effect = ["alpha bravo", "charlie delta"]
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    # Second press — same App, so `_last_clip_text` is in scope.
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    second_kwargs = transcriber.transcribe.call_args_list[1].kwargs
    assert "alpha bravo" in (second_kwargs.get("initial_prompt") or "")


def test_continuity_across_press_skipped_when_window_expired():
    """If the gap between presses exceeds `prev_clip_window_seconds`,
    the previous clip must NOT be used as a hint."""
    app, _, transcriber = _build_app_with_loud_audio(
        prev_clip_window_seconds=10
    )
    transcriber.transcribe.side_effect = ["alpha bravo", "charlie delta"]
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    # Backdate the recorded clip to a time well outside the window.
    app._last_clip_at = app._last_clip_at - 100.0
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    second_kwargs = transcriber.transcribe.call_args_list[1].kwargs
    prompt = second_kwargs.get("initial_prompt") or ""
    assert "alpha bravo" not in prompt


def test_continuity_across_press_skipped_when_window_zero():
    """`prev_clip_window_seconds=0` disables the across-press hint
    entirely (within-press still works)."""
    app, _, transcriber = _build_app_with_loud_audio(
        prev_clip_window_seconds=0
    )
    transcriber.transcribe.side_effect = ["alpha bravo", "charlie delta"]
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    second_kwargs = transcriber.transcribe.call_args_list[1].kwargs
    prompt = second_kwargs.get("initial_prompt") or ""
    assert "alpha bravo" not in prompt


def test_continuity_within_press_overrides_across_press_chunk():
    """When a press starts with a within-press chunk already recorded,
    the most recent (chunk) text wins over the older across-press
    clip. Both should appear in the prompt, but the chunk is the
    final (most-recent) component."""
    from speakinput.app import App

    config = Config(
        audio=AudioConfig(
            silence_threshold=0,
            auto_stop_seconds=0.05,
            prev_clip_window_seconds=60,
        )
    )
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.current_rms.return_value = 0.0
    recorder.drain.return_value = np.full(16000, 0.3, dtype=np.float32)
    transcriber = MagicMock()
    # press 1 → "earlier clip"; press 2 chunk 1 → "mid sentence";
    # press 2 chunk 2 (final release) → "continuing thought"
    transcriber.transcribe.side_effect = [
        "earlier clip",
        "mid sentence",
        "continuing thought",
    ]
    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: transcriber},
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    # Press 2 — first chunk (mid-press auto-stop)
    app.on_hotkey_press(app.config.primary)
    app._on_watchdog_chunk(config.primary)
    # Then the user releases — the final-release path transcribes
    # whatever audio is buffered. That's the chunk whose prompt
    # should contain BOTH the across-press hint ("earlier clip") and
    # the within-press chunk ("mid sentence").
    app.on_hotkey_release(app.config.primary)
    assert transcriber.transcribe.call_count == 3
    third_prompt = transcriber.transcribe.call_args_list[2].kwargs.get("initial_prompt") or ""
    assert "earlier clip" in third_prompt
    assert "mid sentence" in third_prompt
    # "mid sentence" (within-press, last in the concatenation order)
    # should come after "earlier clip" (across-press).
    assert third_prompt.index("earlier clip") < third_prompt.index("mid sentence")


def test_continuity_does_not_update_state_on_empty_transcript():
    """If whisper returns an empty string (silence-gated, hallucination
    etc.), we must NOT update `_last_clip_text` — otherwise the next
    press's prompt would include a stale or empty clip."""
    app, _, transcriber = _build_app_with_loud_audio()
    transcriber.transcribe.return_value = ""
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    assert app._last_clip_text == ""


def test_continuity_prompt_caps_total_length():
    """A long previous clip must be truncated so the assembled prompt
    stays under whisper's 224-token cap. We verify the total prompt
    length is bounded by the cap and the within-press chunk is the
    one dropped first if everything overflows."""
    app, _, transcriber = _build_app_with_loud_audio(
        prev_clip_window_seconds=60
    )
    long_text = "x" * 2000
    transcriber.transcribe.side_effect = [long_text, "second"]
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    app.on_hotkey_press(app.config.primary)
    app.on_hotkey_release(app.config.primary)
    second_kwargs = transcriber.transcribe.call_args_list[1].kwargs
    prompt = second_kwargs.get("initial_prompt") or ""
    # Cap is 400 chars; the long_text must be truncated AND the total
    # must be under the cap.
    assert len(prompt) <= 400


# --- shutdown hardening: liveness, heartbeat, signal handling -------------


def test_liveness_watcher_fires_on_dead_listener():
    """If a listener's underlying thread dies (e.g. pynput's CGEventTap
    was disabled by macOS on sleep/wake), the watcher must call
    `on_dead` exactly once with the listener's key."""
    from speakinput.app import _LivenessWatcher

    on_dead = MagicMock()
    listener = MagicMock()
    listener._key = "alt_r"
    listener.is_running.return_value = False
    listener._thread = MagicMock()
    listener._thread.is_alive.return_value = False

    watcher = _LivenessWatcher(
        listeners=[listener], interval_s=0.01, on_dead=on_dead
    )
    watcher.start()
    try:
        # 0.01s poll × ~5 iterations is well under a second.
        time.sleep(0.1)
    finally:
        watcher.stop()
    # `on_dead` was invoked at least once; the watcher polls and emits
    # every poll after death, so we just check it was called.
    assert on_dead.called
    on_dead.assert_called_with("alt_r")


def test_run_emits_unconditional_warning_on_dead_listener(capsys, monkeypatch):
    """A dead listener must trigger a user-visible warning on stderr,
    NOT just a debug line. The user needs to know the hotkey is dead
    even when running without -d."""
    from speakinput.app import App

    fake_ensure = MagicMock(return_value="/r/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    app = App(
        config=config,
        recorder=MagicMock(),
        transcribers={config.primary.key: MagicMock()},
        injector=MagicMock(),
        feedback=MagicMock(),
        debug=False,  # not in debug mode
    )
    # Manually install a dead listener and a watcher that polls fast.
    from speakinput.app import _LivenessWatcher
    listener = MagicMock()
    listener._key = "alt_r"
    listener.is_running.return_value = False
    listener._thread = MagicMock()
    listener._thread.is_alive.return_value = False
    app.listeners = {"alt_r": listener}
    watcher = _LivenessWatcher(
        listeners=[listener], interval_s=0.01, on_dead=lambda k: print(
            f"[warn] hotkey listener for {k!r} is no longer alive — "
            f"the push-to-talk key will not respond. "
            f"Restart speakinput to recover.",
            file=__import__("sys").stderr,
            flush=True,
        ),
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
    captured = capsys.readouterr()
    assert "hotkey listener for 'alt_r' is no longer alive" in captured.err
    # And it must be visible WITHOUT the [debug] prefix.
    assert "[debug] hotkey listener" not in captured.err


def test_liveness_watcher_does_not_fire_when_alive():
    """A healthy listener (thread alive, is_running True) must not
    trigger `on_dead`."""
    from speakinput.app import _LivenessWatcher

    on_dead = MagicMock()
    listener = MagicMock()
    listener._key = "alt_r"
    listener.is_running.return_value = True
    listener._thread = MagicMock()
    listener._thread.is_alive.return_value = True

    watcher = _LivenessWatcher(
        listeners=[listener], interval_s=0.01, on_dead=on_dead
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
    on_dead.assert_not_called()


def test_liveness_watcher_works_for_pynput_listener_without_thread_attr():
    """Regression test: the watcher previously did
    `is_running() and listener._thread.is_alive()`, which always
    evaluated to False for the pynput-backed `HotkeyListener` (it has
    no `_thread` attribute — pynput uses `_listener`). On macOS this
    caused a false-positive "listener is no longer alive" warning 5s
    after every start, even when the listener was perfectly healthy.

    The fix is to use `is_running()` alone — both backends implement
    it correctly. This test simulates a pynput listener (no `_thread`,
    `is_running()` returns True) and asserts the watcher stays quiet.
    """
    from speakinput.app import _LivenessWatcher

    on_dead = MagicMock()
    listener = MagicMock(spec=["_key", "is_running"])  # no _thread
    listener._key = "alt_r"
    listener.is_running.return_value = True

    watcher = _LivenessWatcher(
        listeners=[listener], interval_s=0.01, on_dead=on_dead
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
    on_dead.assert_not_called()


def test_liveness_watcher_fires_for_pynput_listener_when_dead():
    """Counterpart to the previous test: when the pynput listener's
    `is_running()` returns False (e.g. pynput's CGEventTap was
    disabled by macOS on sleep/wake), the watcher must still call
    `on_dead`. Catches the case where the liveness check is
    accidentally gated on `_thread` and never reports pynput deaths.
    """
    from speakinput.app import _LivenessWatcher

    on_dead = MagicMock()
    listener = MagicMock(spec=["_key", "is_running"])
    listener._key = "alt_r"
    listener.is_running.return_value = False

    watcher = _LivenessWatcher(
        listeners=[listener], interval_s=0.01, on_dead=on_dead
    )
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
    on_dead.assert_called_with("alt_r")


def test_heartbeat_emits_periodically_in_debug(capsys):
    """The heartbeat prints a 'still alive' line every interval when
    debug mode is on. Used to confirm the process is alive after a
    long sleep / idle period."""
    from speakinput.app import _Heartbeat

    hb = _Heartbeat(interval_s=0.05)
    hb.start()
    try:
        time.sleep(0.2)
    finally:
        hb.stop()
    captured = capsys.readouterr()
    # At least one heartbeat line in 0.2s with a 0.05s interval.
    assert "heartbeat" in captured.err
    assert "uptime=" in captured.err


def test_run_installs_sighup_handler(monkeypatch):
    """`App.run()` must install a SIGHUP handler so terminal close
    triggers a clean shutdown. Without it, closing the terminal would
    SIGKILL the process mid-shutdown, leaving the single-instance
    lock orphaned and breaking the next start.sh."""
    from speakinput.app import App
    import signal as _sig_module

    fake_ensure = MagicMock(return_value="/r/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)

    installed: dict[int, object] = {}
    # Bind the real signal.signal at import time (before the monkeypatch
    # below) so the fake_signal below can call it without recursing.
    real_signal_signal = _sig_module.signal

    def fake_signal(signum, handler):
        if hasattr(_sig_module, "SIGHUP") and signum == _sig_module.SIGHUP:
            installed[signum] = handler
        return real_signal_signal(signum, handler)

    monkeypatch.setattr("signal.signal", fake_signal)

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    app = App(
        config=config,
        recorder=MagicMock(),
        transcribers={config.primary.key: MagicMock()},
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app._shutdown.set()
    app.run()

    # SIGHUP must be installed AND must set the shutdown event.
    assert _sig_module.SIGHUP in installed
    installed[_sig_module.SIGHUP](_sig_module.SIGHUP, None)
    assert app._shutdown.is_set()


def test_shutdown_handles_hung_media_resume():
    """If `media_controller.resume()` blocks (e.g. osascript is wedged),
    `shutdown()` must not hang the process. The user has been waiting
    to exit; logging and moving on is the right behavior."""
    app, _, _, _, _ = _build_app()
    # Media controller exists by default (pause_media=True). Patch it
    # to a mock that raises — we want to confirm shutdown tolerates it.
    media = MagicMock()
    media.resume.side_effect = RuntimeError("osascript wedged")
    app.media_controller = media
    # Watchdog is None; that's fine.
    # The real test: shutdown() must complete without raising and
    # must still set _shutdown and stop the listeners.
    app.shutdown()
    assert app._shutdown.is_set()
    for listener in app.listeners.values():
        listener.stop.assert_called()


def test_shutdown_is_idempotent():
    """`shutdown()` is safe to call twice — used by the signal handler
    path (sets event) and the `finally` block. Also called by
    `cli.main()`'s `KeyboardInterrupt` handler."""
    app, _, _, _, _ = _build_app()
    app.shutdown()
    # Second call must not raise, even though all the resources are
    # already torn down.
    app.shutdown()
    assert app._shutdown.is_set()


# --- event worker: hotkey bodies run off the OS event-tap thread ----------


def test_hotkey_callbacks_run_on_worker_thread_not_caller():
    """The press callback handed to the listener must NOT run the press
    body on the calling (event-tap) thread. macOS disables CGEventTaps
    whose callback runs too long, and a release body takes seconds
    (drain → transcribe → inject). The callback must enqueue and return;
    the body runs on the app's worker thread."""
    import threading

    app, recorder, _, _, _ = _build_app()
    seen: dict[str, int] = {}

    real_press = app.on_hotkey_press

    def _recording_press(profile):
        seen["ident"] = threading.get_ident()
        return real_press(profile)

    app.on_hotkey_press = _recording_press  # type: ignore[method-assign]
    app._start_event_worker()
    try:
        cb = app._make_press_cb(app.config.primary)
        main_ident = threading.get_ident()
        cb()  # must return immediately; body runs on the worker
        deadline = time.monotonic() + 2.0
        while "ident" not in seen and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "ident" in seen, "press body never ran"
        assert seen["ident"] != main_ident
        # The body really ran: the recorder was started.
        recorder.start.assert_called_once()
    finally:
        app._stop_event_worker()


def test_event_worker_survives_handler_exception():
    """A raising handler must not kill the worker thread — a dead worker
    with live listeners would be the same silent 'hotkey does nothing'
    failure the worker exists to prevent."""
    app, _, _, _, _ = _build_app()
    ran = []

    def _boom():
        raise RuntimeError("handler bug")

    app._start_event_worker()
    try:
        app._enqueue_event(_boom)
        app._enqueue_event(lambda: ran.append("second"))
        deadline = time.monotonic() + 2.0
        while not ran and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ran == ["second"]
    finally:
        app._stop_event_worker()


def test_press_release_order_preserved_through_queue():
    """Press then release enqueued back-to-back must execute in order:
    release sees a recording session and finalizes it."""
    app, recorder, transcribers, injector, _ = _build_app()
    app._start_event_worker()
    try:
        app._make_press_cb(app.config.primary)()
        app._make_release_cb(app.config.primary)()
        # Generous deadline: the press path shells out to the media
        # backend (osascript on macOS), which can take >1s on a cold
        # start. The queue ordering is what we assert, not speed.
        deadline = time.monotonic() + 5.0
        while injector.inject.call_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        injector.inject.assert_called_once_with("hello world")
        transcribers[app.config.primary.key].transcribe.assert_called_once()
        # Lock released at the end of finalize; a new press is accepted.
        assert not app._busy.locked()
    finally:
        app._stop_event_worker()


# --- escalating SIGINT -----------------------------------------------------


def test_sigint_escalates_on_second_press(monkeypatch, capsys):
    """First Ctrl-C: graceful shutdown (set the event). Second Ctrl-C:
    dump thread stacks and force-exit. This is the escape hatch for a
    wedged teardown — without it, further Ctrl-Cs are silent no-ops and
    the user has to kill -9."""
    import signal as _sig_module

    from speakinput.app import App

    fake_ensure = MagicMock(return_value="/r/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)

    installed: dict[int, object] = {}
    real_signal_signal = _sig_module.signal

    def fake_signal(signum, handler):
        if signum == _sig_module.SIGINT:
            installed[signum] = handler
            return None
        return real_signal_signal(signum, handler)

    monkeypatch.setattr("signal.signal", fake_signal)
    fake_exit = MagicMock()
    monkeypatch.setattr("os._exit", fake_exit)
    import faulthandler

    fake_dump = MagicMock()
    monkeypatch.setattr(faulthandler, "dump_traceback", fake_dump)

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    app = App(
        config=config,
        recorder=MagicMock(),
        transcribers={config.primary.key: MagicMock()},
        injector=MagicMock(),
        feedback=MagicMock(),
    )
    app._shutdown.set()
    app.run()

    handler = installed[_sig_module.SIGINT]
    # First press on an already-set event? Reset to simulate a fresh run.
    app._shutdown.clear()
    handler(_sig_module.SIGINT, None)
    assert app._shutdown.is_set()
    fake_exit.assert_not_called()
    # Second press: force exit with a stack dump.
    handler(_sig_module.SIGINT, None)
    fake_dump.assert_called_once()
    fake_exit.assert_called_once_with(2)
    captured = capsys.readouterr()
    assert "Ctrl-C" in captured.err


# --- liveness watcher: sleep detection + swap ------------------------------


def test_liveness_watcher_detects_sleep_via_clock_skew(monkeypatch):
    """After a suspend, time.monotonic has barely moved but the wall
    clock has jumped. The watcher must call on_sleep with the skew."""
    from speakinput.app import _LivenessWatcher

    state = {"mono": 1000.0, "wall": 5000.0}
    monkeypatch.setattr(time, "monotonic", lambda: state["mono"])
    monkeypatch.setattr(time, "time", lambda: state["wall"])

    on_sleep = MagicMock()
    watcher = _LivenessWatcher(
        listeners=[], interval_s=60.0, on_dead=MagicMock(), on_sleep=on_sleep
    )
    # First call just initializes the baseline.
    watcher._check_sleep()
    # Normal tick: both clocks advance together — no sleep.
    state["mono"] += 5.0
    state["wall"] += 5.0
    watcher._check_sleep()
    on_sleep.assert_not_called()
    # Machine slept for ~100s: monotonic advanced one tick, wall jumped.
    state["mono"] += 5.0
    state["wall"] += 105.0
    watcher._check_sleep()
    on_sleep.assert_called_once()
    slept = on_sleep.call_args[0][0]
    assert 99.0 < slept <= 100.0


def test_liveness_watcher_no_sleep_when_clocks_track(monkeypatch):
    from speakinput.app import _LivenessWatcher

    state = {"mono": 0.0, "wall": 100.0}
    monkeypatch.setattr(time, "monotonic", lambda: state["mono"])
    monkeypatch.setattr(time, "time", lambda: state["wall"])

    on_sleep = MagicMock()
    watcher = _LivenessWatcher(
        listeners=[], interval_s=60.0, on_dead=MagicMock(), on_sleep=on_sleep
    )
    watcher._check_sleep()
    for _ in range(5):
        state["mono"] += 5.0
        state["wall"] += 5.0
        watcher._check_sleep()
    on_sleep.assert_not_called()


def test_liveness_watcher_swap_replaces_and_resets_tracking():
    """swap() must put the new listener into the polled set and mark it
    alive, so its future death fires on_dead exactly once."""
    from speakinput.app import _LivenessWatcher

    on_dead = MagicMock()
    old = MagicMock(spec=["_key", "is_running"])
    old._key = "alt_r"
    old.is_running.return_value = False
    new = MagicMock(spec=["_key", "is_running"])
    new._key = "alt_r"
    new.is_running.return_value = True

    watcher = _LivenessWatcher(
        listeners=[old], interval_s=0.01, on_dead=on_dead
    )
    watcher.swap(old, new)
    watcher.start()
    try:
        time.sleep(0.1)
    finally:
        watcher.stop()
    # The dead `old` listener must NOT fire on_dead after the swap —
    # the watcher now polls the healthy replacement.
    on_dead.assert_not_called()
    assert watcher._listeners == [new]


# --- listener restart + stranded-press abort --------------------------------


def test_restart_listener_replaces_dead_listener():
    """A successful restart swaps the registry entry, stops the old
    listener, and informs the watcher (so it polls the new one)."""
    app, _, _, _, _ = _build_app()
    app._use_evdev = False
    old = MagicMock()
    app.listeners["alt_r"] = old
    watcher = MagicMock()
    app._liveness_watcher = watcher

    assert app._restart_listener("alt_r") is True
    new = app.listeners["alt_r"]
    assert new is not old
    new.start.assert_called_once()
    old.stop.assert_called_once()
    watcher.swap.assert_called_once_with(old, new)


def test_restart_listener_failure_keeps_old(monkeypatch):
    """If construction/start raises, the old listener stays registered
    so the liveness watcher's next tick retries the restart."""
    app, _, _, _, _ = _build_app()
    app._use_evdev = False
    old = MagicMock()
    app.listeners["alt_r"] = old
    monkeypatch.setattr(
        "speakinput.app.HotkeyListener",
        MagicMock(side_effect=RuntimeError("HIToolbox wedged")),
    )
    assert app._restart_listener("alt_r") is False
    assert app.listeners["alt_r"] is old
    old.stop.assert_not_called()


def test_on_listener_dead_restarts_and_enqueues_abort(capsys):
    """Dead listener → restart attempt + a stranded-press abort on the
    event queue (the release event died with the listener)."""
    app, _, _, _, _ = _build_app()
    app._use_evdev = False
    old = MagicMock()
    app.listeners["alt_r"] = old

    app._on_listener_dead("alt_r")
    assert app.listeners["alt_r"] is not old
    fn, args = app._work_q.get_nowait()
    assert fn == app._abort_press
    assert "alt_r" in args[0]
    captured = capsys.readouterr()
    assert "restarted" in captured.err


def test_on_listener_dead_warns_when_restart_fails(monkeypatch, capsys):
    """If the restart fails, the user must still get the old-style
    'restart speakinput' warning — that's the only way they learn the
    hotkey is dead."""
    app, _, _, _, _ = _build_app()
    app._use_evdev = False
    app.listeners["alt_r"] = MagicMock()
    monkeypatch.setattr(
        "speakinput.app.HotkeyListener",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    app._on_listener_dead("alt_r")
    captured = capsys.readouterr()
    assert "no longer alive" in captured.err
    assert "Restart speakinput" in captured.err


def test_on_listener_dead_backs_off_when_flapping(capsys):
    """A listener that dies again right after a restart (e.g. Input
    Monitoring permission revoked — the tap can never come up) must NOT
    be restarted a second time within the backoff window: that would
    flap die→restart→die forever and bury the one message the user
    needs ('fix your permissions, then restart speakinput')."""
    app, _, _, _, _ = _build_app()
    app._use_evdev = False
    old = MagicMock()
    app.listeners["alt_r"] = old
    # Simulate a restart moments ago.
    app._listener_restart_at["alt_r"] = time.monotonic()

    app._on_listener_dead("alt_r")
    # No restart: the registry still holds the dead listener...
    assert app.listeners["alt_r"] is old
    # ...the user got the warning...
    captured = capsys.readouterr()
    assert "no longer alive" in captured.err
    # ...and the stranded-press abort still ran through the queue.
    fn, _ = app._work_q.get_nowait()
    assert fn == app._abort_press


def test_on_system_sleep_restarts_all_listeners_and_aborts(capsys):
    """Wake detection restarts every listener (macOS disables event
    taps across sleep even though the threads stay alive) and enqueues
    a stranded-press abort."""
    from speakinput.config import Profile

    config = Config(
        primary=Profile(key="alt_r"),
        secondary=Profile(key="cmd_r"),
        audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0),
    )
    t = MagicMock()
    t.transcribe.return_value = "x"
    app, _, _, _, _ = _build_app(
        config=config, transcribers={"alt_r": t, "cmd_r": t}
    )
    old_primary = MagicMock()
    old_secondary = MagicMock()
    app.listeners = {"alt_r": old_primary, "cmd_r": old_secondary}

    app._on_system_sleep(120.0)
    assert app.listeners["alt_r"] is not old_primary
    assert app.listeners["cmd_r"] is not old_secondary
    fn, args = app._work_q.get_nowait()
    assert fn == app._abort_press
    assert "wake" in args[0]
    captured = capsys.readouterr()
    assert "system slept" in captured.err


def test_abort_press_releases_busy_and_discards_audio():
    """A stranded press (release lost) must be torn down: busy lock
    released, recorder closed, buffer discarded — NOT transcribed."""
    app, recorder, transcribers, injector, feedback = _build_app()
    app.on_hotkey_press(app.config.primary)
    assert app._busy.locked()

    app._abort_press("test reason")
    assert not app._busy.locked()
    recorder.drain.assert_called()
    recorder.close.assert_called()
    # No transcribe/inject for the discarded buffer.
    transcribers[app.config.primary.key].transcribe.assert_not_called()
    injector.inject.assert_not_called()
    assert app._active_profile is None
    feedback.set_state.assert_called_with("idle")


def test_abort_press_noop_when_no_press_active():
    """Aborting with no active press must be a silent no-op (the wake
    path fires it unconditionally)."""
    app, recorder, _, _, _ = _build_app()
    app._abort_press("no press")
    recorder.close.assert_not_called()
    assert not app._busy.locked()


def test_abort_press_noop_when_release_in_progress():
    """While on_hotkey_release is finalizing (_manual_release_pending),
    the abort must not touch the session — the finalize path owns the
    lock and a double release would corrupt state."""
    app, recorder, _, _, _ = _build_app()
    app.on_hotkey_press(app.config.primary)
    app._manual_release_pending = True  # simulate finalize in flight
    app._abort_press("wake during finalize")
    assert app._busy.locked()  # still held — abort backed off
    # Cleanup: run the real finalize so the lock is released.
    app._manual_release_pending = False
    app._finalize(app.config.primary)
    assert not app._busy.locked()


# --- bounded media resume on shutdown ---------------------------------------


def test_shutdown_completes_when_media_resume_hangs():
    """A media backend that never returns (wedged osascript after
    sleep) must not hang shutdown: the bounded helper thread is joined
    with a timeout and shutdown moves on. Regression test for
    'Ctrl-C does not work, have to kill -9'."""
    import threading

    app, _, _, _, _ = _build_app()
    never = threading.Event()

    def _hung_resume():
        never.wait()  # blocks forever; daemon thread dies at exit

    media = MagicMock()
    media.resume.side_effect = _hung_resume
    app.media_controller = media

    start = time.monotonic()
    app.shutdown()
    elapsed = time.monotonic() - start
    assert app._shutdown.is_set()
    # 3s bounded join + slack. Without the bound this test would hang.
    assert elapsed < 10.0
    for listener in app.listeners.values():
        listener.stop.assert_called()


def test_shutdown_closes_recorder_stream():
    """shutdown() must close the recorder's PortAudio stream itself.

    sounddevice's atexit handler stops+closes any still-open stream
    during interpreter finalization (GIL held). If another thread is
    concurrently stopping the same stream, CoreAudio deadlocks on the
    HAL mutex while its IO thread waits for the GIL — the exact
    three-way deadlock sampled from a stuck production instance
    (main thread in Py_Finalize → Pa_CloseStream; hotkey thread in
    AudioOutputUnitStop; IO thread in startStopCallback). Closing in
    shutdown (serialized via the recorder's stream lock) means atexit
    finds nothing to close."""
    app, recorder, _, _, _ = _build_app()
    app.shutdown()
    recorder.close.assert_called()
