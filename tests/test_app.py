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


def test_run_uses_pynput_listener_on_x11(monkeypatch, capsys):
    """Counterpart to the Wayland test: when XDG_SESSION_TYPE is unset
    or x11, pynput's HotkeyListener is used and the banner doesn't
    mention evdev."""
    from speakinput.app import App
    from speakinput.config import Profile

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)

    config = Config(primary=Profile(key="alt_r"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    assert "evdev" not in captured.err
    import speakinput.app as appmod
    assert appmod.HotkeyListener.called
    assert not appmod.EvdevHotkeyListener.called


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
    assert "[startup] profile 1 : key=alt_r model=small language=auto prompt=set" in captured.err
    assert "[startup] profile 2 : key=cmd_r model=small language=zh prompt=set" in captured.err
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
