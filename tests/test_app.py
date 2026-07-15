"""Tests for the App orchestrator. Mocks audio + transcribers + injector."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from speakinput.config import AudioConfig, Config, Profile


@pytest.fixture(autouse=True)
def _stub_hotkey_listener(monkeypatch):
    """Replace HotkeyListener with a no-op mock for the whole module.

    pynput's keyboard.Listener.init crashes on this test environment
    (macOS HIToolbox) — see earlier commits — so every test that would
    start a real listener uses this stub. The class mock returns a
    fresh listener instance per construction so per-profile assertions
    work as expected.
    """
    fake_listener_cls = MagicMock()
    fake_listener_cls.side_effect = lambda *a, **kw: MagicMock()
    monkeypatch.setattr("speakinput.app.HotkeyListener", fake_listener_cls)
    return fake_listener_cls


def _build_app(debug: bool = False, dry_run: bool = False, config: Config | None = None,
               transcribers: dict | None = None):
    """Build an App with all I/O collaborators mocked out.

    `transcribers` defaults to a single mock keyed by the primary profile's
    hotkey; override it to simulate a multi-profile setup."""
    from speakinput.app import App

    config = config or Config(audio=AudioConfig(silence_threshold=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.stop.return_value = np.zeros(16000, dtype=np.float32)  # 1s of "silence"

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

    config = Config(audio=AudioConfig(silence_threshold=0.005))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
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

    config = Config(audio=AudioConfig(silence_threshold=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
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

    config = Config(audio=AudioConfig(silence_threshold=0.005))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.stop.return_value = np.full(16000, 0.5, dtype=np.float32)
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
    recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
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
        audio=AudioConfig(silence_threshold=0),
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
    recorder.stop.return_value = np.zeros(16000, dtype=np.float32)

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
        audio=AudioConfig(silence_threshold=0),
        primary=Profile(key="alt_r", model="small", language="en"),
        secondary=Profile(key="cmd_r", model="small", language="zh"),
    )
    primary_t = MagicMock()
    primary_t.transcribe.return_value = "english text"
    secondary_t = MagicMock()
    transcribers = {"alt_r": primary_t, "cmd_r": secondary_t}

    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.stop.return_value = np.zeros(16000, dtype=np.float32)

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
