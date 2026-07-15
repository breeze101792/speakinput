"""Tests for the App orchestrator. Mocks audio + transcriber + injector."""

from unittest.mock import MagicMock

import numpy as np
import pytest


def _build_app(debug: bool = False, dry_run: bool = False):
    """Build an App with all I/O collaborators mocked out."""
    from speakinput.app import App
    from speakinput.config import AudioConfig, Config

    # silence_threshold=0 so tests don't get blocked by the silence gate
    # even when they pass zero-RMS audio to the mock recorder.
    config = Config(audio=AudioConfig(silence_threshold=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.stop.return_value = np.zeros(16000, dtype=np.float32)  # 1s of "silence"

    transcriber = MagicMock()
    transcriber.transcribe.return_value = "hello world"

    injector = MagicMock()
    feedback = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcriber=transcriber,
        injector=injector,
        feedback=feedback,
        dry_run=dry_run,
        debug=debug,
    )
    return app, recorder, transcriber, injector, feedback


def test_press_calls_recorder_start_and_marks_listening(capsys):
    app, recorder, _, _, feedback = _build_app()
    app.on_hotkey_press()
    recorder.start.assert_called_once()
    feedback.set_state.assert_called_with("listening")
    # No debug output by default.
    captured = capsys.readouterr()
    assert "[debug]" not in captured.err


def test_press_ignored_when_already_busy(capsys):
    app, recorder, _, _, _ = _build_app()
    app.on_hotkey_press()  # first press acquires the lock
    recorder.start.reset_mock()
    app.on_hotkey_press()  # second press should be ignored
    recorder.start.assert_not_called()


def test_release_press_failed_means_nothing_to_do(capsys):
    """If on_hotkey_press failed (recorder never started), on_hotkey_release is a no-op."""
    app, recorder, transcriber, injector, _ = _build_app()
    # Make is_recording return False (as if start() had failed).
    recorder.is_recording.return_value = False
    app.on_hotkey_release()
    transcriber.transcribe.assert_not_called()
    injector.inject.assert_not_called()


def test_release_happy_path_calls_transcribe_and_inject(capsys):
    app, recorder, transcriber, injector, feedback = _build_app()
    app.on_hotkey_press()
    app.on_hotkey_release()
    transcriber.transcribe.assert_called_once()
    injector.inject.assert_called_once_with("hello world")
    # State should end back at idle.
    feedback.set_state.assert_any_call("idle")


def test_release_empty_transcript_does_not_inject(capsys):
    app, recorder, transcriber, injector, _ = _build_app()
    transcriber.transcribe.return_value = ""
    app.on_hotkey_press()
    app.on_hotkey_release()
    injector.inject.assert_not_called()


def test_release_silence_skips_transcribe(capsys):
    """If audio RMS is below the configured silence_threshold, we must NOT
    call the transcriber. Whisper hallucinates on near-empty audio; better
    to skip the call entirely and log a debug line so the user can see
    why nothing was typed."""
    from speakinput.app import App
    from speakinput.config import AudioConfig, Config

    config = Config(audio=AudioConfig(silence_threshold=0.005))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    # 1s of pure silence — RMS is 0.0, well below the threshold.
    recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
    transcriber = MagicMock()
    injector = MagicMock()

    app = App(
        config=config,
        recorder=recorder,
        transcriber=transcriber,
        injector=injector,
        feedback=MagicMock(),
        debug=True,
    )
    app.on_hotkey_press()
    app.on_hotkey_release()

    transcriber.transcribe.assert_not_called()
    injector.inject.assert_not_called()
    captured = capsys.readouterr()
    assert "silence gate" in captured.err
    assert "skipping transcribe" in captured.err


def test_release_silence_threshold_zero_disables_gate(capsys):
    """silence_threshold=0 must NOT short-circuit — even zero-RMS audio
    goes to the transcriber. This is the escape hatch for users who want
    whisper to see literally everything."""
    from speakinput.app import App
    from speakinput.config import AudioConfig, Config

    config = Config(audio=AudioConfig(silence_threshold=0))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = ""

    app = App(
        config=config,
        recorder=recorder,
        transcriber=transcriber,
        injector=MagicMock(),
        feedback=MagicMock(),
        debug=True,
    )
    app.on_hotkey_press()
    app.on_hotkey_release()

    transcriber.transcribe.assert_called_once()


def test_release_loud_audio_passes_gate(capsys):
    """Audio above the threshold must reach the transcriber."""
    from speakinput.app import App
    from speakinput.config import AudioConfig, Config

    config = Config(audio=AudioConfig(silence_threshold=0.005))
    recorder = MagicMock()
    recorder.is_recording.return_value = True
    # RMS ~ 0.5 (loud tone), well above the threshold.
    recorder.stop.return_value = np.full(16000, 0.5, dtype=np.float32)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "hello"

    app = App(
        config=config,
        recorder=recorder,
        transcriber=transcriber,
        injector=MagicMock(),
        feedback=MagicMock(),
        debug=True,
    )
    app.on_hotkey_press()
    app.on_hotkey_release()

    transcriber.transcribe.assert_called_once()


# --- silence gate config validation ----------------------------------------


def test_validation_rejects_negative_silence_threshold():
    from speakinput.config import AudioConfig, Config

    with pytest.raises(ValueError, match="silence_threshold"):
        Config(audio=AudioConfig(silence_threshold=-0.1)).validate()


def test_dry_run_prints_text_to_stderr_instead_of_typing(capsys):
    app, _, transcriber, injector, _ = _build_app(dry_run=True)
    transcriber.transcribe.return_value = "dry run output"
    app.on_hotkey_press()
    app.on_hotkey_release()
    captured = capsys.readouterr()
    assert "dry run output" in captured.err
    injector.inject.assert_not_called()


# --- debug-mode tests -------------------------------------------------------


def test_debug_mode_logs_press_start_and_end(capsys):
    app, _, _, _, _ = _build_app(debug=True)
    app.on_hotkey_press()
    app.on_hotkey_release()
    captured = capsys.readouterr()
    assert "[debug] key press start (alt_r)" in captured.err
    assert "[debug] key press end" in captured.err
    assert "held" in captured.err  # includes the held-for duration


def test_debug_mode_logs_audio_stats(capsys):
    app, recorder, _, _, _ = _build_app(debug=True)
    # 16000 samples at 16kHz = 1.0s, all-zero so rms=0.
    recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
    app.on_hotkey_press()
    app.on_hotkey_release()
    captured = capsys.readouterr()
    assert "[debug] audio: 16000 samples" in captured.err
    assert "rms=" in captured.err


def test_debug_mode_prints_transcript(capsys):
    app, _, transcriber, _, _ = _build_app(debug=True)
    transcriber.transcribe.return_value = "the quick brown fox"
    app.on_hotkey_press()
    app.on_hotkey_release()
    captured = capsys.readouterr()
    assert "[debug] transcript: 'the quick brown fox'" in captured.err


def test_debug_mode_prints_empty_transcript(capsys):
    """Empty transcript still gets printed so the user can distinguish silence from stuck."""
    app, _, transcriber, _, _ = _build_app(debug=True)
    transcriber.transcribe.return_value = ""
    app.on_hotkey_press()
    app.on_hotkey_release()
    captured = capsys.readouterr()
    assert "[debug] transcript: ''" in captured.err


def test_debug_mode_ignored_press_logs_message(capsys):
    app, recorder, _, _, _ = _build_app(debug=True)
    app.on_hotkey_press()  # acquires lock
    app.on_hotkey_press()  # ignored
    captured = capsys.readouterr()
    assert "press ignored: already busy" in captured.err


def test_no_debug_output_when_disabled(capsys):
    app, _, transcriber, _, _ = _build_app(debug=False)
    transcriber.transcribe.return_value = "should not appear"
    app.on_hotkey_press()
    app.on_hotkey_release()
    captured = capsys.readouterr()
    assert "[debug]" not in captured.err
    assert "should not appear" not in captured.err


def test_transcribe_error_releases_lock_and_returns_to_idle(capsys):
    app, _, transcriber, injector, feedback = _build_app(debug=True)
    transcriber.transcribe.side_effect = RuntimeError("model exploded")
    app.on_hotkey_press()
    app.on_hotkey_release()
    # Lock should be released — a fresh press should now succeed.
    app.on_hotkey_press()
    app.on_hotkey_release()
    assert injector.inject.call_count == 0
    feedback.set_state.assert_any_call("idle")


def test_inject_error_does_not_crash(capsys):
    app, _, _, injector, _ = _build_app(debug=True)
    injector.inject.side_effect = RuntimeError("pbcopy failed")
    app.on_hotkey_press()
    # Should NOT raise — the release path catches injection errors.
    app.on_hotkey_release()


# --- model bootstrap tests -------------------------------------------------


def test_run_bootstraps_model_when_no_transcriber_injected(monkeypatch):
    """If no transcriber was passed in, run() must call ensure_model and
    construct the default WhisperCppTranscriber with the resolved path."""
    from speakinput.app import App
    from speakinput.config import Config

    fake_ensure = MagicMock(return_value="/resolved/model.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config()
    # Recorder, injector, feedback are not used in run() until after bootstrap.
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    assert app.transcriber is None  # confirm precondition

    # Stop the run loop immediately after the listener starts.
    app._shutdown.set()
    app.run()

    fake_ensure.assert_called_once_with("small")
    fake_model_cls.assert_called_once()
    # The path from ensure_model should be the one passed to the constructor.
    assert fake_model_cls.call_args.kwargs["model"] == "/resolved/model.bin"


def test_run_skips_bootstrap_when_transcriber_injected():
    """If a transcriber was passed in (e.g. in tests), run() must not call
    ensure_model — the test owns the transcriber."""
    app, _, transcriber, _, _ = _build_app()
    assert app.transcriber is transcriber  # precondition

    # No monkeypatching of ensure_model — if it were called, the test would
    # fail because pywhispercpp isn't installed in the test env. We assert
    # by simply running the early portion of run() and checking nothing
    # was constructed.
    app._shutdown.set()
    app.run()  # should not raise
    assert app.transcriber is transcriber


def test_run_with_unknown_model_exits(monkeypatch, capsys):
    """A misconfigured model name should exit 2 before the listener starts."""
    from speakinput.app import App
    from speakinput.config import Config
    from speakinput.models import ModelNotFoundError

    fake_ensure = MagicMock(side_effect=ModelNotFoundError("unknown model"))
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)

    config = Config()
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())

    with pytest.raises(SystemExit) as exc_info:
        app.run()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "unknown model" in captured.err


def test_run_with_download_failure_exits(monkeypatch, capsys):
    from speakinput.app import App
    from speakinput.config import Config
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


# --- language / model auto-upgrade -----------------------------------------


def test_run_upgrades_english_only_model_for_zh(monkeypatch, capsys):
    """`base.en` + `language=zh` must auto-upgrade to `base` so Chinese
    actually works. The user is told about the swap in the log."""
    from speakinput.app import App
    from speakinput.config import Config, STTConfig

    fake_ensure = MagicMock(return_value="/resolved/base.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(stt=STTConfig(model="base.en", language="zh"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    # ensure_model was called with the *upgraded* name, not the original.
    fake_ensure.assert_called_once_with("base")
    captured = capsys.readouterr()
    assert "upgrading" in captured.err
    assert "base.en" in captured.err
    assert "base" in captured.err


def test_run_upgrades_english_only_model_for_auto(monkeypatch, capsys):
    """`language=auto` on an English-only model also upgrades, because the
    model can't recognize non-English speech at all."""
    from speakinput.app import App
    from speakinput.config import Config, STTConfig

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(stt=STTConfig(model="small.en", language="auto"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    fake_ensure.assert_called_once_with("small")
    captured = capsys.readouterr()
    assert "upgrading" in captured.err


def test_run_does_not_upgrade_when_english_only_model_with_en(monkeypatch, capsys):
    """If the user explicitly chose an English-only model AND set language=en,
    leave it alone — they wanted the fast English path."""
    from speakinput.app import App
    from speakinput.config import Config, STTConfig

    fake_ensure = MagicMock(return_value="/resolved/base.en.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(stt=STTConfig(model="base.en", language="en"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    fake_ensure.assert_called_once_with("base.en")
    captured = capsys.readouterr()
    assert "upgrading" not in captured.err


def test_run_does_not_upgrade_multilingual_model(monkeypatch, capsys):
    """A multilingual model + zh is the normal path. No upgrade message."""
    from speakinput.app import App
    from speakinput.config import Config, STTConfig

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(stt=STTConfig(model="small", language="zh"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    fake_ensure.assert_called_once_with("small")
    captured = capsys.readouterr()
    assert "upgrading" not in captured.err


# --- startup banner --------------------------------------------------------


def test_run_prints_startup_banner(monkeypatch, capsys):
    """The startup banner must show the active config so the user can verify
    it without opening config.toml."""
    from speakinput.app import App
    from speakinput.config import Config

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config()
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    captured = capsys.readouterr()
    assert "[startup] config   : (defaults — no config.toml found)" in captured.err
    assert "[startup] model    : small" in captured.err
    assert "[startup] language : auto" in captured.err
    assert "[startup] hotkey   : alt_r" in captured.err
    assert "[startup] sample   :" in captured.err
    assert "[startup] silence  :" in captured.err
    assert "[startup] prompt   :" in captured.err  # default is the embedded-vocab bias; full prompt follows
    assert "[startup] inject   :" in captured.err


def test_run_banner_shows_config_source_when_file_loaded(monkeypatch, capsys, tmp_path):
    """When the config was loaded from a file, the banner must show the
    source path so the user can verify the right file is being read."""
    from pathlib import Path

    from speakinput.app import App
    from speakinput.config import Config

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


def test_run_passes_initial_prompt_to_transcriber(monkeypatch):
    """`stt.initial_prompt` from config must reach the WhisperCppTranscriber
    constructor so the decoder gets the lexical prior on every call."""
    from speakinput.app import App
    from speakinput.config import Config, STTConfig

    fake_ensure = MagicMock(return_value="/resolved/small.bin")
    fake_model_cls = MagicMock()
    monkeypatch.setattr("speakinput.app.ensure_model", fake_ensure)
    monkeypatch.setattr("speakinput.app.WhisperCppTranscriber", fake_model_cls)

    config = Config(stt=STTConfig(initial_prompt="kubectl apply -f deployment.yaml"))
    app = App(config=config, recorder=MagicMock(), injector=MagicMock(), feedback=MagicMock())
    app._shutdown.set()
    app.run()

    fake_model_cls.assert_called_once()
    assert (
        fake_model_cls.call_args.kwargs["initial_prompt"]
        == "kubectl apply -f deployment.yaml"
    )
