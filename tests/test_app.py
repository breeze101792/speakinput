"""Tests for the App orchestrator. Mocks audio + transcriber + injector."""

from unittest.mock import MagicMock

import numpy as np


def _build_app(debug: bool = False, dry_run: bool = False):
    """Build an App with all I/O collaborators mocked out."""
    from speakinput.app import App
    from speakinput.config import Config

    config = Config()
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
