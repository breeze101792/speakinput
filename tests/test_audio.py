"""Tests for the audio recorder. Mocks sounddevice so the suite has no mic dependency."""

from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_recorder_with_captured_callback(fake_sd, sample_rate=16000):
    """Build an AudioRecorder whose PortAudio callback we can invoke directly."""
    from speakinput.audio import AudioRecorder

    captured: dict = {}

    def make_stream(**kwargs):
        captured["cb"] = kwargs["callback"]
        return MagicMock()

    fake_sd.InputStream.side_effect = make_stream

    r = AudioRecorder(sample_rate=sample_rate)
    r.start()
    return r, captured["cb"]


@pytest.fixture
def fake_sd(monkeypatch):
    """Install a fake sounddevice by patching the module attribute the recorder
    actually looks up (its own `sd` reference), not just `sys.modules`."""
    from speakinput import audio as audio_mod

    fake = MagicMock()
    fake.query_devices.return_value = []
    fake.InputStream = MagicMock()
    monkeypatch.setattr(audio_mod, "sd", fake, raising=False)
    return fake


def test_recorder_rejects_when_sounddevice_missing(monkeypatch):
    from speakinput import audio as audio_mod
    from speakinput.audio import AudioError, AudioRecorder

    monkeypatch.setattr(audio_mod, "sd", None, raising=False)
    r = AudioRecorder()
    with pytest.raises(AudioError, match="sounddevice"):
        r.start()


def test_recorder_start_creates_stream_and_starts_recording(fake_sd):
    from speakinput.audio import AudioRecorder

    stream_instance = MagicMock()
    fake_sd.InputStream.return_value = stream_instance

    r = AudioRecorder()
    assert not r.is_recording()
    r.start()
    assert r.is_recording()
    fake_sd.InputStream.assert_called_once()
    stream_instance.start.assert_called_once()


def test_recorder_start_is_idempotent(fake_sd):
    from speakinput.audio import AudioRecorder

    r = AudioRecorder()
    r.start()
    r.start()
    assert fake_sd.InputStream.call_count == 1


def test_recorder_stop_returns_empty_when_never_started(fake_sd):
    from speakinput.audio import AudioRecorder

    r = AudioRecorder()
    out = r.stop()
    assert out.size == 0
    assert out.dtype == np.float32


def test_recorder_stop_concatenates_chunks(fake_sd):
    r, cb = _make_recorder_with_captured_callback(fake_sd)
    cb(np.ones((800, 1), dtype=np.float32) * 0.1, 800, None, None)
    cb(np.ones((800, 1), dtype=np.float32) * 0.2, 800, None, None)
    out = r.stop()
    assert not r.is_recording()
    assert out.dtype == np.float32
    assert out.shape == (1600,)
    np.testing.assert_array_equal(out[:800], np.full(800, 0.1, dtype=np.float32))
    np.testing.assert_array_equal(out[800:], np.full(800, 0.2, dtype=np.float32))


def test_recorder_stop_saves_audio_even_when_status_flag_set(fake_sd):
    r, cb = _make_recorder_with_captured_callback(fake_sd)
    cb(np.ones((100, 1), dtype=np.float32), 100, None, "overflow")
    out = r.stop()
    assert out.shape == (100,)


def test_recorder_does_not_alias_input_buffer(fake_sd):
    """PortAudio reuses the buffer between callbacks; the recorder must copy."""
    r, cb = _make_recorder_with_captured_callback(fake_sd)
    buf = np.ones((100, 1), dtype=np.float32)
    cb(buf, 100, None, None)
    # Mutate the original buffer; the recorder's stored copy should not change.
    buf.fill(0.5)
    out = r.stop()
    assert out.shape == (100,)
    np.testing.assert_array_equal(out, np.ones(100, dtype=np.float32))


def test_recorder_chunk_generator_validates_args(fake_sd):
    from speakinput.audio import AudioRecorder

    r = AudioRecorder()
    with pytest.raises(ValueError):
        r.chunk_generator(window_seconds=0)
    with pytest.raises(ValueError):
        r.chunk_generator(window_seconds=1.0, hop_seconds=2.0)


def test_recorder_chunk_generator_is_empty_in_v1(fake_sd):
    from speakinput.audio import AudioRecorder

    r = AudioRecorder()
    gen = r.chunk_generator()
    # In v1 the seam yields nothing.
    assert list(gen) == []


def test_list_input_devices_filters_by_input_channels(fake_sd):
    fake_sd.query_devices.return_value = [
        {"name": "Mic", "max_input_channels": 1, "default_samplerate": 48000.0},
        {"name": "Speaker (no input)", "max_input_channels": 0, "default_samplerate": 48000.0},
    ]
    from speakinput.audio import list_input_devices

    devices = list_input_devices()
    assert len(devices) == 1
    assert devices[0]["name"] == "Mic"
    assert devices[0]["index"] == 0


def test_recorder_current_rms_is_zero_before_any_audio(fake_sd):
    """Before the first audio callback fires, current_rms() returns 0.0."""
    from speakinput.audio import AudioRecorder

    r = AudioRecorder()
    r.start()
    assert r.current_rms() == 0.0
    r.stop()


def test_recorder_current_rms_reflects_last_chunk(fake_sd):
    """The watchdog polls current_rms() to detect silence; it must
    reflect the most recent chunk's RMS, not the running average."""
    from speakinput.audio import AudioRecorder

    r, cb = _make_recorder_with_captured_callback(fake_sd)
    # First chunk: loud (RMS = 0.5)
    cb(np.full((480, 1), 0.5, dtype=np.float32), 480, None, None)
    assert r.current_rms() == pytest.approx(0.5, abs=1e-5)
    # Second chunk: silent
    cb(np.zeros((480, 1), dtype=np.float32), 480, None, None)
    assert r.current_rms() == 0.0
    r.stop()


def test_recorder_current_rms_resets_on_start(fake_sd):
    """Starting a new recording should reset the live RMS to 0.0,
    not carry over from a previous run."""
    from speakinput.audio import AudioRecorder

    r, cb = _make_recorder_with_captured_callback(fake_sd)
    cb(np.full((480, 1), 0.5, dtype=np.float32), 480, None, None)
    r.stop()
    # Now restart — fresh state.
    fake_sd.InputStream.return_value = MagicMock()
    r.start()
    assert r.current_rms() == 0.0
    r.stop()


def test_recorder_drain_returns_buffer_and_keeps_recording(fake_sd):
    """drain() is the chunked-release path: return what was recorded
    since the last start/drain, keep the stream alive so more audio
    can be captured. After drain the live RMS resets to 0."""
    from speakinput.audio import AudioRecorder

    r, cb = _make_recorder_with_captured_callback(fake_sd)
    cb(np.full((480, 1), 0.3, dtype=np.float32), 480, None, None)
    cb(np.full((480, 1), 0.4, dtype=np.float32), 480, None, None)
    drained = r.drain()
    assert r.is_recording()  # still recording
    assert drained.size == 960
    assert r.current_rms() == 0.0  # reset after drain
    # A new callback after drain accumulates into a fresh buffer.
    cb(np.full((240, 1), 0.5, dtype=np.float32), 240, None, None)
    drained2 = r.drain()
    assert drained2.size == 240
    r.close()


def test_recorder_drain_on_empty_buffer_returns_empty(fake_sd):
    """Draining with no audio in flight returns an empty array without
    raising."""
    from speakinput.audio import AudioRecorder

    r = AudioRecorder()
    r.start()
    out = r.drain()
    assert out.size == 0
    assert out.dtype == np.float32
    assert r.is_recording()
    r.close()


def test_recorder_close_is_idempotent(fake_sd):
    """Calling close() twice is a no-op."""
    from speakinput.audio import AudioRecorder

    r, _ = _make_recorder_with_captured_callback(fake_sd)
    r.close()
    r.close()  # second call must not raise
    assert not r.is_recording()


def test_recorder_close_is_serialized_across_threads(fake_sd):
    """Two threads racing close() must not both enter PortAudio's
    stop/close. CoreAudio deadlocks on the HAL mutex when two threads
    stop the same stream concurrently (observed in production: main
    thread in atexit's Pa_Terminate vs. hotkey thread in
    recorder.close()). The stream lock serializes them: the loser sees
    `_stream is None` and no-ops."""
    import threading
    import time

    from speakinput.audio import AudioRecorder

    stream = MagicMock()
    # Make stop() slow so the race window is wide open; the second
    # thread MUST wait for the first to finish instead of piling in.
    stream.stop.side_effect = lambda: time.sleep(0.2)
    fake_sd.InputStream.side_effect = lambda **kw: stream

    r = AudioRecorder()
    r.start()
    assert r.is_recording()

    t = threading.Thread(target=r.close)
    t.start()
    time.sleep(0.05)  # let t get inside close() first
    r.close()  # main thread races in
    t.join(timeout=5.0)
    assert not t.is_alive()
    # Despite the race, PortAudio saw exactly one stop + one close.
    stream.stop.assert_called_once()
    stream.close.assert_called_once()
    assert not r.is_recording()


def test_recorder_stop_equivalent_to_drain_then_close(fake_sd):
    """stop() should still return the full buffer and tear down the
    stream, preserving the existing single-chunk release behavior."""
    from speakinput.audio import AudioRecorder

    r, cb = _make_recorder_with_captured_callback(fake_sd)
    cb(np.full((800, 1), 0.2, dtype=np.float32), 800, None, None)
    cb(np.full((800, 1), 0.3, dtype=np.float32), 800, None, None)
    out = r.stop()
    assert not r.is_recording()
    assert out.size == 1600


# --- mic-presence preflight (device pinned in config) ----------------------


def test_default_device_skips_preflight_check(fake_sd):
    """When `device` is None (system default), we must NOT call
    query_devices on every press — PortAudio re-resolves the default
    on every InputStream(), so the check is unnecessary AND adds
    ~1ms of work to every press for no benefit."""
    from speakinput.audio import AudioRecorder

    r = AudioRecorder()  # default device=None
    r.start()
    fake_sd.query_devices.assert_not_called()


def test_pinned_device_present_uses_it(fake_sd, capsys):
    """When the pinned device IS present, use it as-is. The preflight
    call is the only `query_devices` interaction — no warning, no
    fallback."""
    from speakinput.audio import AudioRecorder

    fake_sd.query_devices.return_value = {"name": "USB Mic"}  # no exception
    r = AudioRecorder(device=2)
    r.start()
    fake_sd.query_devices.assert_called_once_with(2)
    # The stream must be opened with the requested device, not the default.
    kwargs = fake_sd.InputStream.call_args.kwargs
    assert kwargs["device"] == 2
    captured = capsys.readouterr()
    assert "falling back" not in captured.err


def test_pinned_device_gone_falls_back_to_default(fake_sd, capsys):
    """If the pinned device disappeared (USB unplug, Bluetooth off,
    etc.), fall back to system default with a one-line warning.
    Without this, the stream open would raise and the press would
    silently fail."""
    from speakinput.audio import AudioRecorder

    fake_sd.query_devices.side_effect = Exception("device 2 not found")
    r = AudioRecorder(device=2)
    r.start()
    # The InputStream was opened with `device=None` (system default).
    kwargs = fake_sd.InputStream.call_args.kwargs
    assert kwargs["device"] is None
    captured = capsys.readouterr()
    assert "device 2 is not available" in captured.err
    assert "falling back" in captured.err


def test_no_mic_at_all_raises_with_clear_message(fake_sd, capsys):
    """If even the system default is gone (no mic plugged in, or
    Microphone permission revoked), InputStream raises. The recorder
    must re-raise as AudioError with a message that tells the user
    what to do — connect a mic, check System Settings → Microphone."""
    from speakinput.audio import AudioError, AudioRecorder

    fake_sd.query_devices.return_value = []  # no devices at all
    fake_sd.InputStream.side_effect = OSError("no input device")
    r = AudioRecorder()
    with pytest.raises(AudioError, match="audio stream open failed"):
        r.start()
    captured = capsys.readouterr()
    assert "could not open audio input stream" in captured.err
    assert "Microphone permission" in captured.err
    # Recorder must NOT be left in a half-started state — the next
    # press should be able to retry cleanly.
    assert not r.is_recording()


def test_recorder_surfaces_stream_error_on_press():
    """App.on_hotkey_press must surface a clear signal when the
    recorder can't open a stream. The press logs a WARNING (not a
    full stack trace — this is a user-fixable environment problem,
    not a bug) and the feedback state goes to 'error' so the
    menu-bar indicator shows the user the press was registered but
    couldn't capture audio. On release, the feedback returns to
    'idle' so the error glyph doesn't get stuck."""
    from speakinput.app import App
    from speakinput.audio import AudioError
    from speakinput.config import AudioConfig, Config
    from unittest.mock import MagicMock

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    recorder = MagicMock()
    recorder.start.side_effect = AudioError("no mic")
    recorder.is_recording.return_value = False
    feedback = MagicMock()
    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: MagicMock()},
        injector=MagicMock(),
        feedback=feedback,
    )
    profile = app.config.primary
    app.on_hotkey_press(profile)
    # After a failed press, feedback shows the distinct 'error' state —
    # not 'idle' (which would look like nothing happened) and not
    # 'listening' (which would lie about whether capture is running).
    feedback.set_state.assert_any_call("error")
    # The busy lock is released so a fresh press is accepted.
    assert not app._busy.locked()
    # And on release, feedback resets to 'idle' so the error glyph
    # doesn't get stuck after the user lets go.
    app.on_hotkey_release(profile)
    feedback.set_state.assert_any_call("idle")


def test_recorder_unexpected_error_still_logs_traceback(capsys):
    """    Non-AudioError exceptions from recorder.start() are genuine
    bugs (not user-fixable environment problems), so they keep
    getting the full traceback via log.exception. Only the
    AudioError path was demoted to WARNING."""
    from speakinput.app import App
    from speakinput.config import AudioConfig, Config
    from unittest.mock import MagicMock

    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    recorder = MagicMock()
    recorder.start.side_effect = RuntimeError("unexpected")
    feedback = MagicMock()
    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: MagicMock()},
        injector=MagicMock(),
        feedback=feedback,
    )
    app.on_hotkey_press(app.config.primary)
    # Same recovery: feedback shows 'error', busy lock released.
    feedback.set_state.assert_any_call("error")
    assert not app._busy.locked()
    # Sanity: we're not crashing the app on a non-AudioError either.
    # (The traceback goes to logging, not the user.)


def test_portaudio_error_code_translates_to_friendly_message(fake_sd, capsys):
    """The -9986 / 'Invalid Property Value' AUHAL failure that
    motivated this fix must come out as a useful message, not the
    raw 'Internal PortAudio error [PaErrorCode -9986]' string."""
    from speakinput.audio import AudioError, AudioRecorder, _describe_audio_error

    class FakePortAudioError(Exception):
        # Mimic sounddevice.PortAudioError: args = (msg, code)
        def __init__(self, msg: str, code: int) -> None:
            super().__init__(msg, code)

    msg, code = "Error opening InputStream: Internal PortAudio error [PaErrorCode -9986]", -9986
    exc = FakePortAudioError(msg, code)
    r = AudioRecorder()
    fake_sd.InputStream.side_effect = exc
    with pytest.raises(AudioError) as ei:
        r.start()
    # The AudioError message should NOT just be the raw PortAudio
    # message — that was the bug. It should mention code -9986 in
    # a human-readable form.
    assert "-9986" in str(ei.value)
    assert "audio stream open failed" in str(ei.value)
    # And the helper itself returns something actionable.
    assert "CoreAudio" in _describe_audio_error(exc) or "engine" in _describe_audio_error(exc)


def test_describe_audio_error_handles_unknown_codes():
    """Unknown PortAudio codes still produce a usable message
    rather than the raw 'Internal PortAudio error' string."""
    from speakinput.audio import _describe_audio_error

    class FakePAError(Exception):
        def __init__(self, code: int) -> None:
            super().__init__("Error opening InputStream: Internal PortAudio error", code)

    desc = _describe_audio_error(FakePAError(-12345))
    assert "code -12345" in desc
    # Plain exceptions (no int code) don't crash.
    assert "audio device error" in _describe_audio_error(RuntimeError("boom"))


def test_feedback_supports_error_state():
    """The 'error' state must be accepted by the headless feedback
    implementation so the press-failure path doesn't crash the app
    when rumps isn't available."""
    from speakinput.feedback import StderrFeedback

    fb = StderrFeedback()
    fb.set_state("error")  # would raise ValueError before this fix


# --- stability: PortAudio callback survivability, stream-leak prevention ---


def test_on_audio_callback_survives_numpy_failure(fake_sd):
    """The PortAudio callback must NEVER raise — if it does, the
    audio thread dies and either the process is killed or no more
    audio ever arrives. We inject a numpy failure (the indata.copy()
    raising) and verify the callback drops the chunk silently
    rather than letting the exception escape."""
    r, cb = _make_recorder_with_captured_callback(fake_sd)
    # Build a fake indata whose .copy() raises.
    class BadIndata:
        def copy(self):
            raise MemoryError("simulated OOM in numpy")

    # No exception should escape the callback.
    cb(BadIndata(), 160, None, None)
    # The recorder is still alive and can be re-armed.
    assert r.is_recording()


def test_start_failure_closes_partial_stream(fake_sd):
    """If `sd.InputStream.start()` raises after construction, the
    partially-constructed native stream is closed here (not leaked
    until sounddevice's atexit). The fixture's `InputStream` returns
    a MagicMock that records `.close()` calls.
    """
    from speakinput.audio import AudioError, AudioRecorder

    class LeakedStream:
        closed = False
        def start(self):
            raise RuntimeError("AUHAL rejected the stream")
        def close(self):
            self.closed = True

    fake_sd.query_devices.return_value = []
    fake_sd.InputStream.return_value = LeakedStream()
    r = AudioRecorder()
    with pytest.raises(AudioError):
        r.start()
    # The leaked stream must have been closed, not orphaned.
    assert fake_sd.InputStream.return_value.closed is True
    # And the recorder is in a clean state for the next press.
    assert not r.is_recording()
    assert r._stream is None


def test_close_sets_chunks_to_none(fake_sd):
    """After close(), any audio callback that fires must NOT
    silently append to a stale chunks list. The fix sets
    `_chunks = None` in close() so the callback's guard skips
    the append and the next `start()` gets a fresh list."""
    import numpy as np
    r, cb = _make_recorder_with_captured_callback(fake_sd)
    r.close()
    # After close, _chunks must be None (not []). This is the
    # contract the audio callback's `if self._chunks is not None`
    # guard relies on.
    assert r._chunks is None
    # Drive a callback with a real numpy array — should not crash
    # and should not raise.
    cb(np.zeros((1, 160), dtype="float32"), 160, None, None)
    assert r._chunks is None  # callback dropped the chunk
