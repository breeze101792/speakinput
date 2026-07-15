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
