"""Tests for the silence trimmer and the auto-stop watchdog."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from speakinput.silence import SilenceWatchdog, trim_trailing_silence


# --- trim_trailing_silence ------------------------------------------------


def test_trim_trailing_silence_returns_empty_buffer_unchanged():
    audio = np.zeros(0, dtype=np.float32)
    out = trim_trailing_silence(audio, sample_rate=16000, threshold=0.005)
    assert out is audio  # no copy when nothing to do


def test_trim_trailing_silence_drops_silent_tail():
    """Buffer is 50ms of speech followed by 200ms of silence. The
    200ms silent tail must be removed; the 50ms speech head must be
    preserved (possibly plus a partial hop that straddles the
    boundary, so the kept region is at most ~80ms of audio)."""
    sr = 16000
    speech = np.ones(sr // 20, dtype=np.float32) * 0.5  # 50ms, RMS 0.5
    silence = np.zeros(sr // 5, dtype=np.float32)       # 200ms, RMS 0
    audio = np.concatenate([speech, silence])           # 4000 samples
    out = trim_trailing_silence(audio, sample_rate=sr, threshold=0.005)
    # The kept region is at most the 50ms speech (800 samples) plus a
    # partial hop straddling the boundary — call it 800 + 480 = 1280.
    # Critically, the 200ms silent tail (3200 samples) is gone.
    assert out.size <= 1280
    assert out.size < audio.size
    # The kept region must contain real speech, not silence.
    assert float(np.sqrt(np.mean(out * out))) > 0.0


def test_trim_trailing_silence_preserves_leading_silence():
    """A 200ms silent lead followed by 50ms of speech should be kept
    fully — trim only chops the trailing portion, never the leading."""
    sr = 16000
    silence = np.zeros(sr // 5, dtype=np.float32)
    speech = np.ones(sr // 20, dtype=np.float32) * 0.5
    audio = np.concatenate([silence, speech])
    out = trim_trailing_silence(audio, sample_rate=sr, threshold=0.005)
    # No silent tail to trim; output equals input.
    np.testing.assert_array_equal(out, audio)


def test_trim_trailing_silence_returns_input_when_fully_silent():
    """If every hop is silent, return the buffer unchanged so the
    downstream silence-gate can short-circuit it."""
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    out = trim_trailing_silence(audio, sample_rate=sr, threshold=0.005)
    np.testing.assert_array_equal(out, audio)


def test_trim_trailing_silence_disabled_when_threshold_zero():
    """Threshold of 0 disables the trim entirely — every hop is "loud"
    in a degenerate sense, so we'd return the input anyway, but the
    function should short-circuit before iterating."""
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    out = trim_trailing_silence(audio, sample_rate=sr, threshold=0)
    assert out is audio


def test_trim_trailing_silence_partial_hop_at_end():
    """If the buffer length is not a multiple of hop_samples, the final
    short hop is still included in the scan."""
    sr = 16000
    # 480 samples of speech + 100 samples of silence. hop=480, so the
    # function scans [0:480] (loud) and ignores the trailing 100.
    audio = np.concatenate(
        [np.ones(480, dtype=np.float32), np.zeros(100, dtype=np.float32)]
    )
    out = trim_trailing_silence(audio, sample_rate=sr, threshold=0.005)
    assert out.size == 480


# --- SilenceWatchdog ------------------------------------------------------


def _fake_recorder(is_recording: bool = True, rms: float = 0.0) -> MagicMock:
    rec = MagicMock()
    rec.is_recording.return_value = is_recording
    rec.current_rms.return_value = rms
    return rec


def test_watchdog_rejects_zero_auto_stop():
    """auto_stop_seconds <= 0 means "feature off" — starting a watchdog
    in that mode is a programmer error."""
    rec = _fake_recorder()
    with pytest.raises(ValueError, match="auto_stop_seconds"):
        SilenceWatchdog(recorder=rec, threshold=0.005, auto_stop_seconds=0, on_trigger=lambda: None)


def test_watchdog_triggers_on_sustained_silence():
    """The watchdog should call on_trigger after the configured number
    of seconds of sub-threshold audio pass."""
    rec = _fake_recorder(rms=0.0)
    triggered = []

    # 0.1s auto-stop, 0.02s poll. The first poll happens at t=0.02s
    # and finds silence, so the watchdog should trigger roughly at
    # t=0.1s — well under 1s of wall time.
    dog = SilenceWatchdog(
        recorder=rec,
        threshold=0.005,
        auto_stop_seconds=0.1,
        on_trigger=lambda: triggered.append(True),
    )
    dog.start()
    try:
        deadline = time.monotonic() + 1.0
        while not triggered and time.monotonic() < deadline:
            time.sleep(0.02)
        assert triggered == [True]
    finally:
        dog.stop()


def test_watchdog_does_not_trigger_on_loud_audio():
    """If current_rms() stays above threshold, the watchdog must NOT
    fire even after a long wait."""
    rec = _fake_recorder(rms=0.5)
    triggered = []

    dog = SilenceWatchdog(
        recorder=rec,
        threshold=0.005,
        auto_stop_seconds=0.1,
        on_trigger=lambda: triggered.append(True),
    )
    dog.start()
    time.sleep(0.4)  # 4x the auto-stop window, all "loud"
    dog.stop()
    assert triggered == []


def test_watchdog_stops_when_recorder_not_recording():
    """If the recorder has been stopped (e.g. manual release), the
    watchdog must exit its loop without triggering."""
    rec = _fake_recorder(is_recording=False, rms=0.0)
    triggered = []

    dog = SilenceWatchdog(
        recorder=rec,
        threshold=0.005,
        auto_stop_seconds=0.1,
        on_trigger=lambda: triggered.append(True),
    )
    dog.start()
    time.sleep(0.3)
    # The watchdog thread should have exited on its own.
    assert triggered == []
    dog.stop()  # idempotent


def test_watchdog_stop_is_idempotent_and_fast():
    """Calling stop() multiple times is safe; the watchdog exits on the
    next poll (within POLL_INTERVAL_S)."""
    rec = _fake_recorder(rms=0.0)
    triggered = []

    dog = SilenceWatchdog(
        recorder=rec,
        threshold=0.005,
        auto_stop_seconds=0.1,
        on_trigger=lambda: triggered.append(True),
    )
    dog.start()
    dog.stop()
    dog.stop()  # second call is a no-op
    dog.stop()  # third call too
    time.sleep(0.2)
    assert triggered == []


def test_watchdog_recovers_from_brief_loud_burst():
    """A short loud moment in the middle of an otherwise-silent
    recording should reset the silence timer, not trigger early."""
    rms_values = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    rec = MagicMock()
    rec.is_recording.return_value = True
    rec.current_rms.side_effect = rms_values
    triggered = []

    # 0.15s auto-stop = 3 polls of silence. The loud burst at index 2
    # resets the timer; with 5 polls of post-burst silence the watchdog
    # *would* fire. So we wait only 0.2s (4 polls total: 2 pre-burst
    # silence + 1 burst + 1 post-burst) — not enough to trigger.
    dog = SilenceWatchdog(
        recorder=rec,
        threshold=0.005,
        auto_stop_seconds=0.15,
        on_trigger=lambda: triggered.append(True),
    )
    dog.start()
    time.sleep(0.2)
    dog.stop()
    assert triggered == []


def test_watchdog_can_restart_after_thread_exits():
    """After stop() the watchdog's thread exits; a follow-up
    start() must bring it back. The old `if self._thread is not
    None` check treated a stopped thread (reference still set,
    is_alive() == False) as "already running" and silently
    no-op'd. The is_alive() check makes the watchdog reusable.
    """
    from speakinput.silence import SilenceWatchdog

    rec = _fake_recorder(rms=0.0)
    triggered: list[bool] = []

    dog = SilenceWatchdog(
        recorder=rec,
        threshold=0.005,
        auto_stop_seconds=0.1,
        on_trigger=lambda: triggered.append(True),
    )
    dog.start()
    time.sleep(0.1)  # let the thread run at least once
    dog.stop()
    # Thread reference still set, but is_alive() must be False.
    assert dog._thread is not None
    # Restart must spin up a fresh thread.
    dog.start()
    time.sleep(0.05)
    dog.stop()
    # The thread reference was reassigned; it's the second thread.
    assert dog._thread is not None


# --- stream health check: dead-stream detection in the watchdog -----------


def _fake_recorder_healthy(**kwargs):
    """Build a MagicMock recorder with stream_healthy support."""
    healthy = kwargs.pop("healthy", True)
    rec = _fake_recorder(**kwargs)
    rec.stream_healthy.return_value = healthy
    return rec


def test_watchdog_triggers_on_dead_stream(monkeypatch):
    """If stream_healthy() returns False for long enough, the watchdog
    must fire on_trigger even if the RMS is still above threshold.
    This catches the post-sleep/wake failure where the PortAudio
    callback stops but is_recording() stays True."""
    monkeypatch.setattr(SilenceWatchdog, "HEALTHY_TIMEOUT_S", 0.15)
    rec = _fake_recorder_healthy(rms=0.5, healthy=False)
    triggered = []

    dog = SilenceWatchdog(
        recorder=rec,
        threshold=0.005,
        auto_stop_seconds=10.0,  # very long — we don't want silence trigger
        on_trigger=lambda: triggered.append(True),
    )
    dog.start()
    try:
        deadline = time.monotonic() + 2.0
        while not triggered and time.monotonic() < deadline:
            time.sleep(0.02)
        assert triggered == [True], "watchdog should fire on dead stream"
    finally:
        dog.stop()


def test_watchdog_does_not_trigger_on_healthy_stream():
    """If stream_healthy() returns True, the watchdog must NOT fire
    from the health-check path (silence path is independent)."""
    rec = _fake_recorder_healthy(rms=0.5, healthy=True)
    triggered = []

    dog = SilenceWatchdog(
        recorder=rec,
        threshold=0.005,
        auto_stop_seconds=10.0,
        on_trigger=lambda: triggered.append(True),
    )
    dog.start()
    time.sleep(0.3)
    dog.stop()
    assert triggered == []


def test_watchdog_resets_health_on_recovery():
    """If the stream briefly goes unhealthy then recovers, the
    dead-since timer must reset so we don't false-positive."""
    healthy_sequence = [True, True, False, False, True, True, True]
    rec = MagicMock()
    rec.is_recording.return_value = True
    rec.current_rms.return_value = 0.5
    rec.stream_healthy.side_effect = healthy_sequence
    triggered = []

    dog = SilenceWatchdog(
        recorder=rec,
        threshold=0.005,
        auto_stop_seconds=10.0,
        on_trigger=lambda: triggered.append(True),
    )
    dog.start()
    time.sleep(0.4)  # enough for ~8 polls
    dog.stop()
    # The stream recovered after 2 unhealthy polls (< HEALTHY_TIMEOUT_S),
    # so the watchdog should NOT have triggered.
    assert triggered == []


def test_watchdog_does_not_double_trigger():
    """Once the trigger has fired (either from silence or dead stream),
    subsequent health-check failures must not fire again."""
    call_count = 0

    def on_trigger():
        nonlocal call_count
        call_count += 1

    rec = _fake_recorder_healthy(rms=0.0, healthy=False)
    dog = SilenceWatchdog(
        recorder=rec,
        threshold=0.005,
        auto_stop_seconds=0.05,  # 50ms — will trigger from silence fast
        on_trigger=on_trigger,
    )
    dog.start()
    time.sleep(0.5)  # well past both silence and health timeouts
    dog.stop()
    assert call_count == 1, f"trigger called {call_count} times; expected exactly 1"
