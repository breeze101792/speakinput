"""Silence and speech detection helpers.

Three pieces:

- `trim_trailing_silence`: walk a recorded buffer backwards in
  ~30ms hops and drop any trailing portion whose per-hop RMS is below
  `threshold`. The leading portion is preserved. Useful as a
  pre-transcribe cleanup step so whisper doesn't see a long silent
  tail (a real hallucination source).

- `analyse_speech`: short-time energy VAD. Splits the buffer into
  20ms frames, estimates the noise floor as a low percentile of the
  frame RMS values, and counts the frames that stand clear of it.
  A whole-buffer mean RMS cannot do this job: one fan, one keyboard
  click, or one desk bump lifts the mean above any fixed floor, so
  whisper gets fed noise and invents text. A press with no speech in
  it is rejected here instead.

- `SilenceWatchdog`: a small daemon thread that polls
  `AudioRecorder.current_rms()` while a hotkey is held, and calls a
  user-supplied callback when `auto_stop_seconds` of consecutive
  sub-threshold audio has passed. Powers "release the key for me"
  behavior so the user doesn't have to time releases at end of
  sentence.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from speakinput.audio import Recorder

# How wide a slice to look at when measuring RMS for the trailing
# silence trim. 30ms is short enough that we don't accidentally chop
# the end of a word, long enough to swallow a typical consonant
# release or breath. At 16kHz that's 480 samples.
_TRIM_HOP_SAMPLES = 480

# Speech gate defaults. 20ms frames are short enough to resolve a
# syllable onset and long enough for the RMS to be meaningful at
# 16kHz (320 samples per frame).
_SPEECH_FRAME_MS = 20
# A frame counts as speech when its RMS is at least this multiple of
# the estimated noise floor. 2.0 is 6dB — speech peaks sit 10-20dB
# above room tone, so this clears the floor without missing quiet
# syllables. Stationary noise (a fan, a hum) has no frame 6dB above
# its own 10th percentile, which is what makes the gate reject it.
_DEFAULT_NOISE_MARGIN = 2.0
# A press must contain at least this much speech to reach whisper.
# Below ~100ms there is not enough context to decode and whisper tends
# to produce filler, so the default sits just under that.
_DEFAULT_MIN_SPEECH_MS = 80.0
# And at least one run of consecutive speech frames this long. An
# impulse (a click, a tap) energises one or two frames; requiring a
# sustained run is what separates a transient from a voiced syllable.
# Bounded above by `min_speech_ms` so lowering that knob still works.
_DEFAULT_MIN_SPEECH_RUN_MS = 60.0
# Percentile of the frame-RMS distribution used as the noise floor.
# The 10th percentile ignores the quiet gaps between words and the
# loud syllables alike, so it tracks room tone in a buffer that
# contains speech. Frames that are digitally zero are excluded first:
# some devices and OS noise gates zero-fill muted output, and a floor
# of 0 would collapse the relative test back to the absolute one.
_NOISE_FLOOR_PERCENTILE = 10


@dataclass(frozen=True)
class SpeechStats:
    """Result of `analyse_speech`, kept small so it is cheap to log."""

    # Estimated room tone, as a frame RMS in the same units as the audio.
    noise_floor: float
    # Total duration of frames that stood clear of the noise floor.
    speech_ms: float
    # True when the speech totals the minimum duration AND contains
    # one unbroken run long enough to be a syllable rather than a click.
    has_speech: bool


def _frame_rms(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    """RMS per non-overlapping frame. Trailing partial frame is dropped."""
    n_frames = audio.size // frame_samples
    frames = audio[: n_frames * frame_samples].reshape(n_frames, frame_samples)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest unbroken run of True in a 1-D bool array."""
    if mask.size == 0 or not mask.any():
        return 0
    # Pad both ends with False so every run is closed by a False->True
    # and a True->False edge, and read the run lengths off the edges.
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(edges[1::2] - edges[0::2]))


def analyse_speech(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float,
    *,
    frame_ms: float = _SPEECH_FRAME_MS,
    noise_margin: float = _DEFAULT_NOISE_MARGIN,
    min_speech_ms: float = _DEFAULT_MIN_SPEECH_MS,
) -> SpeechStats:
    """Decide whether a recorded buffer contains speech.

    The buffer is split into `frame_ms` frames and each frame's RMS is
    compared against a noise floor estimated from the buffer itself
    (the 10th percentile of frame RMS, ignoring digitally zero
    frames). Frames above `max(threshold, noise_floor * noise_margin)`
    count as speech.

    Speech must then satisfy two conditions: at least `min_speech_ms`
    of accumulated speech frames, and at least one unbroken run of
    60ms. The run requirement is what rejects transients — a click
    energises one or two frames, and two clicks in one press would
    otherwise add up to the duration of a word.

    Why not a whole-buffer mean RMS: a mean is dominated by the
    loudest part of the buffer, so a single click is enough to lift a
    silent press over the floor. A per-frame comparison with a floor
    derived from the same buffer is immune to that.

    The rule is one of level against the buffer's own noise floor, so
    a perfectly steady signal (a hum, a test tone, an AGC-flattened
    voice) never counts as speech: no frame can stand 6dB above its
    own 10th percentile.

    `threshold` is the absolute floor from `[audio].silence_threshold`.
    A value <= 0 disables the gate: the buffer always counts as
    speech. Buffers shorter than one frame never count as speech.
    """
    if audio.size == 0 or sample_rate <= 0:
        return SpeechStats(noise_floor=0.0, speech_ms=0.0, has_speech=False)
    duration_ms = audio.size * 1000.0 / sample_rate
    if threshold <= 0:
        return SpeechStats(noise_floor=0.0, speech_ms=duration_ms, has_speech=True)
    frame_samples = int(sample_rate * frame_ms / 1000.0)
    if frame_samples <= 0 or audio.size < frame_samples:
        return SpeechStats(noise_floor=0.0, speech_ms=0.0, has_speech=False)
    rms = _frame_rms(audio.astype(np.float32, copy=False), frame_samples)
    if rms.size == 0:
        return SpeechStats(noise_floor=0.0, speech_ms=0.0, has_speech=False)
    # Digitally zero frames are not a measurement of room tone: a muted
    # device or an OS noise gate zero-fills them, and letting a floor of
    # 0 through would reduce the relative test to the absolute one.
    audible = rms[rms > 0.0]
    if audible.size == 0:
        return SpeechStats(noise_floor=0.0, speech_ms=0.0, has_speech=False)
    noise_floor = float(np.percentile(audible, _NOISE_FLOOR_PERCENTILE))
    bar = max(threshold, noise_floor * noise_margin)
    speech = rms >= bar
    speech_frames = int(np.count_nonzero(speech))
    speech_ms = speech_frames * frame_ms
    # The run requirement never exceeds the total requirement, so
    # lowering `min_speech_ms` (e.g. for very short words) still lets a
    # proportionally shorter utterance through.
    required_run_ms = min(_DEFAULT_MIN_SPEECH_RUN_MS, min_speech_ms)
    longest_run_ms = _longest_true_run(speech) * frame_ms
    return SpeechStats(
        noise_floor=noise_floor,
        speech_ms=speech_ms,
        has_speech=(
            speech_frames > 0
            and speech_ms >= min_speech_ms
            and longest_run_ms >= required_run_ms
        ),
    )


def _chunk_rms(chunk: np.ndarray) -> float:
    """Root-mean-square of a 1-D float32 chunk. Returns 0.0 for empty."""
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk * chunk)))


def trim_trailing_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float,
    hop_samples: int = _TRIM_HOP_SAMPLES,
) -> np.ndarray:
    """Drop trailing samples whose per-hop RMS is below `threshold`.

    Walks the buffer from the end towards the start, looking for the
    last hop whose RMS is >= threshold. The kept region is everything
    up to the end of that hop. The boundary hop itself may straddle
    speech and silence (its tail samples can be silent and its head
    samples can be loud); we keep the WHOLE hop in that case because
    chopping at the start of the hop would lose real speech.

    If the very last hop in the buffer is the only loud one (i.e. it
    straddles the boundary and there's no fully-loud hop to its
    left), we still keep it — clipping partial-hop speech is
    acceptable, but smuggling silence into the transcription is not.

    Leading silence is NOT touched — only the trailing portion. The
    point is to clip the dead air after the last word, not to scrub
    the buffer clean.

    `hop_samples` defaults to 480 (30ms at 16kHz). Smaller hops are
    more precise but more expensive.
    """
    if audio.size == 0 or threshold <= 0 or hop_samples <= 0:
        return audio
    # Walk backwards in hops. The final hop may be shorter than
    # `hop_samples` so we don't miss any trailing samples.
    n_full = audio.size // hop_samples
    remainder = audio.size - n_full * hop_samples
    # The first hop scanned (from the end) whose RMS >= threshold
    # marks the end of the kept region. We keep that whole hop —
    # including its potentially-silent tail — because the head of the
    # hop is speech we want to keep. The next hop to its right (which
    # we already scanned and found silent) is discarded entirely.
    if remainder > 0:
        tail_start = audio.size - remainder
        if _chunk_rms(audio[tail_start:]) >= threshold:
            return audio  # last partial hop is loud — keep everything
        # tail is silent; check earlier hops
    for i in range(n_full - 1, -1, -1):
        start = i * hop_samples
        end = start + hop_samples
        if _chunk_rms(audio[start:end]) >= threshold:
            return audio[:end]  # keep up to end of this loud hop
    # Every hop is silent — return unchanged so the silence gate in
    # app.py can short-circuit it.
    return audio


class SilenceWatchdog:
    """Auto-stop the recording when N seconds of silence pass.

    Designed to run alongside `App.on_hotkey_press`. The watchdog
    starts on press, polls the recorder's `current_rms()` at ~20 Hz,
    and triggers `on_trigger` exactly once when
    `auto_stop_seconds` of consecutive sub-threshold audio has
    accumulated. The watchdog stops polling as soon as it's triggered
    OR `stop()` is called (which `on_hotkey_release` does).

    Also monitors stream health via `recorder.stream_healthy()`. If
    the PortAudio callback stops delivering audio (common after macOS
    sleep/wake — the HAL silently drops the link without raising or
    closing the stream), the stream is considered dead and
    `on_trigger` fires so the press is finalized instead of hanging
    forever waiting for audio that will never arrive.

    `on_trigger` runs on the watchdog's own thread. The App's
    `on_hotkey_release` is safe to call from any thread — it acquires
    the same `_busy` lock the press path uses, so a manual release
    and a watchdog trigger can never both run the body.
    """

    POLL_INTERVAL_S = 0.05  # 20 Hz

    def __init__(
        self,
        recorder: Recorder,
        threshold: float,
        auto_stop_seconds: float,
        on_trigger: Callable[[], None],
    ) -> None:
        if auto_stop_seconds <= 0:
            raise ValueError("auto_stop_seconds must be > 0 to start a watchdog")
        self._recorder = recorder
        self._threshold = threshold
        # _silence_start is None when the most recent audio was loud;
        # otherwise it's the monotonic time at which the silence run
        # began. When the run reaches `auto_stop_seconds` we trigger.
        self._silence_start: float | None = None
        self._auto_stop_seconds = auto_stop_seconds
        self._on_trigger = on_trigger
        self._stop_event = threading.Event()
        self._triggered = threading.Event()
        self._thread: threading.Thread | None = None
        # Stream health tracking. Set to the monotonic time when
        # `stream_healthy()` first returns False. Cleared when the
        # stream recovers. Used to avoid false-positives on a single
        # slow callback — the stream must be unhealthy for at least
        # `HEALTHY_TIMEOUT_S` before we fire the trigger.
        self._stream_dead_since: float | None = None

    # How long a stream must be unhealthy before we fire the trigger.
    # Longer than `stream_healthy()`'s default 3.0s timeout so we
    # don't race with the health check's own grace period.
    HEALTHY_TIMEOUT_S = 4.0

    def start(self) -> None:
        # Re-startable: if the previous run's thread exited
        # (e.g. after the recorder was torn down on the abort
        # path), a follow-up start() needs to spin up a fresh
        # thread. The old `if self._thread is not None` check
        # would treat a stopped thread (reference still set,
        # is_alive() == False) as "already running" and
        # silently no-op. Each press swaps a fresh
        # SilenceWatchdog in via App._arm_watchdog, so this only
        # matters for the rare reuse-after-stop case.
        if self._thread is not None and self._thread.is_alive():
            return
        # Reset state in case start() is called twice (shouldn't be,
        # but the watchdog is cheap and reset is a no-op after the
        # first start).
        self._stop_event.clear()
        self._triggered.clear()
        self._silence_start = None
        self._thread = None  # drop the stale reference before reassigning
        self._thread = threading.Thread(
            target=self._run, name="silence-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Cancel the watchdog. Safe to call multiple times.

        Doesn't join the thread — it's a daemon, and `on_hotkey_release`
        is on the hot path. Worst case the thread exits one tick
        later, after the release path has already completed."""
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.wait(self.POLL_INTERVAL_S):
            if not self._recorder.is_recording():
                # Recorder closed underneath us; the press loop is over.
                return
            rms = self._recorder.current_rms()
            if rms >= self._threshold:
                self._silence_start = None
            else:
                if self._silence_start is None:
                    self._silence_start = time.monotonic()
                elif time.monotonic() - self._silence_start >= self._auto_stop_seconds:
                    if self._triggered.is_set():
                        return
                    self._triggered.set()
                    self._on_trigger()
                    return
            # Stream health check: detect a dead PortAudio stream
            # (common after macOS sleep/wake). If the HAL stops
            # delivering callbacks, `stream_healthy()` returns False
            # after its own timeout. Once we've seen the stream
            # unhealthy for HEALTHY_TIMEOUT_S, fire the trigger so
            # the press is finalized with whatever audio was buffered
            # — instead of hanging forever waiting for audio that
            # will never arrive.
            try:
                healthy = self._recorder.stream_healthy()
            except Exception:
                healthy = True  # don't let a broken method kill us
            if not healthy:
                if self._stream_dead_since is None:
                    self._stream_dead_since = time.monotonic()
                elif (time.monotonic() - self._stream_dead_since) >= self.HEALTHY_TIMEOUT_S:
                    if self._triggered.is_set():
                        return
                    self._triggered.set()
                    self._on_trigger()
                    return
            else:
                self._stream_dead_since = None
