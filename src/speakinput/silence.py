"""Silence detection helpers.

Two pieces:

- `trim_trailing_silence`: walk a recorded buffer backwards in
  ~30ms hops and drop any trailing portion whose per-hop RMS is below
  `threshold`. The leading portion is preserved. Useful as a
  pre-transcribe cleanup step so whisper doesn't see a long silent
  tail (a real hallucination source).

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
from typing import Callable

import numpy as np

from speakinput.audio import Recorder

# How wide a slice to look at when measuring RMS for the trailing
# silence trim. 30ms is short enough that we don't accidentally chop
# the end of a word, long enough to swallow a typical consonant
# release or breath. At 16kHz that's 480 samples.
_TRIM_HOP_SAMPLES = 480


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
                continue
            if self._silence_start is None:
                self._silence_start = time.monotonic()
                continue
            if time.monotonic() - self._silence_start >= self._auto_stop_seconds:
                # First to claim the trigger wins. Either we or the
                # manual release (which calls stop()) will be first;
                # the loser's no-op.
                if self._triggered.is_set():
                    return
                self._triggered.set()
                self._on_trigger()
                return
