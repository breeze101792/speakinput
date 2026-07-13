"""Application orchestrator: wires components and owns the lifecycle."""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time

from speakinput.audio import AudioRecorder
from speakinput.config import Config
from speakinput.feedback import Feedback, NullFeedback
from speakinput.hotkey import HotkeyListener, resolve_key
from speakinput.injector import Injector, TypingInjector
from speakinput.models import ModelDownloadError, ModelNotFoundError, ensure_model
from speakinput.transcriber import Transcriber, WhisperCppTranscriber

log = logging.getLogger("speakinput")


def _dbg(enabled: bool, msg: str) -> None:
    """Print a debug line to stderr only when debug mode is on."""
    if enabled:
        print(f"[debug] {msg}", file=sys.stderr, flush=True)


class App:
    def __init__(
        self,
        config: Config,
        recorder: AudioRecorder | None = None,
        transcriber: Transcriber | None = None,
        injector: Injector | None = None,
        feedback: Feedback | None = None,
        dry_run: bool = False,
        debug: bool = False,
    ) -> None:
        self.config = config
        self.recorder = recorder or AudioRecorder(
            sample_rate=config.audio.sample_rate,
            device=config.audio.device,
        )
        # Defer the default transcriber to run() so we can resolve and
        # download the model first. Tests that inject a transcriber don't
        # pay this cost.
        self.transcriber = transcriber
        self.injector = injector or TypingInjector(
            restore_clipboard_ms=config.inject.restore_clipboard_ms,
            trailing_space=config.inject.trailing_space,
        )
        self.feedback = feedback or NullFeedback()
        self.dry_run = dry_run
        self.debug = debug
        self._shutdown = threading.Event()
        self._busy = threading.Lock()
        self._press_started_at: float | None = None
        self.listener: HotkeyListener | None = None

    def _build_default_transcriber(self) -> Transcriber:
        """Resolve the model path (downloading if needed) and construct the
        default WhisperCppTranscriber. Raises ModelNotFoundError /
        ModelDownloadError if the model can't be made available.
        """
        model_path = ensure_model(self.config.stt.model)
        return WhisperCppTranscriber(
            model=model_path,
            language=self.config.stt.language,
            beam_size=self.config.stt.beam_size,
        )

    def on_hotkey_press(self) -> None:
        # Guard against re-entry: if a previous press is still being processed,
        # ignore this press. pynput's latch would also catch this, but a
        # second physical press during processing would still fire.
        if self._busy.locked():
            _dbg(self.debug, "press ignored: already busy")
            return
        self._busy.acquire()
        self._press_started_at = time.monotonic()
        _dbg(self.debug, f"key press start ({self.config.hotkey.key})")
        try:
            self.recorder.start()
        except Exception:
            log.exception("failed to start recorder")
            self._busy.release()
            self._press_started_at = None
            return
        self.feedback.set_state("listening")

    def on_hotkey_release(self) -> None:
        if not self.recorder.is_recording():
            # Press callback failed; nothing to do.
            return
        held_for = (
            time.monotonic() - self._press_started_at if self._press_started_at is not None else 0.0
        )
        self._press_started_at = None
        _dbg(self.debug, f"key press end (held {held_for:.2f}s)")
        self.feedback.set_state("processing")
        try:
            audio = self.recorder.stop()
        except Exception:
            log.exception("failed to stop recorder")
            self._busy.release()
            return
        duration = audio.size / max(self.config.audio.sample_rate, 1)
        import numpy as np  # local import keeps the hot path lean

        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        _dbg(
            self.debug,
            f"audio: {audio.size} samples, {duration:.2f}s, rms={rms:.4f}",
        )
        try:
            text = self.transcriber.transcribe(audio, self.config.audio.sample_rate)
        except Exception:
            log.exception("transcription failed")
            self._busy.release()
            self.feedback.set_state("idle")
            return
        # Always print the transcript in debug mode, even when empty, so the
        # user can tell the difference between "silence" and "stuck".
        if self.debug:
            print(f"[debug] transcript: {text!r}", file=sys.stderr, flush=True)
        if text:
            if self.dry_run:
                print(text, file=sys.stderr, flush=True)
            else:
                try:
                    self.injector.inject(text)
                except Exception:
                    log.exception("injection failed")
        self._busy.release()
        self.feedback.set_state("idle")

    def run(self) -> None:
        # Bootstrap: ensure the model is on disk BEFORE we start the hotkey
        # listener, so the user never sees a 141 MB download start mid-session.
        if self.transcriber is None:
            try:
                self.transcriber = self._build_default_transcriber()
            except ModelNotFoundError as exc:
                print(f"model error: {exc}", file=sys.stderr)
                raise SystemExit(2) from exc
            except ModelDownloadError as exc:
                print(f"model error: {exc}", file=sys.stderr)
                raise SystemExit(2) from exc
        self.feedback.start()
        self.listener = HotkeyListener(
            key=resolve_key(self.config.hotkey.key),
            on_press=self.on_hotkey_press,
            on_release=self.on_hotkey_release,
        )
        self.listener.start()
        log.info(
            "speakinput listening: hold %s to record, release to inject",
            self.config.hotkey.key,
        )
        if self.debug:
            _dbg(True, "debug mode ON — every key event and transcript will be logged to stderr")
        # Install signal handlers so Ctrl-C cleans up the recorder.
        signal.signal(signal.SIGINT, lambda *_: self._shutdown.set())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown.set())
        try:
            self._shutdown.wait()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._shutdown.set()
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        try:
            self.feedback.stop()
        except Exception:
            log.exception("feedback stop failed")
