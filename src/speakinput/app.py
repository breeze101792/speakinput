"""Application orchestrator: wires components and owns the lifecycle."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from speakinput.audio import AudioRecorder
from speakinput.config import Config, Profile
from speakinput.feedback import Feedback, NullFeedback
from speakinput.hotkey import (
    EvdevHotkeyListener,
    HotkeyListener,
    resolve_evdev_key,
    resolve_key,
)
from speakinput.injector import Injector, TypingInjector
from speakinput.models import (
    ModelDownloadError,
    ModelNotFoundError,
    ensure_model,
    resolve_for_language,
)
from speakinput.silence import SilenceWatchdog, trim_trailing_silence
from speakinput.transcriber import Transcriber, WhisperCppTranscriber

log = logging.getLogger("speakinput")


def _dbg(enabled: bool, msg: str) -> None:
    """Print a debug line to stderr only when debug mode is on."""
    if enabled:
        print(f"[debug] {msg}", file=sys.stderr, flush=True)


def _build_transcribers(
    profiles: list[Profile],
    transcriber_overrides: dict[str, Transcriber] | None = None,
) -> dict[str, Transcriber]:
    """Construct one Transcriber per profile, sharing instances by model path.

    Two profiles that resolve to the same model file share a single
    `WhisperCppTranscriber` — pywhispercpp loads the model eagerly in
    its constructor, so a shared instance means one copy in RAM
    (~466 MB for `small`). The `transcribe()` call's `language` and
    `initial_prompt` are per-call, not per-instance, so sharing is safe.

    `ensure_model` is also deduped: when two profiles pick the same
    model name we only print the "checking model..." / "model ready"
    lines once.

    `transcriber_overrides` lets tests inject mock transcribers keyed by
    the profile's hotkey (e.g. `{"alt_r": mock1, "cmd_r": mock2}`).
    """
    overrides = transcriber_overrides or {}
    by_name: dict[str, Path] = {}
    by_path: dict[str, Transcriber] = {}
    by_key: dict[str, Transcriber] = {}
    for profile in profiles:
        if profile.key in overrides:
            by_key[profile.key] = overrides[profile.key]
            continue
        # Auto-upgrade: an English-only model with a non-English language
        # (or `auto`) gets swapped for the same-tier multilingual model
        # before we try to download it. The user is told via the message.
        model_name, upgrade_msg = resolve_for_language(profile.model, profile.language)
        if upgrade_msg and model_name not in by_name:
            print(f"[info] {upgrade_msg}", file=sys.stderr, flush=True)
        # Reuse the resolved path if a previous profile already chose
        # the same model name. Avoids redundant "checking model..." lines
        # and a redundant on-disk lookup.
        if model_name in by_name:
            model_path = by_name[model_name]
        else:
            model_path = ensure_model(model_name)
            by_name[model_name] = model_path
        cached = by_path.get(str(model_path))
        if cached is not None:
            by_key[profile.key] = cached
        else:
            t = WhisperCppTranscriber(
                model=model_path,
                language=profile.language,
                beam_size=profile.beam_size,
                initial_prompt=profile.initial_prompt,
            )
            by_path[str(model_path)] = t
            by_key[profile.key] = t
    return by_key


class App:
    def __init__(
        self,
        config: Config,
        recorder: AudioRecorder | None = None,
        transcribers: dict[str, Transcriber] | None = None,
        injector: Injector | None = None,
        feedback: Feedback | None = None,
        dry_run: bool = False,
        debug: bool = False,
        config_source: Path | None = None,
    ) -> None:
        self.config = config
        # Path the config was loaded from, or None when running on
        # baked-in defaults (no user-edited file was found). Surfaced in
        # the startup banner so the user can verify whether their
        # config.toml is actually being read.
        self.config_source = config_source
        self.recorder = recorder or AudioRecorder(
            sample_rate=config.audio.sample_rate,
            device=config.audio.device,
        )
        # Profiles in (key, model, language, prompt) order, primary first.
        # Used to build the per-key transcribers and hotkey listeners.
        self._profiles: list[Profile] = [config.primary]
        if config.secondary is not None:
            self._profiles.append(config.secondary)
        # Defer default-transcriber construction to run() so we can
        # resolve and download models first. Tests inject transcribers
        # via the `transcribers` kwarg and skip the bootstrap step.
        self.transcribers: dict[str, Transcriber] = transcribers or {}
        self.injector = injector or TypingInjector(
            restore_clipboard_ms=config.inject.restore_clipboard_ms,
            trailing_space=config.inject.trailing_space,
        )
        self.feedback = feedback or NullFeedback()
        self.dry_run = dry_run
        self.debug = debug
        self._shutdown = threading.Event()
        self._busy = threading.Lock()
        # Serializes the chunked-release body: when auto-stop fires
        # mid-press it drains, transcribes, and re-arms the watchdog.
        # A manual release that arrives during the chunked body must
        # wait for it to finish before tearing the recorder down.
        # Acquired at the start of the chunked body and held across the
        # whole drain→process→re-arm sequence. NOT held during
        # transcribe+inject (the injection is the slow part, and we
        # want the watchdog's NEXT chunk to be allowed to start
        # recording the user's next sentence while the previous one is
        # being typed out).
        self._body_lock = threading.Lock()
        self._press_started_at: float | None = None
        # The profile that's currently being recorded. Set on press,
        # consumed on release, cleared at the end. None outside of a
        # press/release pair.
        self._active_profile: Profile | None = None
        # True when the user has released the hotkey and we're in the
        # final-cleanup window. The chunked body checks this between
        # drains so a manual release finalizes the session without
        # re-arming the watchdog for another chunk.
        self._manual_release_pending: bool = False
        # Auto-stop watchdog. Started in on_hotkey_press when
        # `auto_stop_seconds > 0`, stopped in on_hotkey_release. None
        # when the feature is disabled. Replaced (not appended) on
        # every chunked re-arm so a stale watchdog is never running.
        self._watchdog: SilenceWatchdog | None = None
        self.listeners: dict[str, HotkeyListener] = {}

    @property
    def active_profile(self) -> Profile | None:
        return self._active_profile

    def on_hotkey_press(self, profile: Profile) -> None:
        # Guard against re-entry: if a previous press is still being processed,
        # ignore this press. pynput's latch would also catch this, but a
        # second physical press during processing would still fire.
        if self._busy.locked():
            _dbg(self.debug, f"press ignored: already busy (key={profile.key})")
            return
        self._busy.acquire()
        self._active_profile = profile
        self._manual_release_pending = False
        self._press_started_at = time.monotonic()
        _dbg(self.debug, f"key press start ({profile.key})")
        try:
            self.recorder.start()
        except Exception:
            log.exception("failed to start recorder")
            self._busy.release()
            self._press_started_at = None
            self._active_profile = None
            return
        # Start the auto-stop watchdog if the user has it enabled. The
        # watchdog polls the recorder's live RMS in a background thread
        # and, when N seconds of silence pass, calls _on_watchdog_chunk
        # which drains the captured audio, transcribes+injects it, and
        # re-arms a fresh watchdog for the next chunk. The manual
        # release path (on_hotkey_release) sets a pending flag and
        # finalizes the session through _finalize().
        auto_stop = self.config.audio.auto_stop_seconds
        if auto_stop > 0 and self.config.audio.silence_threshold > 0:
            self._arm_watchdog(profile)
        self.feedback.set_state("listening")

    def _arm_watchdog(self, profile: Profile) -> None:
        """Create and start a fresh SilenceWatchdog for the current chunk.

        Replaces any prior watchdog. Called by the chunked body after
        each successful drain+inject so a stale watchdog is never
        running. The watchdog's on_trigger is _on_watchdog_chunk.
        """
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None
        wd = SilenceWatchdog(
            recorder=self.recorder,
            threshold=self.config.audio.silence_threshold,
            auto_stop_seconds=self.config.audio.auto_stop_seconds,
            on_trigger=lambda: self._on_watchdog_chunk(profile),
        )
        wd.start()
        self._watchdog = wd

    def _on_watchdog_chunk(self, profile: Profile) -> None:
        """Watchdog fired: silence threshold reached mid-press.

        Drain the captured buffer, transcribe and inject it, then
        re-arm a fresh watchdog for the next sentence. If the user has
        released the hotkey in the meantime, finalize the session
        instead of re-arming.

        Runs on the watchdog's own thread. The _body_lock is acquired
        only across the brief drain→re-arm window; transcribe+inject
        runs OUTSIDE the lock so the next chunk can start recording
        while the previous one is being typed out.
        """
        if self._manual_release_pending:
            # Manual release beat us to the finalization. Just exit;
            # on_hotkey_release will tear the recorder down.
            return
        with self._body_lock:
            if self._manual_release_pending:
                return
            audio = self.recorder.drain()
            if self._manual_release_pending:
                return
        # Transcribe+inject OUTSIDE the body lock so the recorder can
        # start capturing the next chunk while whisper chews on this
        # one. The chunked path keeps the recorder running.
        if audio.size:
            self._process_and_inject(audio, profile)
        # Re-arm only if the user is still holding the key. If they
        # released during the transcribe, the manual-release path
        # will tear the recorder down and we're done.
        if self._manual_release_pending:
            return
        # Don't re-arm if the recorder has been closed out from under
        # us (e.g. shutdown() or finalize() ran on another thread).
        if not self.recorder.is_recording():
            return
        self._arm_watchdog(profile)

    def on_hotkey_release(self, profile: Profile) -> None:
        """Final release path: the user has let go of the hotkey.

        Sets a flag the chunked body checks, stops any active watchdog,
        and runs the final drain+transcribe+inject+close. If the
        chunked body is mid-flight, it sees the flag and bails out
        before re-arming; the finalize below picks up whatever audio
        accumulated during the transcribe.
        """
        if not self.recorder.is_recording():
            # Press callback failed; nothing to do.
            return
        # Tell the chunked body (if any) to bail on re-arming.
        self._manual_release_pending = True
        # Stop the active watchdog. If the watchdog has already fired
        # and is in _on_watchdog_chunk, stopping it has no effect on
        # the chunk body (it doesn't re-check the stop event), but the
        # _arm_watchdog call at the end of the chunk body is a no-op
        # for the final release because we check the flag right after.
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None
        if self._active_profile is not profile:
            # Press and release keys don't match (shouldn't happen with
            # one key held at a time, but guard anyway). Use the active
            # profile for the final transcribe+inject.
            profile = self._active_profile
        held_for = (
            time.monotonic() - self._press_started_at if self._press_started_at is not None else 0.0
        )
        self._press_started_at = None
        _dbg(self.debug, f"key press end (held {held_for:.2f}s, key={profile.key})")
        self.feedback.set_state("processing")
        self._finalize(profile)

    def _finalize(self, profile: Profile | None) -> None:
        """Drain whatever audio is buffered, transcribe+inject, tear
        down the recorder, and release the busy lock.

        Called by on_hotkey_release (manual) after a final drain.
        Idempotent-ish: if the recorder isn't recording, just releases
        the busy lock and returns.
        """
        try:
            audio = self.recorder.drain() if self.recorder.is_recording() else np.zeros(0, dtype=np.float32)
        except Exception:
            log.exception("failed to drain recorder")
            audio = np.zeros(0, dtype=np.float32)
        try:
            self.recorder.close()
        except Exception:
            log.exception("failed to close recorder")
        if audio.size:
            self._process_and_inject(audio, profile)
        self._busy.release()
        self._active_profile = None
        self._manual_release_pending = False
        self._watchdog = None
        self.feedback.set_state("idle")

    def _process_and_inject(self, audio: np.ndarray, profile: Profile | None) -> None:
        """Trim, silence-gate, transcribe, and inject a chunk of audio.

        Pulled out of the release path so the chunked watchdog body
        and the final release body share it. Silent short-circuits
        are normal (the user spoke but auto-stopped during a brief
        pause) — debug-mode logs make the difference between "silence
        skipped" and "stuck" observable.
        """
        if not self.recorder.is_recording() and not audio.size:
            return
        if profile is None:
            profile = self._active_profile
        if profile is None:
            log.warning("no active profile; skipping inject")
            return
        import numpy as np  # local import keeps the hot path lean

        duration = audio.size / max(self.config.audio.sample_rate, 1)
        # Trim trailing silence BEFORE computing the silence-gate RMS,
        # so a long silent tail doesn't fool the gate into thinking the
        # whole buffer is silent. The trim only chops trailing samples,
        # never the leading portion of the speech.
        threshold = self.config.audio.silence_threshold
        if threshold > 0 and audio.size:
            trimmed = trim_trailing_silence(
                audio, self.config.audio.sample_rate, threshold
            )
            if trimmed.size != audio.size:
                _dbg(
                    self.debug,
                    f"trimmed trailing silence: {audio.size} -> {trimmed.size} samples "
                    f"({(audio.size - trimmed.size) / max(self.config.audio.sample_rate, 1):.2f}s)",
                )
                audio = trimmed
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        _dbg(
            self.debug,
            f"audio: {audio.size} samples, {duration:.2f}s, rms={rms:.4f}",
        )
        # Silence gate: short-circuit near-empty recordings so whisper
        # doesn't hallucinate on them. The default 0.005 is well above
        # the noise floor of a quiet room but well below any actual
        # speech. Set to 0 in config.toml to disable.
        if threshold > 0 and rms < threshold:
            _dbg(
                self.debug,
                f"silence gate: rms={rms:.4f} < {threshold:.4f}, skipping transcribe",
            )
            return
        try:
            transcriber = self.transcribers[profile.key]
        except KeyError:
            log.exception("no transcriber wired for key %r", profile.key)
            return
        try:
            text = transcriber.transcribe(audio, self.config.audio.sample_rate)
        except Exception:
            log.exception("transcription failed")
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

    def _make_press_cb(self, profile: Profile):
        def cb() -> None:
            self.on_hotkey_press(profile)
        return cb

    def _make_release_cb(self, profile: Profile):
        def cb() -> None:
            self.on_hotkey_release(profile)
        return cb

    def run(self) -> None:
        # Bootstrap: ensure models are on disk BEFORE we start the hotkey
        # listener, so the user never sees a 141 MB download start mid-session.
        # If the test (or another caller) injected transcribers, skip.
        if not self.transcribers:
            try:
                self.transcribers = _build_transcribers(self._profiles)
            except ModelNotFoundError as exc:
                print(f"model error: {exc}", file=sys.stderr)
                raise SystemExit(2) from exc
            except ModelDownloadError as exc:
                print(f"model error: {exc}", file=sys.stderr)
                raise SystemExit(2) from exc
        self._print_banner()
        self.feedback.start()
        use_evdev = (
            sys.platform == "linux"
            and os.environ.get("XDG_SESSION_TYPE") == "wayland"
        )
        # Print the chosen hotkey backend so the user can verify which
        # code path is active (helpful for permission issues on Wayland).
        if use_evdev:
            print(
                "[startup] hotkey   : evdev (Linux Wayland session — pynput bypassed)",
                file=sys.stderr,
                flush=True,
            )
        for profile in self._profiles:
            if use_evdev:
                listener = EvdevHotkeyListener(
                    keycode=resolve_evdev_key(profile.key),
                    on_press=self._make_press_cb(profile),
                    on_release=self._make_release_cb(profile),
                )
            else:
                listener = HotkeyListener(
                    key=resolve_key(profile.key),
                    on_press=self._make_press_cb(profile),
                    on_release=self._make_release_cb(profile),
                )
            self.listeners[profile.key] = listener
            listener.start()
        keys = ", ".join(p.key for p in self._profiles)
        log.info("speakinput listening: hold %s to record, release to inject", keys)
        if self.debug:
            _dbg(True, "debug mode ON — every key event and transcript will be logged to stderr")
        # Install signal handlers so Ctrl-C cleans up the recorder.
        signal.signal(signal.SIGINT, lambda *_: self._shutdown.set())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown.set())
        try:
            self._shutdown.wait()
        finally:
            self.shutdown()

    def _print_banner(self) -> None:
        """Print a one-line-per-field startup summary so the user can verify
        the active config without having to open config.toml. Each line is
        independent so it's easy to grep."""
        cfg = self.config
        device = cfg.audio.device if cfg.audio.device is not None else "default"
        inject_mode = "off (dry-run)" if self.dry_run else "on"
        threshold = cfg.audio.silence_threshold
        threshold_str = "off" if threshold == 0 else f"{threshold:g}"
        auto_stop = cfg.audio.auto_stop_seconds
        auto_stop_str = "off" if auto_stop == 0 else f"{auto_stop:g}s"
        if self.config_source is not None:
            source_str = str(self.config_source)
        else:
            source_str = "(defaults — no config.toml found)"
        # Profile lines: one per profile, listing the STT settings; a
        # closing summary line tells the user how many distinct model
        # files we actually loaded (so they can see when their dedup
        # worked).
        profile_lines = []
        for i, p in enumerate(self._profiles, start=1):
            prompt = "off" if not p.initial_prompt else "set"
            profile_lines.append(
                f"profile {i} : key={p.key} model={p.model} "
                f"language={p.language} prompt={prompt}"
            )
        distinct = len({id(t) for t in self.transcribers.values()})
        total = len(self.transcribers)
        dedupe_str = (
            f" (shared: 1 transcriber, {total} profiles)"
            if total > 1 and distinct == 1
            else ""
        )
        lines = [
            f"config   : {source_str}",
            *profile_lines,
            f"models   : loaded {distinct} into memory{dedupe_str}",
            f"sample   : {cfg.audio.sample_rate} Hz, device={device}",
            f"silence  : rms<{threshold_str} -> skip; auto-stop after {auto_stop_str}",
            f"inject   : {inject_mode}, trailing_space={cfg.inject.trailing_space}",
        ]
        for line in lines:
            print(f"[startup] {line}", file=sys.stderr, flush=True)

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None
        for key, listener in self.listeners.items():
            try:
                listener.stop()
            except Exception:
                log.exception("listener stop failed (key=%s)", key)
        # Keep the listeners dict around so post-shutdown tests can
        # verify which keys were wired. pynput's Listener objects are
        # safe to leave stopped but referenced.
        try:
            self.feedback.stop()
        except Exception:
            log.exception("feedback stop failed")
