"""Application orchestrator: wires components and owns the lifecycle."""

from __future__ import annotations

import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from speakinput.audio import AudioError, AudioRecorder
from speakinput.config import Config, Profile
from speakinput.feedback import Feedback, NullFeedback
from speakinput.hotkey import (
    EvdevHotkeyListener,
    HotkeyListener,
    probe_evdev_available,
    resolve_evdev_key,
    resolve_key,
)
from speakinput.injector import Injector, select_injector_with_fallback
from speakinput.media import MediaController
from speakinput.models import (
    ModelDownloadError,
    ModelNotFoundError,
    ensure_model,
    resolve_for_language,
)
from speakinput.silence import SilenceWatchdog, analyse_speech, trim_trailing_silence
from speakinput.transcriber import Transcriber, WhisperCppTranscriber, _gpu_summary, create_transcriber

log = logging.getLogger("speakinput")

# Lazy-initialised OpenCC converters.
# We use the pure-Python reimplementation to avoid native build deps.
_OPENCC_S2T: Any = None
_OPENCC_T2S: Any = None
# Guards the OpenCC cache against a double-build race: the worker
# thread and a watchdog chunk body can both call `_opencc("s2t")`
# in parallel and each construct a fresh OpenCC object before either
# stores it. The winner is fine, the loser's converter is GC'd, but
# the wasted model load is ~50ms and shows up as a one-shot
# latency spike on the first Chinese-text press. Cheap to lock.
_OPENCC_LOCK = threading.Lock()


def _opencc(direction: str) -> Any | None:
    global _OPENCC_S2T, _OPENCC_T2S
    with _OPENCC_LOCK:
        cache = _OPENCC_S2T if direction == "s2t" else _OPENCC_T2S
        if cache is not None:
            return cache if cache is not False else None
        try:
            from opencc import OpenCC
            cc = OpenCC(direction)
            if direction == "s2t":
                _OPENCC_S2T = cc
            else:
                _OPENCC_T2S = cc
            return cc
        except Exception:
            log.warning("opencc not available; zh_conversion disabled")
            if direction == "s2t":
                _OPENCC_S2T = False
            else:
                _OPENCC_T2S = False
            return None


def _contains_chinese(text: str) -> bool:
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def _convert_text(text: str, direction: str) -> str:
    if direction == "off":
        return text
    cc = _opencc("s2t" if direction == "traditional" else "t2s")
    if cc is None:
        return text
    try:
        return cc.convert(text)
    except Exception:
        log.exception("opencc conversion failed")
        return text


def _dbg(enabled: bool, msg: str) -> None:
    """Print a debug line to stderr only when debug mode is on."""
    if enabled:
        print(f"[debug] {msg}", file=sys.stderr, flush=True)


def _bounded_call(fn, *args, timeout_s: float, name: str, **kwargs) -> tuple[bool, Any]:
    """Run `fn(*args, **kwargs)` on a helper thread with a bounded join.

    Returns `(success, result)`. If `fn` returns within `timeout_s`,
    `success` is True and `result` is the return value. If it doesn't,
    `success` is False and `result` is None — the helper thread is a
    daemon and is abandoned (it dies with the process, or finishes
    later and its result is discarded).

    Used to bound native calls that can hang indefinitely: whisper
    inference (GPU hang, CUDA OOM recovery, Vulkan device-lost, Metal
    command buffer stuck) and pynput inject (HID event system wedged
    after sleep, focused app unresponsive). A hang in either holds
    the event worker thread and the busy lock forever — every later
    press is silently ignored as "already busy" and the only way out
    is a manual restart. The bounded call turns that hang into a
    recoverable failure: the caller logs, releases the lock, and the
    next press works.

    The thread is a daemon so it never blocks process exit. The
    `name` parameter shows up in thread dumps (faulthandler on the
    second Ctrl-C) so the user can see which call hung.
    """
    if timeout_s <= 0:
        return True, fn(*args, **kwargs)
    box: list[Any] = [None]
    error: list[BaseException | None] = [None]
    done = threading.Event()

    def _runner() -> None:
        try:
            box[0] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            error[0] = exc
        finally:
            done.set()

    t = threading.Thread(target=_runner, name=f"speakinput-{name}", daemon=True)
    t.start()
    if done.wait(timeout_s):
        if error[0] is not None:
            raise error[0]
        return True, box[0]
    print(
        f"[warn] {name} did not finish in {timeout_s:.0f}s — abandoning "
        f"it and continuing. This usually follows a GPU hang or a wedged "
        f"HID/clipboard subsystem.",
        file=sys.stderr,
        flush=True,
    )
    return False, None


def _probe_evdev_or_diag() -> tuple[bool, str | None]:
    """Run the evdev availability probe and return both the result AND
    the error message if it failed.

    `probe_evdev_available()` in `speakinput.hotkey` only returns a bool,
    which silently discards the `HotkeyError` reason. The user then sees
    a missing/quiet pynput-fallback banner and has no way to know why
    evdev wasn't picked. We re-implement the probe here so we can keep
    the diagnostic string and surface it in the startup banner.

    The second call to `find_keyboard_device()` exists only to capture
    the `HotkeyError` reason; if it returns a device (race condition
    where a keyboard appeared between the probe and this call), we
    close the device before discarding it so we don't leak an fd.
    """
    from speakinput.hotkey import find_keyboard_device

    if probe_evdev_available():
        return True, None
    try:
        dev = find_keyboard_device()
    except Exception as exc:  # HotkeyError or anything else
        return False, str(exc)
    # find_keyboard_device returned — close the device to avoid leaking
    # an fd. This branch is only hit on a race condition (the probe
    # failed but a device appeared moments later); on a healthy system
    # the probe and this call agree.
    try:
        dev.close()  # type: ignore[attr-defined]
    except Exception:
        pass
    return False, "evdev probe inconsistency: probe failed but find_keyboard_device succeeded"


class _LivenessWatcher:
    """Background thread that polls listener liveness every `interval_s`.

    Each `HotkeyListener` (pynput) and `EvdevHotkeyListener` runs its
    own run loop on a daemon thread. If the run loop returns or
    raises, the thread dies and the hotkey stops working — but the
    process is still alive, so the user has no way to tell from the
    outside. This watcher checks `is_alive()` periodically and
    invokes `on_dead(key)` the first time a listener transitions from
    alive to dead. One warning per death; not spammy.

    It also detects system sleep. macOS (and most laptops on lid-close)
    can suspend the process for an unbounded time; on wake,
    `time.monotonic()` has barely advanced (it freezes while suspended)
    but the wall clock has jumped. When the skew between the two
    exceeds `sleep_threshold_s`, `on_sleep(skew_s)` is invoked — macOS
    disables CGEventTaps across sleep, so the listeners must be
    recreated even though their threads still look alive (which is why
    the thread-liveness check alone never notices this case).

    The watcher is itself a daemon thread and exits when `stop()` is
    called or when the process is about to exit.
    """

    # If a listener's thread is alive but no events have arrived for
    # this many seconds, treat it as dead. macOS can disable the
    # CGEventTap (brief sleep/wake, focus churn, permission transition)
    # and the pynput run loop will just time out without ever
    # delivering events; this is the backstop for that case when the
    # self-healing tap re-enable in `_mac_tap_heal.py` can't recover
    # (e.g. Input Monitoring permission permanently revoked, the tap
    # object is gone). 30s is long enough to not false-positive on a
    # user who pauses to think, but short enough that a post-wake
    # dead listener is recovered quickly — the self-healing re-enables
    # the tap within ~1s on success; if it fails, this threshold
    # catches it within 30s instead of 5 minutes.
    _STALE_EVENT_THRESHOLD_S = 30.0

    def __init__(
        self,
        listeners: list,
        interval_s: float,
        on_dead,
        on_sleep=None,
        sleep_threshold_s: float = 5.0,
        worker_thread: threading.Thread | None = None,
        on_worker_dead=None,
        busy_lock: threading.Lock | None = None,
        press_started_at=None,
        busy_lock_timeout_s: float = 0.0,
        on_busy_stuck=None,
    ) -> None:
        self._listeners = listeners
        self._interval_s = interval_s
        self._on_dead = on_dead
        self._on_sleep = on_sleep
        # The sleep threshold used to be 15s because most laptops
        # don't sleep for less than that, and the rare sub-15s suspend
        # (a quick lid-close) wasn't worth the false-positive risk of
        # restarting the listener. In practice macOS disables the
        # CGEventTap on ANY sleep, including sub-5s lid-close / clamshell
        # transitions, and the listener can't recover on its own (the
        # self-heal only re-enables a still-present tap). Lowering the
        # threshold to 5s catches those events and triggers the
        # restart path so the user doesn't have to relaunch the app.
        self._sleep_threshold_s = sleep_threshold_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Map of listener -> True. We track "was alive" to fire
        # `on_dead` exactly once per transition (alive -> dead), not
        # every poll after the thread dies.
        self._was_alive: dict[int, bool] = {id(lst): True for lst in listeners}
        # Set to True the first time the heartbeat check fires for a
        # given listener, so we only call `on_dead` ONCE per wedged
        # episode (subsequent polls see the listener is being
        # restarted and don't double-fire). Reset to False by
        # `swap()` when the listener is recreated.
        self._heartbeat_wedged: dict[int, bool] = {}
        # Clock samples for the sleep detector. `monotonic` freezes
        # while the machine is suspended; `wall` keeps going. Both are
        # re-sampled every poll tick.
        self._last_mono: float | None = None
        self._last_wall: float | None = None
        # Event worker liveness monitoring. The worker thread runs
        # press/release bodies; if it dies (an edge case the try/except
        # in _event_worker_main doesn't catch, or a future bug), every
        # press is silently enqueued and never processed. The watcher
        # polls `worker_thread.is_alive()` and fires `on_worker_dead`
        # once on the alive->dead transition so the App can restart it.
        self._worker_thread = worker_thread
        self._on_worker_dead = on_worker_dead
        self._worker_was_alive = True
        # Busy-lock watchdog. The universal backstop for ANY hang in
        # the processing pipeline: a transcribe hang, an inject hang,
        # a model reload hang, or any future code path that holds the
        # busy lock too long. When the lock has been held continuously
        # for longer than `busy_lock_timeout_s`, `on_busy_stuck` is
        # invoked so the App can fire _abort_press and release the
        # lock. Without this, a stuck press makes every later press
        # silently ignored as "already busy" and the only way out is
        # a manual restart. `press_started_at` is a callable returning
        # the monotonic timestamp the current press started (or None
        # when no press is active) so the watcher can read it fresh
        # each tick without holding a reference to a stale closure.
        self._busy_lock = busy_lock
        self._press_started_at = press_started_at
        self._busy_lock_timeout_s = busy_lock_timeout_s
        self._on_busy_stuck = on_busy_stuck
        # Fire on_busy_stuck at most once per stuck episode so we
        # don't enqueue multiple aborts. Reset when the lock is
        # released (the next poll sees it unheld).
        self._busy_stuck_fired = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="speakinput-liveness", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def swap(self, old, new) -> None:
        """Replace a listener that was just recreated (restart path).

        Keeps the watcher's list and the alive-transition tracking in
        sync with `App.listeners` after a dead/tap-disabled listener is
        swapped for a fresh one. The new listener starts out "alive" so
        a future death fires `on_dead` exactly once.
        """
        for i, lst in enumerate(self._listeners):
            if lst is old:
                self._listeners[i] = new
                break
        else:
            self._listeners.append(new)
        self._was_alive.pop(id(old), None)
        self._was_alive[id(new)] = True
        # Clear the heartbeat-wedged flag so the freshly recreated
        # listener gets a fresh grace period before the next backstop
        # restart fires.
        self._heartbeat_wedged.pop(id(old), None)
        self._heartbeat_wedged[id(new)] = False

    def swap_worker(self, new_worker: threading.Thread) -> None:
        """Replace the event worker thread after a restart.

        Called by `App._restart_event_worker` after spinning up a fresh
        worker. Resets the alive-transition tracking so the new worker's
        future death fires `on_worker_dead` exactly once.
        """
        self._worker_thread = new_worker
        self._worker_was_alive = True

    def _check_sleep(self) -> None:
        now_mono = time.monotonic()
        now_wall = time.time()
        last_mono, last_wall = self._last_mono, self._last_wall
        self._last_mono, self._last_wall = now_mono, now_wall
        if last_mono is None or last_wall is None or self._on_sleep is None:
            return
        skew = (now_wall - last_wall) - (now_mono - last_mono)
        if skew > self._sleep_threshold_s:
            try:
                self._on_sleep(skew)
            except Exception:
                log.exception("on_sleep callback failed")

    def _run(self) -> None:
        self._last_mono = time.monotonic()
        self._last_wall = time.time()
        while not self._stop_event.wait(self._interval_s):
            try:
                self._check_sleep()
                self._check_worker()
                self._check_busy_lock()
                now = time.monotonic()
                for listener in self._listeners:
                    key = getattr(listener, "_key", None) or getattr(
                        listener, "_keycode", "?"
                    )
                    # `is_running()` is the authoritative liveness check for
                    # both backends: `HotkeyListener` wraps pynput's listener
                    # (which exposes `is_alive()` on the underlying thread)
                    # and `EvdevHotkeyListener` wraps a `threading.Thread`.
                    # Both check the actual thread liveness, so the extra
                    # `_thread.is_alive()` lookup below is redundant and
                    # broken for the pynput backend (which has no `_thread`
                    # attribute — that name belongs to the evdev listener
                    # only). Using `is_running()` alone fixes a false-positive
                    # "listener is dead" warning on macOS.
                    alive = bool(listener.is_running())
                    was = self._was_alive.get(id(listener), True)
                    if was and not alive:
                        try:
                            self._on_dead(key)
                        except Exception:
                            log.exception("on_dead callback failed (key=%r)", key)
                    self._was_alive[id(listener)] = alive
                    # Heartbeat backstop: thread is alive but the
                    # listener hasn't delivered any event for a long
                    # time. This catches the macOS CGEventTap being
                    # disabled in a way the self-heal in
                    # `_mac_tap_heal.py` can't fix (e.g. Input
                    # Monitoring permission permanently revoked, the
                    # tap object gone). The threshold is long enough
                    # to not false-positive on a user who walks away
                    # from the keyboard to think about what to say
                    # next, but short enough that a wedged tap is
                    # recovered within a few minutes of the user
                    # sitting back down.
                    if alive and not self._heartbeat_wedged.get(id(listener), False):
                        last_evt = listener.last_event_at()
                        if last_evt is None:
                            # No events ever — the listener has been
                            # alive for at least one poll cycle
                            # without seeing a single key. Treat as
                            # freshly-started (the on_dead above
                            # handles the "never came up" case via
                            # `is_running()`); don't double-fire.
                            continue
                        if (now - last_evt) >= self._STALE_EVENT_THRESHOLD_S:
                            self._heartbeat_wedged[id(listener)] = True
                            try:
                                self._on_dead(key)
                            except Exception:
                                log.exception(
                                    "on_dead callback failed on heartbeat (key=%r)",
                                    key,
                                )
            except Exception:
                # A watcher that dies silently is worse than any single
                # bad poll — keep looping so the next tick gets a shot.
                log.exception("liveness poll failed")

    def _check_worker(self) -> None:
        """Poll the event worker thread and fire `on_worker_dead` once.

        The event worker runs press/release bodies on a single thread.
        If it dies (an edge case the try/except in
        `_event_worker_main` doesn't catch, or a future bug), every
        press is silently enqueued and never processed — the user's
        hotkey does nothing and there's no visible signal. The watcher
        polls `worker_thread.is_alive()` and fires `on_worker_dead`
        on the alive->dead transition so the App can restart it.
        """
        if self._worker_thread is None or self._on_worker_dead is None:
            return
        alive = self._worker_thread.is_alive()
        if self._worker_was_alive and not alive:
            self._worker_was_alive = False
            try:
                self._on_worker_dead()
            except Exception:
                log.exception("on_worker_dead callback failed")
        # Update the tracked state so a future restart (swap_worker)
        # re-arms the transition. Don't set alive=True here when the
        # thread is alive — swap_worker does that explicitly.
        if not alive:
            return

    def _check_busy_lock(self) -> None:
        """Universal backstop: abort a press that's been stuck too long.

        If the busy lock has been held continuously for longer than
        `busy_lock_timeout_s`, fire `on_busy_stuck` so the App can
        abort the press and release the lock. This catches ANY hang in
        the processing pipeline — a transcribe hang, an inject hang, a
        model reload hang, or any future code path that holds the lock
        too long. Without it, a stuck press makes every later press
        silently ignored as "already busy" and the only way out is a
        manual restart.

        Fires at most once per stuck episode. The flag resets when the
        lock is released (the next poll sees it unheld).
        """
        if (
            self._busy_lock is None
            or self._on_busy_stuck is None
            or self._busy_lock_timeout_s <= 0
        ):
            return
        held = self._busy_lock.locked()
        if not held:
            self._busy_stuck_fired = False
            return
        # Read the press start time fresh each tick via the callable
        # so we always see the current state (the App mutates
        # `_press_started_at` directly on press/abort).
        started = self._press_started_at() if self._press_started_at is not None else None
        if started is None:
            # Lock is held but no press is recorded — the press
            # callback is mid-acquire (between `acquire()` and setting
            # `_press_started_at`). This window is microseconds; skip
            # this tick and check again next poll.
            return
        if self._busy_stuck_fired:
            return
        held_for = time.monotonic() - started
        if held_for >= self._busy_lock_timeout_s:
            self._busy_stuck_fired = True
            try:
                self._on_busy_stuck(held_for)
            except Exception:
                log.exception("on_busy_stuck callback failed")


class _Heartbeat:
    """Background thread that prints a "still here" line every interval.

    Debug-only — confirms the main loop is alive when the user comes
    back to a terminal that hasn't moved in a while (e.g. after sleep).
    Cheap: one print + one wait per interval.
    """

    def __init__(self, interval_s: float) -> None:
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = time.monotonic()

    def start(self) -> None:
        # Re-startable: if stop() was called and the thread exited,
        # a follow-up start() should bring it back. The old
        # `if self._thread is not None` check treated a stopped
        # thread (reference still set, but is_alive() == False) as
        # "already running" and silently no-op'd. `_LivenessWatcher`
        # does the right thing by always creating a new
        # `_LivenessWatcher` instance, so this only matters for the
        # heartbeat and watchdog.
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="speakinput-heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            uptime = time.monotonic() - self._start
            print(
                f"[debug] heartbeat: still alive, uptime={uptime:.0f}s",
                file=sys.stderr,
                flush=True,
            )


def _build_transcribers(
    profiles: list[Profile],
    transcriber_overrides: dict[str, Transcriber] | None = None,
    *,
    use_gpu: bool | None = None,
    gpu_device: int = 0,
    n_threads: int = 0,
    engine: str = "whispercpp",
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

    `use_gpu` / `gpu_device` / `n_threads` are forwarded to the
    `WhisperCppTranscriber` constructor — see
    `speakinput.transcriber` for the auto-detect behavior.

    `engine` selects the STT engine. See `create_transcriber` in
    `speakinput.transcriber` for the engine registry and fallback
    behavior.
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
            if engine == "whispercpp":
                # Fast path: use the class directly so tests that
                # monkeypatch WhisperCppTranscriber still work.
                t = WhisperCppTranscriber(
                    model=model_path,
                    language=profile.language,
                    beam_size=profile.beam_size,
                    initial_prompt=profile.initial_prompt,
                    use_gpu=use_gpu,
                    gpu_device=gpu_device,
                    n_threads=n_threads,
                )
            else:
                t = create_transcriber(
                    engine=engine,
                    model=model_path,
                    language=profile.language,
                    beam_size=profile.beam_size,
                    initial_prompt=profile.initial_prompt,
                    use_gpu=use_gpu,
                    gpu_device=gpu_device,
                    n_threads=n_threads,
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
            mic_failover_scan=config.recovery.mic_failover_scan,
            mic_open_retries=config.recovery.mic_open_retries,
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
        self.injector = injector or select_injector_with_fallback(
            config.inject,
            enable_fallback=config.recovery.injector_fallback,
        )
        self.media_controller = (
            MediaController()
            if config.audio.pause_media
            else None
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
        # Serializes the *injection* of typed text. The body lock above
        # only covers the brief drain→re-arm window; transcribe+inject
        # runs OUTSIDE it. Two threads (a chunked body that just
        # finished its drain and the manual-release finalizer that
        # fired in parallel) can both be inside `injector.inject`
        # simultaneously, which on the Unicode/clipboard path means
        # two concurrent paste-and-restore sequences whose Ctrl-V and
        # restore interleave. The lock makes the second inject wait
        # for the first to finish; injects are seconds, not minutes,
        # so the worst-case wait is bounded by the time the first
        # pbcopy + pynput.pressed(Ctrl) + pynput.tap(V) takes.
        # Acquired outside the injector so we can never deadlock with
        # the injector's own internal unicode-path lock.
        self._inject_lock = threading.Lock()
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
        # Background liveness watcher and heartbeat — both started in
        # `run()` once the listeners are actually live. None until then.
        self._liveness_watcher: _LivenessWatcher | None = None
        self._heartbeat: _Heartbeat | None = None
        # Which hotkey backend `run()` picked. Remembered so a dead or
        # sleep-disabled listener can be recreated with the same backend
        # (`_restart_listener`). Defaults to pynput.
        self._use_evdev = False
        # Per-key monotonic timestamp of the last listener restart. If a
        # restarted listener dies again almost immediately (e.g. Input
        # Monitoring / Accessibility permission was revoked — the tap
        # can NEVER come up), restarting on every death would flap
        # forever: die → restart → die 5s later → restart → ... The
        # backoff makes the second death inside the window warn instead
        # of restart, which is the signal the user actually needs.
        self._listener_restart_at: dict[str, float] = {}
        # Configurable backoff for listener restart (from [recovery]).
        self._listener_restart_min_interval = (
            config.recovery.listener_restart_min_interval_s
        )
        # Single-consumer FIFO that serializes hotkey press/release
        # bodies on a dedicated worker thread. The pynput CGEventTap
        # callback only enqueues and returns immediately: macOS disables
        # event taps whose callback runs too long, and a release body
        # here can take SECONDS (drain → close stream → whisper
        # transcribe → inject). Running that on the tap thread got the
        # tap killed mid-session, which presented as "the hotkey
        # randomly stops working until restart". The queue is created
        # eagerly so callbacks can enqueue before `run()` starts the
        # worker (it starts it before any listener can fire).
        self._work_q: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        # Layered continuity hints fed into whisper's initial_prompt.
        #
        # `_last_clip_text` + `_last_clip_at`: the text + monotonic
        # timestamp of the most recent successful transcription. Lives
        # across presses. Updated at the END of every non-empty
        # transcribe. Used as a "picking up where I left off" hint for
        # the NEXT press when the gap is under
        # `audio.prev_clip_window_seconds`.
        #
        # `_press_start_clip_text`: a snapshot of `_last_clip_text`
        # taken at the start of each press. This is what the current
        # press's transcribes see as their "across-press" hint — it's
        # frozen at press-start so updating `_last_clip_text` mid-press
        # (e.g. after the first chunk) doesn't change the across-press
        # hint for the second chunk of the SAME press. The semantic is:
        # the across-press hint is "the last thing I said BEFORE I
        # pressed the key this time", not "the last thing I said
        # 200ms ago in this very press".
        #
        # `_last_chunk_text`: the text of the previous auto-stopped
        # chunk WITHIN the current press. Reset on press, updated on
        # each chunk. Used as a within-press continuity hint that is
        # always passed (not gated by the across-press window). The
        # assumption is: if you paused mid-sentence to let a chunk
        # type out, the next sentence is part of the same thought.
        #
        # All three fields are guarded by `_prompt_lock` because the
        # chunked watchdog body and the final release body can touch
        # them from different threads.
        self._last_clip_text: str = ""
        self._last_clip_at: float = 0.0
        self._press_start_clip_text: str = ""
        self._last_chunk_text: str = ""
        self._prompt_lock = threading.Lock()
        # Dead-mic detector. Counts consecutive *held* recordings whose
        # RMS is below `audio.silence_threshold` (i.e. the silence gate
        # fired). When the count reaches `audio.dead_mic_consecutive`,
        # the user gets a prominent "microphone detected but no audio"
        # signal — the classic dead-mic symptom where the device streams
        # but the analog stage is silent/muted. Any loud recording
        # resets the count. `_dead_mic_warned` makes sure we warn once
        # per consecutive-run, not on every silent press.
        self._consecutive_silent = 0
        self._dead_mic_warned = False

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
        # Reset the within-press chunk text and snapshot the across-press
        # hint. The snapshot is what this press's chunks will see as
        # their "previous press" hint — freezing it at press-start means
        # a chunk's own transcribe doesn't immediately become its own
        # across-press prompt for the NEXT chunk of the same press.
        with self._prompt_lock:
            self._last_chunk_text = ""
            self._press_start_clip_text = self._last_clip_text
        _dbg(self.debug, f"key press start ({profile.key})")
        try:
            self.recorder.start()
        except AudioError as exc:
            # The recorder already printed a clear stderr message
            # explaining what went wrong (mic gone, permission denied,
            # bad sample rate, CoreAudio rejecting the stream, etc.).
            # AudioError is a *user-facing* condition — almost always a
            # fixable environment problem, not a bug — so log at
            # WARNING without a stack trace. A full traceback here
            # (the old `log.exception` behavior) made users think the
            # app had crashed when really their mic just wasn't
            # available. Switch the menu-bar / stderr feedback to the
            # 'error' state so the user gets an unmistakable signal
            # that the press was registered but capture failed. The
            # release path will see `is_recording() == False` and
            # become a no-op, so we don't have to do anything else
            # here. Busy lock is released so the next press is
            # accepted normally.
            log.warning("could not start recording: %s", exc)
            try:
                self.feedback.set_state("error")
            except Exception:
                pass
            self._busy.release()
            self._press_started_at = None
            self._active_profile = None
            return
        except Exception:
            # Anything else (a genuine bug, not an AudioError) is
            # worth a stack trace — it's a developer-visible failure,
            # not a user-fixable environment problem. Still recover
            # gracefully so the app keeps running and the user can
            # try again.
            log.exception("failed to start recorder")
            try:
                self.feedback.set_state("error")
            except Exception:
                pass
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
        if self.media_controller is not None:
            if self.media_controller.pause():
                _dbg(self.debug, "paused media playback")
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
            # Press callback failed (mic missing, permission denied, etc.)
            # and the error handler already showed 'error' feedback.
            # Reset to 'idle' on release so the menu-bar icon doesn't
            # get stuck on the error glyph after the user lets go.
            # The busy lock was released in on_hotkey_press's error
            # branch, so nothing else needs cleaning up here.
            try:
                self.feedback.set_state("idle")
            except Exception:
                pass
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
        if self.media_controller is not None:
            self.media_controller.resume()
            _dbg(self.debug, "resumed media playback")
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
        # Keep the dead-mic warning visible after the release so the
        # user notices it. A recording that passes the speech gate
        # later clears it (the `_dead_mic_warned` flag is reset in
        # `_process_and_inject` once audio is accepted for transcribe).
        if self._dead_mic_warned:
            try:
                self.feedback.set_state("mic_warning")
            except Exception:
                pass
        else:
            try:
                self.feedback.set_state("idle")
            except Exception:
                pass

    def _process_and_inject(self, audio: np.ndarray, profile: Profile | None) -> None:
        """Trim, gate, transcribe, and inject a chunk of audio.

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
        # Two-stage gate, both keyed on `silence_threshold`.
        #
        # Stage 1, absolute floor: a buffer quieter than the floor means
        # no signal reached the mic at all. This is also the dead-mic
        # signal, so it feeds the consecutive-silent counter.
        if threshold > 0 and rms < threshold:
            _dbg(
                self.debug,
                f"silence gate: rms={rms:.4f} < {threshold:.4f}, skipping transcribe",
            )
            self._note_silent_recording()
            return
        # Stage 2, speech activity: the buffer has energy, but is it
        # speech? Room tone, a fan, or a single keyboard click clears the
        # absolute floor and then whisper invents text from it. A
        # per-frame comparison against the buffer's own noise floor
        # rejects those. This is NOT a dead-mic symptom — the mic is
        # delivering audio, it just isn't the user talking — so it does
        # not touch the consecutive-silent counter.
        if threshold > 0:
            stats = analyse_speech(
                audio,
                self.config.audio.sample_rate,
                threshold,
                noise_margin=self.config.audio.vad_noise_margin,
                min_speech_ms=self.config.audio.vad_min_speech_ms,
            )
            _dbg(
                self.debug,
                f"speech gate: noise_floor={stats.noise_floor:.4f}, "
                f"speech={stats.speech_ms:.0f}ms, "
                f"min={self.config.audio.vad_min_speech_ms:.0f}ms",
            )
            if not stats.has_speech:
                _dbg(
                    self.debug,
                    f"speech gate: {stats.speech_ms:.0f}ms of speech above "
                    f"the noise floor "
                    f"(< {self.config.audio.vad_min_speech_ms:.0f}ms min), "
                    f"skipping transcribe",
                )
                return
        # The audio passed the gate — the mic is delivering a signal.
        # Reset the dead-mic counter so the warning only fires on a
        # genuine sustained dead-mic run, not one bad press.
        self._consecutive_silent = 0
        self._dead_mic_warned = False
        try:
            transcriber = self.transcribers[profile.key]
        except KeyError:
            log.exception("no transcriber wired for key %r", profile.key)
            return
        # Build the layered initial_prompt. Three sources, in priority
        # order — whisper treats the prompt as a single bias blob, but
        # ordering matters: vocabulary names (the configured prompt)
        # set the lexical stage, the across-press topic sets the
        # subject, and the within-press chunk is the most-recent
        # sentence and gives the decoder direct continuity.
        prompt = self._build_initial_prompt(profile)
        if self.debug and prompt:
            _dbg(
                self.debug,
                f"initial_prompt: {prompt!r} (len={len(prompt)} chars)",
            )
        try:
            text = self._transcribe_with_recovery(
                transcriber, audio, profile, prompt
            )
        except Exception:
            log.exception("transcription failed after recovery attempts")
            return
        zh_conversion = profile.zh_conversion if profile else "traditional"
        if text and zh_conversion != "off" and _contains_chinese(text):
            original = text
            text = _convert_text(text, zh_conversion)
            if self.debug and text != original:
                print(
                    f"[debug] zh_conversion: {original!r} -> {text!r}",
                    file=sys.stderr,
                    flush=True,
                )
        # Always print the transcript in debug mode, even when empty, so the
        # user can tell the difference between "silence" and "stuck".
        if self.debug:
            print(f"[debug] transcript: {text!r}", file=sys.stderr, flush=True)
        if text:
            self._record_transcript(text)
            if self.dry_run:
                print(text, file=sys.stderr, flush=True)
            else:
                try:
                    # Serialize inject calls across the chunked body
                    # and the finalizer. Without this, a watchdog
                    # fire that races a manual release (or the
                    # abort→resume shutdown path on top of a finalizing
                    # inject) can have two threads inside
                    # `injector.inject(text)` at the same time. For
                    # the Unicode/clipboard path, that means two
                    # concurrent `_pbcopy` + Ctrl-V sequences whose
                    # paste and restore interleave — the user sees
                    # half of one sentence, then half of the other.
                    # For the ASCII path on wtype/ydotool, two
                    # concurrent `wtype -- foo` and `wtype -- bar`
                    # shell-outs interleave keystrokes. The lock
                    # makes the second inject wait for the first to
                    # finish; cheap because injects are seconds, not
                    # minutes. Acquired OUTSIDE the injector's own
                    # internal lock so we never deadlock with the
                    # injector's clip-restore scheduling path.
                    #
                    # The inject itself is bounded by
                    # `recovery.inject_timeout_s`. The subprocess-based
                    # backends (wtype, ydotool, pbcopy) are already
                    # bounded by _SUBPROCESS_TIMEOUT_S inside the
                    # injector; this bound covers the pynput path
                    # (TypingInjector), which has no internal timeout
                    # and can hang indefinitely when the HID event
                    # system wedges (observed after sleep/wake, during
                    # focus churn). A hang here holds the event worker
                    # and the busy lock forever — the bounded call
                    # turns it into a recoverable failure.
                    with self._inject_lock:
                        ok, _ = _bounded_call(
                            self.injector.inject,
                            text,
                            timeout_s=self.config.recovery.inject_timeout_s,
                            name="inject",
                        )
                        if not ok:
                            log.warning(
                                "inject exceeded %.0fs timeout; "
                                "text was not typed. The HID/clipboard "
                                "subsystem may be wedged.",
                                self.config.recovery.inject_timeout_s,
                            )
                except Exception:
                    log.exception("injection failed")

    def _note_silent_recording(self) -> None:
        """Account for a near-silent recording and warn if the mic looks dead.

        Called every time the silence gate fires (a held recording came
        back below `silence_threshold`). Increments the consecutive-silent
        counter; when it reaches `audio.dead_mic_consecutive`, emit a loud
        "microphone detected but no audio" warning once and switch the
        menu-bar/feedback state to 'mic_warning'. The warning is a manual
        signal, not a crash — the user can keep talking to see if it
        clears (a loud recording resets the counter) or act on the
        README's restart/driver guidance.
        """
        consecutive = self.config.audio.dead_mic_consecutive
        if consecutive <= 0:
            return
        self._consecutive_silent += 1
        if self._consecutive_silent < consecutive or self._dead_mic_warned:
            return
        self._dead_mic_warned = True
        print(
            f"[warn] microphone detected but no audio: "
            f"{consecutive} consecutive silent recordings. The mic device "
            f"streams but no signal is reaching it — it may be muted, "
            f"switched off, or the audio driver is wedged. See README → "
            f"Troubleshooting → 'Microphone produces no sound even though "
            f"it's detected'.",
            file=sys.stderr,
            flush=True,
        )
        try:
            self.feedback.set_state("mic_warning")
        except Exception:
            pass

    # Whisper's initial_prompt has a 224-token cap (the n_text_ctx of
    # tiny/base/small). Long prompts either get truncated or confuse the
    # decoder for unrelated speech. We cap each component independently
    # so a single very long previous clip doesn't crowd out the others.
    _PROMPT_COMPONENT_CHARS = 200
    # Hard cap on the assembled prompt — well under whisper's 224-token
    # limit even for BPE-heavy text. If everything is present and the
    # sum exceeds this, we drop the most expendable component
    # (within-press chunk) first, then the across-press hint, before
    # touching the static lexical bias.
    _PROMPT_TOTAL_CHARS = 400
    # The components are joined with this separator. The total length
    # budget must account for `len(_PROMPT_SEP) * (n - 1)` extra chars
    # so a 3-component join can't sneak past the cap.
    _PROMPT_SEP = ", "
    _PROMPT_SEP_LEN = len(_PROMPT_SEP)

    def _build_initial_prompt(self, profile: Profile) -> str:
        """Assemble the layered `initial_prompt` for the next transcribe.

        Order: configured lexical bias → across-press hint (if recent) →
        within-press chunk (always, when present). Empty components are
        dropped. The result is hard-capped at `_PROMPT_TOTAL_CHARS` to
        stay well under whisper's 224-token prompt limit; components are
        dropped from least-important to most-important to fit.

        Thread-safety: reads the cross-press and within-press state
        under `_prompt_lock`. The watchdog body (different thread) and
        the release path can both call this concurrently.
        """
        static = (profile.initial_prompt or "").strip()
        with self._prompt_lock:
            prev_clip = self._press_start_clip_text
            prev_clip_age = (
                time.monotonic() - self._last_clip_at if self._last_clip_at else float("inf")
            )
            prev_chunk = self._last_chunk_text
        # Drop the across-press hint if it's missing or too old. The
        # `prev_clip_age` here is the age of the press-start snapshot,
        # which is the same as the age of `_last_clip_text` at the
        # moment the press started (the snapshot is taken then).
        if (
            not prev_clip
            or prev_clip_age > self.config.audio.prev_clip_window_seconds
            or self.config.audio.prev_clip_window_seconds <= 0
        ):
            prev_clip = ""
        components = [c for c in (static, prev_clip, prev_chunk) if c]
        if not components:
            return ""
        # Each component is independently capped; the assembled string
        # is then capped to `_PROMPT_TOTAL_CHARS` by dropping the
        # least-important component first.
        capped = [c[-self._PROMPT_COMPONENT_CHARS:] for c in components]
        sep_budget = self._PROMPT_SEP_LEN * (len(capped) - 1)
        while (
            sum(len(c) for c in capped) + sep_budget > self._PROMPT_TOTAL_CHARS
            and len(capped) > 1
        ):
            # Drop the within-press chunk first (least important for
            # topic-level bias), then the across-press hint.
            capped.pop(-1)
            sep_budget = self._PROMPT_SEP_LEN * (len(capped) - 1)
        return self._PROMPT_SEP.join(capped)

    def _transcribe_with_recovery(
        self,
        transcriber: Transcriber,
        audio: np.ndarray,
        profile: Profile,
        prompt: str,
    ) -> str:
        """Transcribe audio with automatic crash + hang recovery.

        Two recovery dimensions:

        1. **Crash** — if transcribe() raises, reload the model and
           retry up to `recovery.transcribe_retries` times with backoff.
           Handles a whisper model in a bad state (GPU memory
           corruption, corrupted model file, native crash). Reloading
           clears the state and the retry usually succeeds.

        2. **Hang** — if transcribe() takes longer than
           `recovery.transcribe_timeout_s`, abandon it. The native
           call is wedged (GPU hang, CUDA OOM recovery, Vulkan
           device-lost, Metal command buffer stuck) and will never
           return. Without the timeout, a hang here holds the event
           worker and the busy lock forever — every later press is
           silently ignored as "already busy" and the only way out is
           a manual restart. The timeout turns the hang into a
           recoverable failure: we treat it like a crash (reload +
           retry) so the next attempt gets a fresh model instance.

        The user never sees a "transcription failed" error unless ALL
        retries are exhausted. The audio chunk is preserved across
        retries (it's just a numpy array, not tied to the model).
        """
        retries = self.config.recovery.transcribe_retries
        backoff = self.config.recovery.transcribe_backoff_s
        timeout = self.config.recovery.transcribe_timeout_s
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                ok, text = _bounded_call(
                    transcriber.transcribe,
                    audio,
                    self.config.audio.sample_rate,
                    initial_prompt=prompt or None,
                    timeout_s=timeout,
                    name="transcribe",
                )
                if ok:
                    return text  # type: ignore[return-value]
                # Timed out — treat like a crash: reload + retry.
                raise TimeoutError(
                    f"transcribe exceeded {timeout:.0f}s timeout"
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    break
                log.warning(
                    "transcribe attempt %d/%d failed: %s; reloading model",
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                if backoff > 0:
                    time.sleep(backoff)
                new_t = self._reload_transcriber(profile)
                if new_t is not None:
                    transcriber = new_t
                else:
                    break
        if last_exc is not None:
            raise last_exc
        return ""

    def _reload_transcriber(self, profile: Profile) -> Transcriber | None:
        """Reload the transcriber for `profile` after a crash or hang.

        Reconstructs the transcriber from scratch (re-downloads the
        model if it was deleted, re-initializes the engine). On
        success, updates the `self.transcribers` dict so subsequent
        presses use the fresh instance. On failure, returns None and
        the caller falls back to the old (possibly wedged) instance.

        The reload itself is bounded by the transcribe timeout: a
        GPU init hang (driver wedged, CUDA context lost) would
        otherwise hold the event worker and the busy lock forever,
        defeating the recovery the reload is supposed to provide.
        The model download is NOT bounded (it can legitimately take
        minutes on a slow connection) — but the download only happens
        once, and a hang there is a network issue the user can see.
        """
        try:
            model_name, _ = resolve_for_language(profile.model, profile.language)
            model_path = ensure_model(model_name)
            timeout = self.config.recovery.transcribe_timeout_s
            ok, new_t = _bounded_call(
                create_transcriber,
                engine=self.config.transcribe.engine,
                model=model_path,
                language=profile.language,
                beam_size=profile.beam_size,
                initial_prompt=profile.initial_prompt,
                use_gpu=self.config.transcribe.use_gpu,
                gpu_device=self.config.transcribe.gpu_device,
                n_threads=self.config.transcribe.n_threads,
                timeout_s=timeout,
                name="transcriber-reload",
            )
            if not ok:
                log.warning(
                    "transcriber reload for %r exceeded %.0fs timeout",
                    profile.key,
                    timeout,
                )
                return None
            self.transcribers[profile.key] = new_t  # type: ignore[assignment]
            print(
                f"[info] transcriber for {profile.key!r} reloaded successfully",
                file=sys.stderr,
                flush=True,
            )
            return new_t  # type: ignore[return-value]
        except Exception:
            log.exception("transcriber reload failed for %r", profile.key)
            return None

    def _record_transcript(self, text: str) -> None:
        """Update continuity state after a successful non-empty transcribe.

        Sets both the within-press chunk text (used by the next chunk
        in the same press) and the across-press clip text + timestamp
        (used by the next press). Guarded by `_prompt_lock` because
        the chunked watchdog body and the release path can land on
        the same `_last_clip_text` update.
        """
        with self._prompt_lock:
            self._last_chunk_text = text
            self._last_clip_text = text
            self._last_clip_at = time.monotonic()

    # --- event worker: keep heavy work off the hotkey callback thread ------

    def _start_event_worker(self) -> None:
        """Start the thread that runs press/release bodies. Idempotent."""
        if self._worker_thread is not None:
            return
        self._worker_thread = threading.Thread(
            target=self._event_worker_main, name="speakinput-events", daemon=True
        )
        self._worker_thread.start()

    def _stop_event_worker(self, timeout_s: float = 2.0) -> None:
        """Ask the worker to drain and exit; wait at most `timeout_s`.

        The sentinel is queued behind any in-flight job, so a transcribe
        that is already running gets to finish (bounded by `timeout_s`).
        The thread is a daemon, so a job that outlives the timeout can't
        block process exit.
        """
        self._work_q.put(None)
        if self._worker_thread is not None:
            try:
                self._worker_thread.join(timeout=timeout_s)
            except RuntimeError:
                # join() on a thread that was never started.
                pass
        self._worker_thread = None

    def _enqueue_event(self, fn, *args) -> None:
        """Run `fn(*args)` on the event worker, in FIFO order.

        Called from the hotkey listener's callback thread. Keeps the
        callback O(µs) so macOS never disables the event tap for being
        slow, and serializes press/release bodies so the lock discipline
        matches the old everything-on-the-listener-thread behavior.
        """
        self._work_q.put((fn, args))

    def _event_worker_main(self) -> None:
        while True:
            item = self._work_q.get()
            if item is None:
                return
            fn, args = item
            try:
                fn(*args)
            except Exception:
                # The worker MUST NOT die: a dead worker with live
                # listeners is the same "hotkey silently does nothing"
                # failure this thread exists to prevent.
                log.exception(
                    "hotkey event handler failed: %s",
                    getattr(fn, "__name__", repr(fn)),
                )

    def _restart_event_worker(self) -> bool:
        """Recreate the event worker thread after it died.

        The liveness watcher detects a dead worker (via
        `_check_worker`) and calls `_on_worker_dead`, which enqueues
        this restart. The old thread reference is replaced; the
        queue is NOT drained (any enqueued press/release events are
        lost, but the lock state is preserved — a dead worker mid-
        press leaves the busy lock held, which the busy-lock
        watchdog will abort). Returns True on success.
        """
        if self._shutdown.is_set():
            return False
        self._worker_thread = None
        try:
            self._worker_thread = threading.Thread(
                target=self._event_worker_main,
                name="speakinput-events",
                daemon=True,
            )
            self._worker_thread.start()
        except Exception:
            log.exception("failed to restart event worker")
            return False
        if self._liveness_watcher is not None:
            self._liveness_watcher.swap_worker(self._worker_thread)
        print(
            "[info] event worker died and was restarted",
            file=sys.stderr,
            flush=True,
        )
        return True

    def _on_worker_dead(self) -> None:
        """Liveness callback: the event worker thread died.

        The worker's try/except in `_event_worker_main` catches
        exceptions from individual handlers, so a dead worker means
        something killed the thread itself (a native crash inside
        whisper/pynput that escaped the Python-level catch, or an
        uncaught BaseException like SystemExit). Restart it; if the
        restart fails, warn the user — the hotkey is dead until they
        restart speakinput.
        """
        if self._shutdown.is_set():
            return
        restarted = self._restart_event_worker()
        if not restarted:
            print(
                "[warn] event worker is no longer alive and could not be "
                "restarted — the hotkey will not respond. Restart "
                "speakinput to recover.",
                file=sys.stderr,
                flush=True,
            )

    def _on_busy_stuck(self, held_for: float) -> None:
        """Liveness callback: the busy lock has been held too long.

        The universal backstop for ANY hang in the processing pipeline
        — a transcribe hang (GPU wedge), an inject hang (HID wedge), a
        model reload hang, or any future code path that holds the lock
        too long. Fires `_abort_press` on the event worker (or inline
        if the worker is dead) to release the lock, close the
        recorder, and reset the press state. The buffered audio is
        discarded — it may contain arbitrarily long ambient audio, and
        transcribing it would stall the worker for minutes.

        Enqueued on the event worker so it's serialized with any
        in-flight handler (the abort's body-lock acquire waits for a
        chunked body to finish its drain). If the worker is dead, run
        the abort inline — the lock is already held by the dead
        worker's orphaned handler, and _abort_press's body-lock
        acquire will time out gracefully.
        """
        if self._shutdown.is_set():
            return
        print(
            f"[warn] busy lock held for {held_for:.0f}s — aborting stuck "
            f"press to recover. This usually means a transcribe or inject "
            f"call hung (GPU/HID wedge).",
            file=sys.stderr,
            flush=True,
        )
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._enqueue_event(self._abort_press, "busy lock watchdog timeout")
        else:
            # Worker is dead — run the abort inline. The body-lock
            # acquire may fail (the dead worker's handler is still
            # holding it), but _abort_press handles that gracefully.
            self._abort_press("busy lock watchdog timeout (worker dead)")

    def _abort_press(self, reason: str = "") -> None:
        """Cancel an in-flight press whose release event will never arrive.

        Runs on the event worker (serialized with press/release
        callbacks). Triggered when the hotkey listener died or the
        machine slept while a key was held: the release is gone for
        good, so without this the busy lock would be held forever, the
        recorder would keep buffering audio (~230 MB/hour), and every
        later press would be ignored as "already busy".

        The buffered audio is DISCARDED, not transcribed: it can
        contain arbitrarily long ambient audio (the recorder ran the
        whole time the hotkey was dead), and transcribing it would
        stall the worker for minutes.

        No-op when no press is active, when a release is already being
        finalized (the finalize path owns the lock then), or during
        shutdown.
        """
        if self._shutdown.is_set():
            return
        if not self._busy.locked() or self._manual_release_pending:
            return
        print(
            f"[warn] abandoned a stuck push-to-talk press"
            f"{f' ({reason})' if reason else ''}; "
            f"the buffered audio was discarded",
            file=sys.stderr,
            flush=True,
        )
        if self._watchdog is not None:
            try:
                self._watchdog.stop()
            except Exception:
                pass
            self._watchdog = None
        # Acquire the body lock around the drain so a chunked
        # watchdog body (which is the only other thread that touches
        # the recorder's chunks list besides the worker) can't
        # simultaneously call drain() and have us both end up
        # returning a half-drained buffer. `_process_and_inject` runs
        # OUTSIDE the body lock; the abort's `_process_and_inject`
        # is not called (we're discarding the audio) so releasing
        # the lock right after drain is fine. The close() inside
        # recorder uses the recorder's own stream lock and is safe
        # regardless.
        try:
            with self._body_lock:
                try:
                    self.recorder.drain()
                except Exception:
                    pass
                try:
                    self.recorder.close()
                except Exception:
                    pass
        except Exception:
            # Lock acquire shouldn't fail, but if it does (a chunk
            # body is wedged), don't let it freeze shutdown — log
            # and continue.
            log.exception("body lock acquire failed during press abort")
        if self.media_controller is not None:
            try:
                self.media_controller.resume()
            except Exception:
                log.exception("media resume failed during press abort")
        self._press_started_at = None
        self._active_profile = None
        self._manual_release_pending = False
        try:
            self.feedback.set_state("idle")
        except Exception:
            pass
        try:
            self._busy.release()
        except RuntimeError:
            # Lost a race with a concurrent finalize that already
            # released the lock — nothing left to do.
            pass

    # --- listener restart: recover from dead threads and sleep-disabled taps ---

    def _restart_listener(self, key: str) -> bool:
        """Recreate the hotkey listener for `key`. Returns True on success.

        Used when a listener's run loop died (crash, tap invalidation)
        and after system sleep (macOS disables CGEventTaps across
        sleep/wake without killing the thread, so the liveness check
        alone can't see it). On failure the old listener stays in the
        registry so the liveness watcher's next tick retries.
        """
        profile = next((p for p in self._profiles if p.key == key), None)
        old = self.listeners.get(key)
        if profile is None or old is None:
            return False
        try:
            if self._use_evdev:
                new_listener = EvdevHotkeyListener(
                    keycode=resolve_evdev_key(key),
                    on_press=self._make_press_cb(profile),
                    on_release=self._make_release_cb(profile),
                )
            else:
                new_listener = HotkeyListener(
                    key=resolve_key(key),
                    on_press=self._make_press_cb(profile),
                    on_release=self._make_release_cb(profile),
                )
            new_listener.start()
        except Exception:
            log.exception("failed to restart hotkey listener for %r", key)
            return False
        try:
            old.stop()
        except Exception:
            pass
        self.listeners[key] = new_listener
        self._listener_restart_at[key] = time.monotonic()
        if self._liveness_watcher is not None:
            self._liveness_watcher.swap(old, new_listener)
        return True

    # A restarted listener that dies again within this many seconds is
    # considered unrecoverable by restarting (permission revoked, HID
    # subsystem wedged, ...) — warn the user instead of flapping.
    # Configurable via [recovery].listener_restart_min_interval_s so
    # users who want always-restart can set it to 0.
    _LISTENER_RESTART_MIN_INTERVAL_S = 60.0  # replaced by config at init

    def _on_listener_dead(self, key: str) -> None:
        """Liveness callback: a hotkey listener's run loop died.

        Tries to restart the listener in place. Only if the restart
        fails (or the listener is flapping — died again right after a
        restart) does the user get the "restart speakinput" warning — a
        successful restart is a self-healing event worth one info line.
        Either way, any press that was active when the listener died is
        aborted: its release event died with the listener.
        """
        if self._shutdown.is_set():
            return
        last_restart = self._listener_restart_at.get(key, 0.0)
        flapping = (time.monotonic() - last_restart) < self._listener_restart_min_interval
        restarted = False
        if not flapping:
            restarted = self._restart_listener(key)
        if restarted:
            print(
                f"[info] hotkey listener for {key!r} died and was restarted",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[warn] hotkey listener for {key!r} is no longer alive and "
                f"could not be restarted — the push-to-talk key will not "
                f"respond. Restart speakinput to recover.",
                file=sys.stderr,
                flush=True,
            )
        self._enqueue_event(self._abort_press, f"hotkey listener for {key!r} died")

    def _on_system_sleep(self, slept_s: float) -> None:
        """Sleep callback: the wall clock jumped while monotonic froze.

        macOS disables CGEventTaps across sleep/wake; the listener
        threads survive and pass the liveness check, but no key event
        is ever delivered again. We do NOT restart the listeners here
        because the old listener's cleanup (`CGEventTapEnable(tap,
        False)` in `_mac_tap_heal.py`'s finally block) races with the
        new listener's tap enable — the old thread may not have exited
        yet when the new listener opens its tap, and the old finally
        disables it. Instead, we rely on the self-healing mechanism
        (`_mac_tap_heal.py`) to re-enable the tap within ~1s. If it
        fails, the heartbeat backstop (30s stale threshold) will
        detect the dead listener and restart it.

        We DO close the audio recorder so the next start() opens a
        fresh stream against the re-initialized CoreAudio HAL.
        """
        if self._shutdown.is_set():
            return
        print(
            f"[warn] system slept for ~{slept_s:.0f}s — "
            f"audio recorder closed, hotkey tap will self-heal",
            file=sys.stderr,
            flush=True,
        )
        # Proactively close the recorder so the next start() opens a
        # completely fresh stream. The old stream handle is dead after
        # wake (CoreAudio HAL reinitializes and drops the callback
        # link). Leaving it open means is_recording() stays True on a
        # silent stream, confusing the watchdog and the release path.
        try:
            self.recorder.close()
        except Exception:
            log.exception("failed to close recorder after sleep")
        self._enqueue_event(self._abort_press, "system wake")

    def _resume_media_bounded(self, timeout_s: float) -> None:
        """Resume paused media without letting a wedged backend block shutdown.

        On macOS the media backend shells out to osascript, which talks
        to System Events / Spotify over AppleEvents. After sleep/wake
        those services can wedge, and even `subprocess.run(timeout=...)`
        can then block waiting on an uninterruptible child. Run the
        resume on a daemon thread and bound the wait: if it doesn't
        finish in `timeout_s`, log and move on. The daemon dies with
        the process.
        """
        def _do_resume() -> None:
            try:
                self.media_controller.resume()
            except Exception:
                log.exception("media resume failed during shutdown")

        t = threading.Thread(
            target=_do_resume, name="speakinput-media-resume", daemon=True
        )
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            print(
                f"[warn] media resume did not finish in {timeout_s:.0f}s — "
                f"continuing shutdown without it",
                file=sys.stderr,
                flush=True,
            )

    def _make_press_cb(self, profile: Profile):
        def cb() -> None:
            self._enqueue_event(self.on_hotkey_press, profile)
        return cb

    def _make_release_cb(self, profile: Profile):
        def cb() -> None:
            self._enqueue_event(self.on_hotkey_release, profile)
        return cb

    def run(self) -> None:
        # Bootstrap: ensure models are on disk BEFORE we start the hotkey
        # listener, so the user never sees a 141 MB download start mid-session.
        # If the test (or another caller) injected transcribers, skip.
        if not self.transcribers:
            try:
                self.transcribers = _build_transcribers(
                    self._profiles,
                    use_gpu=self.config.transcribe.use_gpu,
                    gpu_device=self.config.transcribe.gpu_device,
                    n_threads=self.config.transcribe.n_threads,
                    engine=self.config.transcribe.engine,
                )
            except ModelNotFoundError as exc:
                print(f"model error: {exc}", file=sys.stderr)
                raise SystemExit(2) from exc
            except ModelDownloadError as exc:
                print(f"model error: {exc}", file=sys.stderr)
                raise SystemExit(2) from exc
        self._print_banner()
        self.feedback.start()
        # Backend selection on Linux: prefer evdev (reads /dev/input
        # directly, works on Wayland AND X11 AND headless), fall back to
        # pynput only if evdev can't find a keyboard. The previous
        # `XDG_SESSION_TYPE == "wayland"` gate was too narrow — many
        # real Wayland sessions (tmux, SSH, containers) don't propagate
        # the env var, so speakinput would silently fall through to the
        # X11-only pynput backend and stop detecting key presses.
        use_evdev = False
        evdev_probe_error: str | None = None
        if sys.platform == "linux":
            use_evdev, evdev_probe_error = _probe_evdev_or_diag()
        self._use_evdev = use_evdev
        # Always print the hotkey backend so the user can verify which
        # code path is active — a missing or wrong banner has been a
        # recurring source of "hotkey does nothing" reports that turn
        # out to be the wrong backend being picked silently.
        if use_evdev:
            print(
                "[startup] hotkey   : evdev (Linux — reads /dev/input directly)",
                file=sys.stderr,
                flush=True,
            )
        else:
            platform_tag = (
                "Linux (evdev unavailable — pynput fallback)"
                if sys.platform == "linux"
                else "macOS / Windows (pynput)"
            )
            print(
                f"[startup] hotkey   : pynput ({platform_tag})",
                file=sys.stderr,
                flush=True,
            )
            if evdev_probe_error:
                # Surface the underlying reason so the user can fix
                # permissions / attach a keyboard / etc. without
                # grepping through Python source. The pynput backend
                # on Linux+X11 will silently fail to grab global keys
                # without a reachable X server, so this hint is the
                # difference between "hotkey is broken" and "here's
                # exactly why and what to do".
                print(
                    f"[startup] hotkey   : evdev probe failed: {evdev_probe_error}",
                    file=sys.stderr,
                    flush=True,
                )
        # Start the event worker BEFORE any listener can fire, so every
        # press/release callback lands on the queue instead of running
        # inside the OS event-tap callback (macOS disables slow taps).
        self._start_event_worker()
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
        # We handle SIGINT, SIGTERM, and SIGHUP. SIGINT is Ctrl-C in the
        # foreground terminal; SIGTERM is what launchctl / `kill` send by
        # default; SIGHUP is what the kernel sends when the controlling
        # terminal is closed (e.g. the user closed the Terminal window
        # that started speakinput). Without a SIGHUP handler, the
        # process would die mid-shutdown the next time the user opens
        # the same terminal — leaving the recorder thread orphaned and
        # the next start.sh failing to acquire the single-instance lock.
        #
        # SIGINT escalates: the FIRST Ctrl-C sets the shutdown event and
        # the `finally` block below runs the teardown. If teardown hangs
        # (wedged osascript after sleep, a stuck native call, ...), a
        # plain "set the event again" handler would make every further
        # Ctrl-C a silent no-op — the user then has to `kill -9` with no
        # idea why. So the SECOND Ctrl-C dumps every thread's stack (so
        # the hang is diagnosable from the terminal scrollback) and
        # force-exits via os._exit, which the kernel handles regardless
        # of what any thread is blocked in. SIGTERM/SIGHUP stay
        # single-shot; start.sh already escalates TERM -> KILL itself.
        def _on_sigint(*_) -> None:
            if not self._shutdown.is_set():
                print(
                    "[shutdown] Ctrl-C received — cleaning up "
                    "(press Ctrl-C again to force quit)",
                    file=sys.stderr,
                    flush=True,
                )
                self._shutdown.set()
                return
            print(
                "[shutdown] second Ctrl-C — forcing immediate exit. "
                "Thread stacks follow (include them in any bug report):",
                file=sys.stderr,
                flush=True,
            )
            try:
                import faulthandler

                faulthandler.dump_traceback(file=sys.stderr)
            except Exception:
                pass
            os._exit(2)

        signal.signal(signal.SIGINT, _on_sigint)
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown.set())
        try:
            signal.signal(signal.SIGHUP, lambda *_: self._shutdown.set())
        except (AttributeError, ValueError):
            # SIGHUP doesn't exist on Windows; signal.signal also fails
            # if called from a non-main thread. Either way, skip it.
            pass
        # Background liveness watcher. Polls every 5s whether the
        # listener threads are still alive, and watches for system sleep
        # (wall/monotonic clock skew). pynput's macOS backend uses a
        # CGEventTap that macOS disables on sleep/wake or when the user
        # revokes Input Monitoring — the thread keeps running but stops
        # delivering events, and the user has no way to tell. A dead
        # Thread object (`.is_alive() == False`) means the run loop
        # returned or raised; either way the hotkey is dead. Both cases
        # trigger a listener restart (`_on_listener_dead` /
        # `_on_system_sleep`); only a failed restart warns the user to
        # restart the app. Daemon so it dies with the process; the
        # checks are cheap (one bool per listener, two clock reads).
        #
        # The watcher also monitors the event worker thread (restarts it
        # if it dies) and the busy lock (aborts a press that's been
        # stuck too long — the universal backstop for ANY hang in the
        # processing pipeline). Both are off when the corresponding
        # config field is 0.
        self._liveness_watcher = _LivenessWatcher(
            listeners=list(self.listeners.values()),
            interval_s=5.0,
            on_dead=self._on_listener_dead,
            on_sleep=self._on_system_sleep,
            worker_thread=self._worker_thread,
            on_worker_dead=self._on_worker_dead,
            busy_lock=self._busy,
            press_started_at=lambda: self._press_started_at,
            busy_lock_timeout_s=self.config.recovery.busy_lock_timeout_s,
            on_busy_stuck=self._on_busy_stuck,
        )
        self._liveness_watcher.start()
        # Heartbeat: print a one-line "still here" every 60s in debug
        # mode. Helps the user distinguish a hung process (no heartbeat
        # after a minute) from a healthy one that's just not being
        # spoken to. Off by default because it's noise.
        if self.debug:
            self._heartbeat = _Heartbeat(interval_s=60.0)
            self._heartbeat.start()
        else:
            self._heartbeat = None
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
        prev_window = cfg.audio.prev_clip_window_seconds
        prev_window_str = "off" if prev_window == 0 else f"{prev_window:g}s"
        if threshold == 0:
            speech_str = "off"
        else:
            speech_str = (
                f"{cfg.audio.vad_min_speech_ms:g}ms "
                f"@{cfg.audio.vad_noise_margin:g}x noise"
            )
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
                f"language={p.language} prompt={prompt} zh_conversion={p.zh_conversion}"
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
            f"silence  : rms<{threshold_str} -> skip; speech {speech_str}; "
            f"auto-stop after {auto_stop_str}",
            f"continuity: across-press hint within {prev_window_str} (within-press always on)",
            f"inject   : {inject_mode}, trailing_space={cfg.inject.trailing_space}",
            f"transcribe: {_gpu_summary(cfg.transcribe.use_gpu, cfg.transcribe.gpu_device)}",
        ]
        for line in lines:
            print(f"[startup] {line}", file=sys.stderr, flush=True)

    def shutdown(self) -> None:
        # Stop the background helpers first so they don't try to use
        # the listeners/watchdog while we're tearing them down.
        if self._liveness_watcher is not None:
            try:
                self._liveness_watcher.stop()
            except Exception:
                log.exception("liveness watcher stop failed")
            self._liveness_watcher = None
        if self._heartbeat is not None:
            try:
                self._heartbeat.stop()
            except Exception:
                log.exception("heartbeat stop failed")
            self._heartbeat = None
        self._shutdown.set()
        if self.media_controller is not None:
            # A hung osascript / playerctl must NOT block shutdown. The
            # subprocess timeout normally bounds it, but after sleep/wake
            # the AppleEvent services can wedge hard enough that even the
            # timeout-kill blocks — so the whole resume runs on a helper
            # thread with a bounded join.
            self._resume_media_bounded(timeout_s=3.0)
        if self._watchdog is not None:
            try:
                self._watchdog.stop()
            except Exception:
                log.exception("watchdog stop failed")
            self._watchdog = None
        for key, listener in self.listeners.items():
            try:
                listener.stop()
            except Exception:
                log.exception("listener stop failed (key=%s)", key)
        # Keep the listeners dict around so post-shutdown tests can
        # verify which keys were wired. pynput's Listener objects are
        # safe to leave stopped but referenced.
        #
        # Stop the event worker after the listeners so no new events can
        # be enqueued behind the sentinel. The sentinel lets an
        # in-flight transcribe+inject finish (bounded); the daemon
        # thread dies with the process if it outlives the timeout.
        self._stop_event_worker(timeout_s=2.0)
        # Close the audio stream HERE, on the main thread, before the
        # interpreter starts finalizing. sounddevice registers an atexit
        # handler that stops+closes any still-open stream; if the stream
        # is left open, that handler runs inside _Py_Finalize (GIL held)
        # while another thread may be mid-stop of the same stream —
        # CoreAudio's HAL mutex + the IO thread's wait for the GIL then
        # form a three-way deadlock that even SIGINT can't escape
        # (observed via `sample` on a stuck instance). Closing here is
        # serialized with any in-flight worker close via the recorder's
        # stream lock, so at most one thread is ever inside CoreAudio,
        # and atexit finds nothing left to close.
        try:
            self.recorder.close()
        except Exception:
            log.exception("recorder close failed during shutdown")
        try:
            self.feedback.stop()
        except Exception:
            log.exception("feedback stop failed")
