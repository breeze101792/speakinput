"""Global hotkey listener for push-to-talk.

Two backends are available, picked per-platform by `App.run()`:

* `HotkeyListener` (pynput) — used on macOS and on Linux/Windows when an
  X11 display is reachable. pynput grabs global keys via the XRecord
  extension on X11; on macOS it uses the HIToolbox framework.

* `EvdevHotkeyListener` (evdev) — used on Linux Wayland sessions where
  pynput's X11 backend cannot reach a display. evdev reads the Linux
  kernel input subsystem directly, so it works on any Wayland
  compositor without an XWayland bridge.

Both backends expose the same public surface (start / stop / join /
is_running + on_press / on_release callbacks). The latch flag (`_pressed`)
is shared between them — it guards against key-repeat firing the press
callback multiple times per physical hold, which is what both macOS
HID events and Linux evdev do.
"""

from __future__ import annotations

import threading
from typing import Callable, Protocol

try:
    from pynput import keyboard
except ImportError:  # pragma: no cover - pynput is a hard dep at runtime
    keyboard = None  # type: ignore[assignment]

try:
    import evdev
    from evdev import ecodes as _ecodes
except ImportError:  # evdev is only installed on Linux; absent elsewhere
    evdev = None  # type: ignore[assignment]
    _ecodes = None  # type: ignore[assignment]

from speakinput.config import VALID_HOTKEYS

_KEY_NAMES = {
    "alt_r": lambda k: k.alt_r,
    "ctrl_r": lambda k: k.ctrl_r,
    "cmd_r": lambda k: k.cmd_r,
    "shift_r": lambda k: k.shift_r,
    "caps_lock": lambda k: k.caps_lock,
    "f12": lambda k: k.f12,
}


# Linux kernel keycodes for the same six canonical names. Picked so
# `cmd_r` lands on the right-Super physical key, matching what pynput
# already does on X11 Linux.
_EVDEV_KEY_MAP: dict[str, int] = {
    "alt_r": _ecodes.KEY_RIGHTALT if _ecodes is not None else 100,
    "ctrl_r": _ecodes.KEY_RIGHTCTRL if _ecodes is not None else 97,
    "cmd_r": _ecodes.KEY_RIGHTMETA if _ecodes is not None else 126,
    "shift_r": _ecodes.KEY_RIGHTSHIFT if _ecodes is not None else 54,
    "caps_lock": _ecodes.KEY_CAPSLOCK if _ecodes is not None else 58,
    "f12": _ecodes.KEY_F12 if _ecodes is not None else 88,
}


# Threshold for "this looks like a real keyboard" — a full keyboard
# reports at least ~50 KEY_* codes (letters, modifiers, function row,
# arrows, numpad, etc.). A media-key or consumer-control device reports
# only a handful. 30 is a generous floor that comfortably excludes those
# but accepts a stripped-down laptop internal keyboard.
_KEYBOARD_KEY_COUNT_MIN = 30

# Keycodes that only a real alphanumeric keyboard exposes. A consumer-
# control sibling (media keys, brightness, etc.) carries 100+ KEY_*
# codes — KEY_PLAY, KEY_BRIGHTNESSUP, etc. — but never KEY_A or
# KEY_SPACE. Requiring both is a high-confidence "this is a keyboard"
# signal that survives even when EV_REP isn't reported (Bluetooth
# receivers and some gaming keyboards don't expose EV_REP through
# evdev, even though they're real keyboards).
_KEYBOARD_SANITY_KEYS = (
    _ecodes.KEY_A if _ecodes is not None else 30,
    _ecodes.KEY_SPACE if _ecodes is not None else 57,
)


class HotkeyError(RuntimeError):
    """Raised by the evdev backend when no usable keyboard is found."""


def resolve_key(name: str):
    if name not in VALID_HOTKEYS:
        raise ValueError(f"unknown hotkey {name!r}; expected one of {VALID_HOTKEYS}")
    if keyboard is None:
        raise RuntimeError("pynput is not installed; cannot resolve hotkey")
    return _KEY_NAMES[name](keyboard.Key)


def resolve_evdev_key(name: str) -> int:
    """Return the Linux kernel keycode for one of VALID_HOTKEYS.

    Symmetric to `resolve_key`, but for the evdev backend.
    """
    if name not in VALID_HOTKEYS:
        raise ValueError(f"unknown hotkey {name!r}; expected one of {VALID_HOTKEYS}")
    if evdev is None:
        raise RuntimeError(
            "evdev is not installed; cannot resolve hotkey on this platform"
        )
    return _EVDEV_KEY_MAP[name]


def probe_evdev_available() -> bool:
    """Return True if evdev can find an accessible keyboard.

    Used by `App.run()` to decide between the evdev and pynput backends
    on Linux. We call the same selection logic as `find_keyboard_devices`
    but close the candidate devices immediately so we don't leak fds
    when we're only probing availability. The actual listener will
    re-open the devices on `start()`.
    """
    if evdev is None:
        return False
    try:
        devs = find_keyboard_devices()
    except HotkeyError:
        return False
    for dev in devs:
        try:
            dev.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    return True


def find_keyboard_devices() -> list:
    """Scan /dev/input/event* and return ALL keyboard-shaped candidates.

    Multi-keyboard support: a typical Linux box has more than one real
    keyboard plugged in (e.g. a Bluetooth receiver exposing its own
    keyboard HID interface AND a USB gaming keyboard). The previous
    implementation picked one "best" device and ignored the rest, which
    meant the user typed on the keyboard that *lost* the ranking and the
    hotkey silently never fired. Now we return every device that looks
    like a real alphanumeric keyboard, and `EvdevHotkeyListener` opens
    them all in parallel — the `_pressed` latch dedups events when more
    than one device fires for the same physical key.

    Selection — a device is included if it exposes >=
    `_KEYBOARD_KEY_COUNT_MIN` `EV_KEY` codes AND has the sanity keys
    (`KEY_A` + `KEY_SPACE`). Consumer-control siblings (media keys,
    brightness, etc.) carry 100+ KEY_* codes but never letters or
    space, so the sanity-key check rejects them without ranking
    gymnastics. Virtual devices (`ydotoold`, `uinput`-based relays)
    have no `phys` path — they're excluded by the same sanity-key
    check in practice (a virtual relay only emits the keys its
    trigger maps to, never the full alphabet), but we also keep the
    `phys` ranking for tie-breaking in `find_keyboard_device`
    (single-device legacy path used by the probe and tests).

    Returns a list of `evdev.InputDevice` opened for reading, sorted
    best-first by the same ranking as `find_keyboard_device`. Caller
    is responsible for `close()`ing each (the listener does this in
    `stop`).

    Raises `HotkeyError` if no suitable device is found, with a message
    pointing at the usual culprits (no `/dev/input` access, user not
    in the `input` group, no keyboard attached).
    """
    if evdev is None:
        raise HotkeyError("evdev is not installed")
    candidates: list[tuple[tuple, object]] = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except (OSError, PermissionError):
            # Skip devices we can't open (e.g. another process has them
            # exclusively) and keep scanning. The user will still see a
            # clear error if nothing works.
            continue
        try:
            caps = dev.capabilities()
        except OSError:
            dev.close()
            continue
        key_codes = caps.get(_ecodes.EV_KEY, set())
        if len(key_codes) < _KEYBOARD_KEY_COUNT_MIN:
            dev.close()
            continue
        # Higher score = better candidate. Built as a tuple so Python's
        # tuple comparison does the right thing (sanity keys first, then
        # phys, then EV_REP, then key count).
        has_sanity = all(k in key_codes for k in _KEYBOARD_SANITY_KEYS)
        has_phys = bool(getattr(dev, "phys", ""))
        has_rep = _ecodes.EV_REP in caps
        candidates.append(
            ((has_sanity, has_phys, has_rep, len(key_codes)), dev)
        )
    if not candidates:
        raise HotkeyError(
            "no keyboard device found in /dev/input — check that the user is in "
            "the `input` group, that a keyboard is attached, and that "
            "/dev/input is readable. (A v2 follow-up will add a "
            "[hotkey].device_path config override for multi-keyboard setups.)"
        )
    # Sort best-first so the parallel listener opens the most-likely
    # device first. We keep every sanity-key-matching device — the
    # whole point is to listen on all real keyboards simultaneously.
    candidates.sort(key=lambda x: x[0], reverse=True)
    # Reject any candidate that doesn't have the sanity keys at all —
    # a consumer-control sibling with 158 keys but no KEY_A is NOT a
    # keyboard, and listening on it would just waste an fd. The score
    # already ranks them last, but we drop them entirely because
    # opening them for read_loop spams kernel events we'll never match.
    sane = [dev for score, dev in candidates if score[0]]
    if not sane:
        # Every candidate failed the sanity-key check. Sort by score
        # and take the best one anyway — better to listen on something
        # that might occasionally fire than to give up entirely. This
        # branch is rare (e.g. a very minimal laptop internal keyboard
        # that for some reason doesn't report KEY_A through evdev), but
        # it preserves the "find a keyboard or raise" contract.
        kept = [candidates[0][1]]
        # Close the rest; we only kept one.
        for _, dev in candidates[1:]:
            try:
                dev.close()
            except Exception:
                pass
    else:
        # Among sanity-key-matching devices, drop virtual ones
        # (empty `phys`) IF at least one device has a real `phys` path.
        # Virtual devices like `ydotoold` advertise the full alphabet
        # (they relay all keys), so the sanity-key check alone can't
        # exclude them. The `phys` path is the reliable signal: real
        # USB/Lenovo/Bluetooth keyboards report e.g.
        # ``usb-0000:03:00.0-3.4.4/input0``; ydotoold and uinput-based
        # relays report an empty string. We only enforce this when a
        # real phys path exists somewhere in the pool so that a fully
        # virtual setup (headless CI running tests with only uinput
        # devices) still gets a listener instead of a hard failure.
        has_any_phys = any(
            bool(getattr(d, "phys", "")) for d in sane
        )
        if has_any_phys:
            kept = []
            for d in sane:
                if getattr(d, "phys", ""):
                    kept.append(d)
                else:
                    try:
                        d.close()
                    except Exception:
                        pass
        else:
            kept = sane
        # Close any candidate we didn't keep (the sanity-key failures).
        for score, dev in candidates:
            if not score[0]:
                try:
                    dev.close()
                except Exception:
                    pass
    return kept


def find_keyboard_device():
    """Scan /dev/input/event* and return the best single keyboard.

    Backward-compatible single-device wrapper around
    `find_keyboard_devices`. Used by `probe_evdev_available` and by
    tests that inject a single fake device — `EvdevHotkeyListener`
    itself uses the plural form so it can listen on every real
    keyboard at once (multi-keyboard setups are common on Linux:
    Bluetooth receiver + USB keyboard, laptop internal + external,
    etc.).

    Returns the highest-ranked `evdev.InputDevice`, opened for reading.
    The caller is responsible for `close()`ing it. Raises `HotkeyError`
    if no suitable device is found.
    """
    devs = find_keyboard_devices()
    for d in devs[1:]:
        try:
            d.close()
        except Exception:
            pass
    return devs[0]


class HotkeyListenerProtocol(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def join(self) -> None: ...
    def is_running(self) -> bool: ...


class HotkeyListener:
    """Detects press and release of a single configurable key.

    Uses a manual pynput keyboard.Listener (not GlobalHotKeys) so we can
    distinguish press from release. A latch flag guards against macOS
    key-repeat events firing `on_press` multiple times per physical press.
    """

    def __init__(
        self,
        key,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        suppress: bool = False,
    ) -> None:
        if keyboard is None:
            raise RuntimeError("pynput is not installed")
        self._key = key
        self._on_press = on_press
        self._on_release = on_release
        self._suppress = suppress
        self._pressed = False
        self._listener = None  # type: ignore[assignment]

    def _matches(self, key) -> bool:  # noqa: ANN001 - pynput's Key | KeyCode | None
        if self._key is None or key is None:
            return False
        return key == self._key

    def _handle_press(self, key) -> None:  # noqa: ANN001
        if self._matches(key) and not self._pressed:
            self._pressed = True
            try:
                self._on_press()
            except Exception:
                self._pressed = False
                raise

    def _handle_release(self, key) -> None:  # noqa: ANN001
        if self._matches(key) and self._pressed:
            self._pressed = False
            try:
                self._on_release()
            except Exception:
                raise

    def start(self) -> None:
        if self._listener is not None:
            return
        if keyboard is None:
            raise RuntimeError("pynput is not installed")
        self._listener = keyboard.Listener(  # type: ignore[attr-defined]
            on_press=self._handle_press,
            on_release=self._handle_release,
            suppress=self._suppress,
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None

    def join(self) -> None:
        if self._listener is not None:
            self._listener.join()

    def is_running(self) -> bool:
        return self._listener is not None and self._listener.is_alive()


class EvdevHotkeyListener:
    """Detects press and release of a single configurable key via evdev.

    Used on Linux Wayland sessions where pynput's X11 backend can't
    reach a display. The listener opens every keyboard-shaped device
    in `/dev/input/event*` (see `find_keyboard_devices`) and runs
    `read_loop()` on a background thread PER device. The shared
    `_pressed` latch dedups events so `on_press` fires exactly once per
    physical press even when more than one device reports it (rare, but
    possible when a Bluetooth receiver and a USB keyboard both emit
    the same keycode, or when one physical keyboard exposes multiple
    event nodes).

    The multi-device design fixes the "wrong keyboard picked" bug: a
    user with two keyboards (e.g. a Bluetooth receiver + a USB gaming
    keyboard) used to have a 50/50 chance the listener attached to the
    one they weren't typing on. Now both are listened to and the latch
    dedups. The cost is one thread + one fd per extra keyboard — almost
    always 1-2, cheap.

    Linux evdev sends `EV_KEY / value=2` (KEY_REPEATED) events when a
    key is held. The `_pressed` latch (same pattern as the pynput
    listener) suppresses these so `on_press` fires exactly once per
    physical press.

    No `EVIOCGRAB` is used — events flow to the focused application as
    well as to us, so the user can type normally in other apps while
    speakinput is running. The trade-off is that another key-grabber
    running on the same device (e.g. sxhkd) can also see the hotkey
    and double-fire. Documented in the README's Wayland section.
    """

    def __init__(
        self,
        keycode: int,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        device=None,  # type: ignore[assignment]
    ) -> None:
        if evdev is None:
            raise RuntimeError("evdev is not installed")
        self._keycode = keycode
        self._on_press = on_press
        self._on_release = on_release
        # Optional injection point for tests — pass a single MagicMock
        # (shaped like an InputDevice with .path/.close/.read_loop) or
        # a list of them. The listener treats both uniformly: it
        # always reads from a list. A single device is wrapped.
        self._device_arg = device
        self._devices: list = []
        self._threads: list[threading.Thread] = []
        self._pressed = False
        # evdev can fire the same key event from two devices in
        # parallel (a USB keyboard and a laptop built-in, say). The
        # latch check+set is not atomic against another device
        # thread without a lock — both threads can see
        # `self._pressed is False`, both pass the guard, both call
        # on_press. The user then gets two parallel push-to-talk
        # sessions: the second press acquires the busy lock only
        # after the first releases (at finalize), so the first
        # press's release is delivered to a now-orphaned on_release
        # call. Symptoms: every other press looks like it "ate"
        # the next one. Lock the latch check+set so only one
        # device thread can win the press and only one can win the
        # release.
        self._latch_lock = threading.Lock()

    def _open_devices(self) -> list:
        if self._device_arg is not None:
            # Accept either a single device or an iterable of devices
            # for test ergonomics and symmetry with the auto path.
            if isinstance(self._device_arg, (list, tuple)):
                return list(self._device_arg)
            return [self._device_arg]
        return find_keyboard_devices()

    def _handle_event(self, event) -> None:  # noqa: ANN001 - evdev event tuple
        # evdev events are (sec, usec, type, code, value). We only care
        # about EV_KEY events on our target keycode. value: 1 = press,
        # 0 = release, 2 = repeat (suppressed by the latch).
        if event.type != _ecodes.EV_KEY:
            return
        if event.code != self._keycode:
            return
        # The latch is the only piece of state shared between per-
        # device read threads, so it gets a dedicated lock (separate
        # from anything the user's callback might take). Cheap —
        # just a `compare-and-set` of a bool.
        if event.value == 1:  # press
            with self._latch_lock:
                if self._pressed:
                    return  # key-repeat OR another device already fired
                self._pressed = True
            self._on_press()
        elif event.value == 0:  # release
            with self._latch_lock:
                if not self._pressed:
                    return  # release without prior press; noop
                self._pressed = False
            self._on_release()
        # value == 2 (repeat) is intentionally ignored.

    def _loop(self, device) -> None:  # noqa: ANN001
        try:
            for event in device.read_loop():
                try:
                    self._handle_event(event)
                except Exception:
                    # A bug in the user's callback shouldn't kill the
                    # listener thread silently. Swallow and continue;
                    # the next event gets a clean shot.
                    # (Pynput's listener has the same behavior.)
                    import sys

                    print(
                        "[hotkey] callback raised; continuing",
                        file=sys.stderr,
                        flush=True,
                    )
        except OSError:
            # The device was closed (stop() path). read_loop() unblocks
            # with OSError on closed-fd. Normal exit.
            pass

    def start(self) -> None:
        if self._threads:
            return
        self._devices = self._open_devices()
        for dev in self._devices:
            t = threading.Thread(
                target=self._loop,
                args=(dev,),
                name=f"speakinput-evdev-hotkey({getattr(dev, 'path', '?')})",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        for dev in self._devices:
            try:
                dev.close()
            except Exception:
                pass
        # close() unblocks each read_loop(); join with a short timeout
        # so a stuck kernel read doesn't block shutdown.
        for t in self._threads:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        self._devices = []
        self._threads = []
        # Reset under the latch lock so an in-flight device thread
        # that lost the race to close() doesn't see stale state on
        # the next start().
        with self._latch_lock:
            self._pressed = False

    def join(self) -> None:
        for t in self._threads:
            try:
                t.join()
            except Exception:
                pass

    def is_running(self) -> bool:
        # The listener is "running" while ANY device thread is alive.
        # If all of them died (e.g. every keyboard was unplugged at
        # once), we're effectively dead. The liveness watcher in app.py
        # uses this to warn the user.
        return bool(self._threads) and any(t.is_alive() for t in self._threads)
