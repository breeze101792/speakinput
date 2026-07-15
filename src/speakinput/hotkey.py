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


def find_keyboard_device():
    """Scan /dev/input/event* and return the best keyboard candidate.

    Selection logic — we have to pick one device out of typically several
    `EV_KEY`-capable siblings (the "SINO WEALTH Gaming KB" exposes
    event12 for the main keyboard, event14 for consumer control, event15
    for system control, and a ydotool virtual device may show up first
    in the list with hundreds of keys). The right answer is the
    physical *main* keyboard, not the most-capable-by-raw-count device.

    Algorithm: collect every device that exposes >= `_KEYBOARD_KEY_COUNT_MIN`
    `EV_KEY` codes, then rank by:

    1. **Has a physical `phys` path** (e.g. ``usb-0000:03:00.0-3.4.4/input0``)
       — virtual devices (`ydotoold`, `uinput`-based relays) have an
       empty `phys` and are demoted.
    2. **Exposes `EV_REP` capability** (auto-repeat, which only real
       keyboards implement) — consumer-control siblings don't have it.
    3. **Most keycodes** — breaks ties when two physical devices tie on
       the above.

    Returns an `evdev.InputDevice` opened for reading. The caller is
    responsible for `close()`ing it (the listener does this in `stop`).

    Raises `HotkeyError` if no suitable device is found, with a message
    pointing at the usual culprits (no `/dev/input` access, user not in
    the `input` group, no keyboard attached).
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
        # tuple comparison does the right thing (physical first, then
        # EV_REP, then key count).
        has_phys = bool(getattr(dev, "phys", ""))
        has_rep = _ecodes.EV_REP in caps
        candidates.append(
            ((has_phys, has_rep, len(key_codes)), dev)
        )
    if not candidates:
        raise HotkeyError(
            "no keyboard device found in /dev/input — check that the user is in "
            "the `input` group, that a keyboard is attached, and that "
            "/dev/input is readable. (A v2 follow-up will add a "
            "[hotkey].device_path config override for multi-keyboard setups.)"
        )
    # Pick the best; close the rest so we don't leak fd's.
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0][1]
    for _, dev in candidates[1:]:
        try:
            dev.close()
        except Exception:
            pass
    return chosen


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
    reach a display. The listener opens the first keyboard-shaped
    device in `/dev/input/event*` (see `find_keyboard_device`) and
    runs `read_loop()` on a background thread.

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
        # Optional injection point for tests — pass a MagicMock that
        # has .path, .close(), and .read_loop() shaped correctly.
        self._device_arg = device
        self._device = None  # type: ignore[assignment]
        self._thread: threading.Thread | None = None
        self._pressed = False

    def _open_device(self):
        if self._device_arg is not None:
            return self._device_arg
        return find_keyboard_device()

    def _handle_event(self, event) -> None:  # noqa: ANN001 - evdev event tuple
        # evdev events are (sec, usec, type, code, value). We only care
        # about EV_KEY events on our target keycode. value: 1 = press,
        # 0 = release, 2 = repeat (suppressed by the latch).
        if event.type != _ecodes.EV_KEY:
            return
        if event.code != self._keycode:
            return
        if event.value == 1:  # press
            if self._pressed:
                return  # key-repeat; latch holds
            self._pressed = True
            self._on_press()
        elif event.value == 0:  # release
            if not self._pressed:
                return  # release without prior press; noop
            self._pressed = False
            self._on_release()
        # value == 2 (repeat) is intentionally ignored.

    def _loop(self) -> None:
        try:
            for event in self._device.read_loop():
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
        if self._thread is not None:
            return
        self._device = self._open_device()
        self._thread = threading.Thread(
            target=self._loop, name="speakinput-evdev-hotkey", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        if self._thread is not None:
            # close() unblocks read_loop(); join with a short timeout
            # so a stuck kernel read doesn't block shutdown.
            self._thread.join(timeout=1.0)
            self._thread = None
        self._pressed = False

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
