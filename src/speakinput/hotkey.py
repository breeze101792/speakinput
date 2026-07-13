"""Global hotkey listener for push-to-talk."""

from __future__ import annotations

from typing import Callable, Protocol

# Note: pynput is intentionally NOT imported at module load time. See the
# same note in injector.py — importing pynput starts a Carbon/HIToolbox
# background thread that can abort the process on misconfigured macOS
# systems (notably the test runner without Input Monitoring permission).

from speakinput.config import VALID_HOTKEYS

_KEY_NAMES = {
    "alt_r": lambda k: k.alt_r,
    "ctrl_r": lambda k: k.ctrl_r,
    "cmd_r": lambda k: k.cmd_r,
    "shift_r": lambda k: k.shift_r,
    "caps_lock": lambda k: k.caps_lock,
    "f12": lambda k: k.f12,
}


def resolve_key(name: str):
    if name not in VALID_HOTKEYS:
        raise ValueError(f"unknown hotkey {name!r}; expected one of {VALID_HOTKEYS}")
    from pynput import keyboard  # lazy import — see module docstring

    if keyboard is None:
        raise RuntimeError("pynput is not installed; cannot resolve hotkey")
    return _KEY_NAMES[name](keyboard.Key)


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
        from pynput import keyboard  # lazy import — see module docstring

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
