"""Output injection: type the recognized text into the focused field.

Three backends are available, picked per-platform by `select_injector`:

* `TypingInjector` (pynput) — used on macOS (HIToolbox), Windows
  (SendInput), and Linux when an X11 display is reachable. pynput
  synthesizes keystrokes via XTest on X11 Linux. On a pure Wayland
  session without XWayland, those synthetic events go nowhere.

* `WtypeInjector` (`wtype`) — used on Linux Wayland. wtype speaks
  the wlroots `virtual-keyboard-unstable-v1` protocol directly, so
  it works on any wlroots-based compositor (Sway, swayfx, Hyprland,
  river). It is daemon-free. Not supported on GNOME / KDE.

* `YdotoolInjector` (`ydotool`) — used on Linux Wayland as a
  fallback when wtype is unavailable. ydotool uses a user-mode
  uinput driver; it requires the `ydotoold` daemon to be running
  and reachable via a Unix socket (`$YDOTOOL_SOCKET` or
  `/run/user/<uid>/.ydotool_socket`).

All three expose the same public surface — `inject(text)` — and
share the same ASCII / Unicode split. ASCII text is typed
character-by-character through the backend (fast, no side effects).
Unicode (CJK, accents, emoji) goes through the clipboard: write to
clipboard, send the paste shortcut, restore the prior clipboard
contents after a short delay. The restore is what keeps this from
being a destructive operation on the user's clipboard.
"""

from __future__ import annotations

import os
import shutil
import string
import subprocess
import sys
import threading
from typing import Protocol

try:
    from pynput.keyboard import Controller, Key
except ImportError:  # pragma: no cover - pynput is a hard dep at runtime
    Controller = None  # type: ignore[assignment]
    Key = None  # type: ignore[assignment]

try:
    import pyperclip
except ImportError:  # pragma: no cover - pyperclip is a hard dep
    pyperclip = None  # type: ignore[assignment]

from speakinput.config import InjectConfig


_PRINTABLE_ASCII = set(string.printable) - set("\t\n\r\x0b\x0c")


class Injector(Protocol):
    def inject(self, text: str) -> None: ...


def is_ascii_safe(text: str) -> bool:
    """True when `text` can be sent character-by-character without losing meaning.

    Allows printable ASCII plus common whitespace. Anything outside (CJK,
    accented chars, emoji, box-drawing) goes through the clipboard path.
    """
    return all(ch in _PRINTABLE_ASCII or ch in (" ", "\t") for ch in text)


def _wl_copy(text: str) -> None:
    """Write `text` to the Wayland clipboard via `wl-copy`.

    Used as a fallback when `pyperclip` isn't installed or doesn't
    work on the current session (e.g. a pure Wayland box without
    any X11 backend that pyperclip can find).
    """
    subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True)


def _pbcopy(text: str) -> None:
    """Write text to the system clipboard. macOS→pbcopy, Linux→pyperclip."""
    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    elif pyperclip is not None:
        pyperclip.copy(text)
    else:
        # Last-resort Wayland fallback. Doesn't apply on macOS, where
        # pbcopy is always available.
        _wl_copy(text)


def _pbcopy_restore(text: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    elif pyperclip is not None:
        pyperclip.copy(text)
    else:
        try:
            _wl_copy(text)
        except Exception:
            pass


def _clipboard_read() -> str | None:
    """Snapshot the current clipboard for later restore. None on failure."""
    if pyperclip is not None:
        try:
            return pyperclip.paste()
        except Exception:
            return None
    # No pyperclip — try wl-paste.
    try:
        result = subprocess.run(
            ["wl-paste", "-n"], capture_output=True, check=True
        )
        return result.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None


def _schedule_clipboard_restore(value: str, delay_ms: int) -> None:
    """Restore the clipboard after `delay_ms` milliseconds."""
    if delay_ms <= 0:
        return
    threading.Timer(delay_ms / 1000.0, _pbcopy_restore, args=(value,)).start()


def _run_subprocess(args: list[str], env: dict | None = None) -> None:
    """Run a subprocess synchronously, raising on failure with stderr attached."""
    subprocess.run(args, check=True, env=env)


class TypingInjector:
    """Types text into the focused field using pynput (HIToolbox / X11 / Win32)."""

    def __init__(
        self,
        restore_clipboard_ms: int = 50,
        trailing_space: bool = True,
    ) -> None:
        if Controller is None:
            raise RuntimeError("pynput is not installed")
        self._controller = Controller()
        self._restore_ms = restore_clipboard_ms
        self._trailing_space = trailing_space
        # Used by the Unicode path. None means "no prior clipboard to restore".
        self._prior_clipboard: str | None = None
        self._lock = threading.Lock()

    def inject(self, text: str) -> None:
        if not text:
            return
        payload = text + " " if self._trailing_space else text
        if is_ascii_safe(payload):
            self._controller.type(payload)
            return
        self._inject_unicode(payload)

    def _inject_unicode(self, text: str) -> None:
        with self._lock:
            self._prior_clipboard = _clipboard_read()
            _pbcopy(text)
            # Hold Cmd (macOS) or Ctrl (everywhere else), press V, release.
            modifier = Key.cmd if sys.platform == "darwin" else Key.ctrl
            with self._controller.pressed(modifier):
                self._controller.tap("v")

            if self._restore_ms > 0 and self._prior_clipboard is not None:
                captured = self._prior_clipboard
                _schedule_clipboard_restore(captured, self._restore_ms)


class WtypeInjector:
    """Types text into the focused field via `wtype` (wlroots virtual-keyboard).

    ASCII path: `wtype -- <text>` (one call, fast).
    Unicode path: `wl-copy` + `wtype -M ctrl -k v` (clipboard paste).
    The prior clipboard is snapshotted and restored after
    `restore_clipboard_ms` to avoid clobbering whatever the user
    had there.

    The `wtype` binary must be on PATH. The `wl-copy` binary is
    only needed for the Unicode path; if it's missing, the ASCII
    path still works.
    """

    def __init__(
        self,
        restore_clipboard_ms: int = 50,
        trailing_space: bool = True,
    ) -> None:
        if shutil.which("wtype") is None:
            raise RuntimeError(
                "wtype is not installed; install it from your distro's "
                "package manager (Arch: pacman -S wtype, Debian: "
                "apt install wtype). The WtypeInjector is the default "
                "Wayland typing backend."
            )
        self._restore_ms = restore_clipboard_ms
        self._trailing_space = trailing_space
        self._prior_clipboard: str | None = None
        self._lock = threading.Lock()

    def inject(self, text: str) -> None:
        if not text:
            return
        payload = text + " " if self._trailing_space else text
        if is_ascii_safe(payload):
            # `wtype -- <text>` types the text without interpreting
            # the leading `--` as a flag. Newlines are typed as
            # `Return` automatically by wtype.
            _run_subprocess(["wtype", "--", payload])
            return
        self._inject_unicode(payload)

    def _inject_unicode(self, text: str) -> None:
        with self._lock:
            self._prior_clipboard = _clipboard_read()
            _pbcopy(text)
            # Send Ctrl+V: `-M ctrl` holds the modifier, `-k v` taps the key.
            _run_subprocess(["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"])

            if self._restore_ms > 0 and self._prior_clipboard is not None:
                captured = self._prior_clipboard
                _schedule_clipboard_restore(captured, self._restore_ms)


class YdotoolInjector:
    """Types text into the focused field via `ydotool` (uinput-based).

    ASCII path: `ydotool type <text>` (one call).
    Unicode path: same `wl-copy` + paste shortcut pattern, but the
    paste is sent as `ydotool key 29:1 47:1 47:0 29:0` (left-Ctrl
    down, V down, V up, Ctrl up — kernel keycodes).

    Requires the `ydotoold` daemon to be running. The socket is
    discovered from `YDOTOOL_SOCKET` if set, else from
    `/run/user/<uid>/.ydotool_socket`, else an error.
    """

    def __init__(
        self,
        restore_clipboard_ms: int = 50,
        trailing_space: bool = True,
    ) -> None:
        if shutil.which("ydotool") is None:
            raise RuntimeError(
                "ydotool is not installed; install ydotool and start "
                "ydotoold (systemctl --user enable --now ydotool). "
                "The YdotoolInjector is the fallback Wayland typing backend."
            )
        self._restore_ms = restore_clipboard_ms
        self._trailing_space = trailing_space
        self._socket = os.environ.get("YDOTOOL_SOCKET") or _default_ydotool_socket()
        if self._socket is None:
            raise RuntimeError(
                "YDOTOOL_SOCKET is not set and /run/user/<uid>/.ydotool_socket "
                "is not present; start ydotoold (systemctl --user enable --now "
                "ydotool) or set YDOTOOL_SOCKET to its socket path."
            )
        self._prior_clipboard: str | None = None
        self._lock = threading.Lock()

    def inject(self, text: str) -> None:
        if not text:
            return
        payload = text + " " if self._trailing_space else text
        env = {"YDOTOOL_SOCKET": self._socket, **os.environ}
        if is_ascii_safe(payload):
            _run_subprocess(["ydotool", "type", "--", payload], env=env)
            return
        self._inject_unicode(payload, env=env)

    def _inject_unicode(self, text: str, env: dict) -> None:
        with self._lock:
            self._prior_clipboard = _clipboard_read()
            _pbcopy(text)
            # Linux keycodes: KEY_LEFTCTRL = 29, KEY_V = 47.
            # Send left-Ctrl down, V down, V up, left-Ctrl up.
            _run_subprocess(
                ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
                env=env,
            )

            if self._restore_ms > 0 and self._prior_clipboard is not None:
                captured = self._prior_clipboard
                _schedule_clipboard_restore(captured, self._restore_ms)


def _default_ydotool_socket() -> str | None:
    """Return the standard ydotoold socket path for the current user, or None."""
    uid = os.getuid() if hasattr(os, "getuid") else None
    if uid is None:
        return None
    candidate = f"/run/user/{uid}/.ydotool_socket"
    return candidate if os.path.exists(candidate) else None


def select_injector(config: InjectConfig) -> Injector:
    """Pick the right `Injector` for the current platform + user override.

    `config.backend` is one of:

    - `"auto"` (default): platform-driven selection
    - `"pynput"`: always TypingInjector
    - `"wtype"`: always WtypeInjector (raises if missing)
    - `"ydotool"`: always YdotoolInjector (raises if missing)

    Auto rules:

    - macOS, Windows, Linux+X11: pynput
    - Linux+Wayland: wtype → ydotool → pynput (last-ditch — works
      if XWayland is also running)
    """
    backend = (config.backend or "auto").lower()

    if backend == "pynput":
        return TypingInjector(
            restore_clipboard_ms=config.restore_clipboard_ms,
            trailing_space=config.trailing_space,
        )
    if backend == "wtype":
        return WtypeInjector(
            restore_clipboard_ms=config.restore_clipboard_ms,
            trailing_space=config.trailing_space,
        )
    if backend == "ydotool":
        return YdotoolInjector(
            restore_clipboard_ms=config.restore_clipboard_ms,
            trailing_space=config.trailing_space,
        )
    if backend != "auto":
        raise ValueError(
            f"unknown inject.backend {config.backend!r}; expected one of "
            f"'auto', 'pynput', 'wtype', 'ydotool'"
        )

    # Auto: pick by platform.
    if sys.platform == "darwin":
        return TypingInjector(
            restore_clipboard_ms=config.restore_clipboard_ms,
            trailing_space=config.trailing_space,
        )
    if sys.platform == "win32":
        return TypingInjector(
            restore_clipboard_ms=config.restore_clipboard_ms,
            trailing_space=config.trailing_space,
        )
    # Linux
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        for cls in (WtypeInjector, YdotoolInjector):
            try:
                return cls(
                    restore_clipboard_ms=config.restore_clipboard_ms,
                    trailing_space=config.trailing_space,
                )
            except RuntimeError:
                # Try the next backend. We don't warn here — the user
                # picked auto, so falling through silently is the
                # right behavior. The final pynput fallback will warn
                # at injection time if it really doesn't work.
                continue
        # Neither wtype nor ydotool. pynput is the last resort; it
        # only works on XWayland, but that's better than nothing.
        return TypingInjector(
            restore_clipboard_ms=config.restore_clipboard_ms,
            trailing_space=config.trailing_space,
        )
    # Linux + X11 (or session type unknown)
    return TypingInjector(
        restore_clipboard_ms=config.restore_clipboard_ms,
        trailing_space=config.trailing_space,
    )
