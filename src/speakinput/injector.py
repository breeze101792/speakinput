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

# Bounded wait for any subprocess we shell out to. pbcopy can hang
# indefinitely if the macOS pasteboard server is unresponsive (observed
# after long sleeps); wtype/ydotool can hang on a wedged Wayland
# compositor or uinput subsystem. Without a timeout, a single hung
# subprocess blocks the event worker indefinitely and the user's
# hotkey silently stops working.
_SUBPROCESS_TIMEOUT_S = 5.0


class Injector(Protocol):
    def inject(self, text: str) -> None: ...


def is_ascii_safe(text: str) -> bool:
    """True when `text` can be sent character-by-character without losing meaning.

    Allows printable ASCII plus common whitespace. Anything outside (CJK,
    accented chars, emoji, box-drawing) goes through the clipboard path.
    """
    return all(ch in _PRINTABLE_ASCII or ch in (" ", "\t") for ch in text)


def _wl_paste() -> str | None:
    """Snapshot the Wayland clipboard via `wl-paste -n`.

    Bounded by `_SUBPROCESS_TIMEOUT_S` — a wedged wl-paste (clipboard
    manager crashed, session in transition) must not freeze the
    event worker.
    """
    try:
        cp = subprocess.run(
            ["wl-paste", "-n"],
            capture_output=True,
            check=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
        return cp.stdout.decode("utf-8", errors="replace")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return None


def _pbcopy(text: str) -> None:
    """Write text to the system clipboard. macOS→pbcopy, Linux→pyperclip → wl-copy."""
    if sys.platform == "darwin":
        # Use a hard timeout: a wedged pbcopy after sleep has been
        # observed to block the event worker for the entire wait
        # window. On macOS the kernel reaps the timed-out pbcopy
        # when its parent process exits.
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    elif pyperclip is not None:
        try:
            pyperclip.copy(text)
        except Exception:
            # pyperclip can raise on a wedged X11 selection owner
            # (clipboard manager crashed). Fall through to wl-copy
            # on Wayland, or re-raise on X11 — the unicode path will
            # catch the failure and bail without typing garbage.
            if os.environ.get("XDG_SESSION_TYPE") == "wayland":
                _wl_copy(text)
            else:
                raise
    else:
        # Last-resort Wayland fallback. Doesn't apply on macOS, where
        # pbcopy is always available.
        _wl_copy(text)


def _wl_copy(text: str) -> None:
    """Write `text` to the Wayland clipboard via `wl-copy`.

    Used as a fallback when `pyperclip` isn't installed or doesn't
    work on the current session (e.g. a pure Wayland box without
    any X11 backend that pyperclip can find). Bounded by
    `_SUBPROCESS_TIMEOUT_S` so a hung `wl-copy` doesn't freeze
    the event worker.
    """
    subprocess.run(
        ["wl-copy"],
        input=text.encode("utf-8"),
        check=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
    )


def _pbcopy_restore(text: str) -> None:
    """Best-effort restore of the prior clipboard contents.

    Runs on a daemon timer thread (see `_schedule_clipboard_restore`).
    Any failure is swallowed silently — the user's next paste will
    see whatever the last inject wrote, which is the most common
    expectation. Surface a single warning per failure for debugging,
    bounded so a permanent failure (e.g. pbcopy missing) doesn't
    spam the log on every press.
    """
    try:
        if sys.platform == "darwin":
            subprocess.run(
                ["pbcopy"],
                input=text.encode("utf-8"),
                check=True,
                timeout=_SUBPROCESS_TIMEOUT_S,
            )
        elif pyperclip is not None:
            pyperclip.copy(text)
        else:
            _wl_copy(text)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        # Best-effort. The clipboard will keep the last injected
        # text, which is usually what the user wants.
        pass


def _clipboard_read() -> str | None:
    """Snapshot the current clipboard for later restore. None on failure."""
    if pyperclip is not None:
        try:
            return pyperclip.paste()
        except Exception:
            return None
    # No pyperclip — try wl-paste.
    return _wl_paste()


def _schedule_clipboard_restore(value: str, delay_ms: int) -> None:
    """Restore the clipboard after `delay_ms` milliseconds.

    The restore thread is a *daemon* — the default
    `threading.Timer(daemon=False)` (Timer inherits from Thread,
    which is non-daemon by default) would let a pending restore
    block process exit indefinitely. `restore_clipboard_ms` is
    user-configurable with no upper cap, so a 60-second restore
    would freeze shutdown for a minute. Daemon means: the kernel
    reaps the timer on exit and the user's next paste sees
    whatever the last inject wrote.
    """
    if delay_ms <= 0:
        return
    t = threading.Timer(delay_ms / 1000.0, _pbcopy_restore, args=(value,))
    t.daemon = True
    t.start()


def _run_subprocess(args: list[str], env: dict | None = None) -> None:
    """Run a subprocess synchronously, raising on failure with stderr attached.

    Bounded by `_SUBPROCESS_TIMEOUT_S`: a hung `wtype` or `ydotool`
    (Wayland compositor wedged, ydotoold stuck) would otherwise freeze
    the event worker and the user's hotkey.
    """
    subprocess.run(
        args,
        check=True,
        env=env,
        timeout=_SUBPROCESS_TIMEOUT_S,
    )


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
        # Serializes both ASCII and Unicode inject calls so a chunked
        # watchdog body and a finalize can't interleave their typing
        # through pynput (which has its own internal lock — this one
        # is for the unicode clipboard-paste path, where the lock
        # also keeps the snapshot/restore/restore sequence atomic).
        self._lock = threading.Lock()

    def inject(self, text: str) -> None:
        if not text:
            return
        payload = text + " " if self._trailing_space else text
        if is_ascii_safe(payload):
            with self._lock:
                self._controller.type(payload)
            return
        self._inject_unicode(payload)

    def _inject_unicode(self, text: str) -> None:
        with self._lock:
            self._prior_clipboard = _clipboard_read()
            try:
                _pbcopy(text)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
                # Clipboard write failed (wedged pbcopy after sleep,
                # X11 selection owner dead, etc.). Don't paste
                # whatever stale contents the clipboard still has —
                # that would type the wrong text. Bail out and let
                # the caller log the failure.
                print(
                    "[warn] clipboard write failed; skipping unicode inject",
                    file=sys.stderr,
                    flush=True,
                )
                return
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
        # Serializes the whole `inject()` call. Two concurrent
        # `wtype` processes (one from the chunked body, one from
        # finalize) would interleave their keystrokes in the focused
        # field; locking makes the second wait for the first to
        # finish. Cheap — `wtype` returns in milliseconds.
        self._lock = threading.Lock()

    def inject(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            payload = text + " " if self._trailing_space else text
            if is_ascii_safe(payload):
                # `wtype -- <text>` types the text without interpreting
                # the leading `--` as a flag. Newlines are typed as
                # `Return` automatically by wtype.
                _run_subprocess(["wtype", "--", payload])
                return
            self._inject_unicode(payload)

    def _inject_unicode(self, text: str) -> None:
        self._prior_clipboard = _clipboard_read()
        try:
            _pbcopy(text)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
            print(
                "[warn] clipboard write failed; skipping unicode inject",
                file=sys.stderr,
                flush=True,
            )
            return
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
        # Serializes the whole `inject()` call. Same rationale as
        # WtypeInjector — ydotool doesn't synchronize its own
        # concurrent invocations, and two simultaneous calls would
        # interleave kernel keycodes.
        self._lock = threading.Lock()

    def inject(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            env = {"YDOTOOL_SOCKET": self._socket, **os.environ}
            payload = text + " " if self._trailing_space else text
            if is_ascii_safe(payload):
                _run_subprocess(["ydotool", "type", "--", payload], env=env)
                return
            self._inject_unicode(payload, env=env)

    def _inject_unicode(self, text: str, env: dict) -> None:
        self._prior_clipboard = _clipboard_read()
        try:
            _pbcopy(text)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
            print(
                "[warn] clipboard write failed; skipping unicode inject",
                file=sys.stderr,
                flush=True,
            )
            return
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
        missing: list[str] = []
        for cls in (WtypeInjector, YdotoolInjector):
            try:
                return cls(
                    restore_clipboard_ms=config.restore_clipboard_ms,
                    trailing_space=config.trailing_space,
                )
            except RuntimeError as exc:
                # Remember which backend was missing so we can warn
                # the user below (the silent fall-through to pynput
                # is the difference between "hotkey does nothing on
                # pure-Wayland" and a useful "install wtype" hint).
                missing.append(f"{cls.__name__}: {exc}")
                continue
        # Neither wtype nor ydotool is available. pynput is the last
        # resort: it only works when XWayland is also running. On a
        # pure-Wayland session without XWayland, pynput's synthetic
        # events go nowhere and the user types into the void with
        # no error. Tell them now, while we still have a stderr
        # pointer, so the next press isn't mysterious.
        if missing:
            print(
                "[warn] no Wayland typing backend found (wtype and ydotool "
                "both missing). Falling back to pynput, which only types "
                "when XWayland is available. On a pure Wayland session, "
                "the typed text will go nowhere. Install wtype (preferred) "
                "or ydotool for reliable typing.",
                file=sys.stderr,
                flush=True,
            )
        return TypingInjector(
            restore_clipboard_ms=config.restore_clipboard_ms,
            trailing_space=config.trailing_space,
        )
    # Linux + X11 (or session type unknown)
    return TypingInjector(
        restore_clipboard_ms=config.restore_clipboard_ms,
        trailing_space=config.trailing_space,
    )
