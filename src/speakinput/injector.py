"""Output injection: type the recognized text into the focused field."""

from __future__ import annotations

import string
import subprocess
import sys
import threading
from typing import Protocol

# Note: pynput is intentionally NOT imported at module load time. Importing
# `pynput.keyboard` starts a background thread that queries the macOS input
# source; on a misconfigured system (or in a test process without Input
# Monitoring permission) that thread can abort the process with a Carbon /
# HIToolbox error. We lazy-import on first use inside TypingInjector.

try:
    import pyperclip
except ImportError:  # pragma: no cover - pyperclip is a hard dep
    pyperclip = None  # type: ignore[assignment]


_PRINTABLE_ASCII = set(string.printable) - set("\t\n\r\x0b\x0c")


class Injector(Protocol):
    def inject(self, text: str) -> None: ...


def is_ascii_safe(text: str) -> bool:
    """True when `text` can be sent character-by-character without losing meaning.

    Allows printable ASCII plus common whitespace. Anything outside (CJK,
    accented chars, emoji, box-drawing) goes through the clipboard path.
    """
    return all(ch in _PRINTABLE_ASCII or ch in (" ", "\t") for ch in text)


def _pbcopy(text: str) -> None:
    """Write text to the macOS clipboard via pbcopy. Falls back to pyperclip elsewhere."""
    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    elif pyperclip is not None:
        pyperclip.copy(text)
    else:  # pragma: no cover - pyperclip is a hard dep
        raise RuntimeError("No clipboard backend available")


def _pbcopy_restore(text: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    elif pyperclip is not None:
        pyperclip.copy(text)


class TypingInjector:
    """Types text into the focused field.

    ASCII path: character-by-character via pynput (fast, no side effects).
    Unicode path: write to clipboard, send Cmd+V, restore prior clipboard
    contents after a short delay. The restore is what keeps this from being
    a destructive operation on the user's clipboard.
    """

    def __init__(
        self,
        restore_clipboard_ms: int = 50,
        trailing_space: bool = True,
    ) -> None:
        # Lazy: see module docstring. pynput.keyboard.Controller triggers a
        # background thread that can abort on macOS in some environments.
        self._controller = None
        self._key = None
        self._restore_ms = restore_clipboard_ms
        self._trailing_space = trailing_space
        # Used by the Unicode path. None means "no prior clipboard to restore".
        self._prior_clipboard: str | None = None
        self._lock = threading.Lock()

    def _ensure_pynput(self):
        if self._controller is None:
            # We import through `sys.modules` so tests can swap
            # `sys.modules['pynput.keyboard']` for a stub without our module
            # holding a direct reference. A direct `from … import …` would
            # capture the import in our globals and bypass the swap.
            import importlib

            keyboard_mod = importlib.import_module("pynput.keyboard")
            self._controller = keyboard_mod.Controller()
            self._key = keyboard_mod.Key
        return self._controller, self._key

    def inject(self, text: str) -> None:
        if not text:
            return
        payload = text + " " if self._trailing_space else text
        if is_ascii_safe(payload):
            controller, _ = self._ensure_pynput()
            controller.type(payload)
            return
        self._inject_unicode(payload)

    def _inject_unicode(self, text: str) -> None:
        with self._lock:
            prior = None
            try:
                if pyperclip is not None:
                    prior = pyperclip.paste()
            except Exception:
                prior = None
            self._prior_clipboard = prior

            _pbcopy(text)
            # Hold Cmd, press V, release. On non-mac we use ctrl+v; both pynput
            # and our restore flow are no-ops on platforms without a clipboard
            # backend, so this path is effectively macOS-only in v1.
            controller, key = self._ensure_pynput()
            modifier = key.cmd if sys.platform == "darwin" else key.ctrl
            with controller.pressed(modifier):
                controller.tap("v")

            if self._restore_ms > 0 and self._prior_clipboard is not None:
                # Schedule the restore. We capture the value now in case
                # another injection interleaves before the timer fires.
                captured = self._prior_clipboard
                threading.Timer(
                    self._restore_ms / 1000.0,
                    _pbcopy_restore,
                    args=(captured,),
                ).start()
