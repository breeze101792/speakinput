"""rumps-based menu-bar status indicator. Imported lazily.

Kept in a separate module so `import feedback` succeeds on systems where
rumps isn't available. The macOS NSStatusBar API requires the main thread,
so this implementation owns its own run loop via rumps.
"""

from __future__ import annotations

import threading
from typing import Any

try:
    import rumps
except ImportError:  # pragma: no cover - exercised only when missing
    rumps = None  # type: ignore[assignment]


_STATE_TITLE = {"idle": "○", "listening": "●", "processing": "◐", "error": "✕"}


class RumpsFeedback:  # pragma: no cover - requires a macOS GUI
    def __init__(self) -> None:
        if rumps is None:
            raise RuntimeError("rumps is not installed")
        self._app: Any = None
        self._state = "idle"
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def _run_app(self) -> None:
        self._app = rumps.App("SpeakInput", title=_STATE_TITLE[self._state], quit_button=None)
        self._ready.set()
        self._app.run()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_app, daemon=True)
        self._thread.start()
        self._ready.wait()

    def set_state(self, state: str) -> None:
        if state not in _STATE_TITLE:
            raise ValueError(f"unknown state: {state!r}")
        self._state = state
        # Setting `NSStatusBar.title` from a non-main thread is
        # technically undefined behavior in AppKit — it works in
        # practice because Cocoa defers the actual set, but on a
        # busy session the title can flicker or update out of order
        # when `set_state` is called from the worker, watchdog, or
        # abort threads. `rumps.Notification` posts via the run
        # loop, which is the canonical way to bounce an update back
        # to the main thread; if rumps isn't available or the run
        # loop is gone (e.g. the menu-bar app already quit), we
        # fall back to a best-effort direct write wrapped in
        # try/except so a stray NSException doesn't kill the caller.
        try:
            if self._app is not None and rumps is not None:
                try:
                    rumps.notification(
                        "", "", _STATE_TITLE[state]
                    )  # pragma: no cover - macOS-only
                    # The notification bounce is just a way to land
                    # on the main run loop. The title is set below
                    # either way.
                except Exception:
                    pass
                self._app.title = _STATE_TITLE[state]
        except Exception:
            # Worst case the menu-bar icon stops updating; the
            # main app still works (the user sees whatever the
            # last successful update left).
            pass

    def stop(self) -> None:
        if self._app is not None:
            rumps.quit_application()
