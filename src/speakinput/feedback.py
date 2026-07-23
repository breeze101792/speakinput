"""Optional menu-bar status indicator.

The implementation degrades gracefully: if `rumps` isn't installed, all
methods are no-ops so the app still runs headless. The CLI prints status
to stderr as a fallback.
"""

from __future__ import annotations

import sys
import threading
from typing import Protocol


class Feedback(Protocol):
    def set_state(self, state: str) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


_STATES = ("idle", "listening", "processing", "error")


class NullFeedback:
    """No-op feedback for headless runs and tests."""

    def set_state(self, state: str) -> None:  # noqa: ARG002
        return

    def start(self) -> None:
        return

    def stop(self) -> None:
        return


class StderrFeedback:
    """Prints the current state to stderr. Useful when rumps isn't available."""

    def __init__(self) -> None:
        self._state = "idle"
        # Multiple threads (worker, watchdog, abort) can call
        # `set_state` concurrently. CPython's print is atomic
        # under the GIL, but on free-threaded (PEP 703) Python
        # the stderr write can be torn across two states. Cheap
        # insurance — the lock also serializes the state read+write
        # pair so a `set_state("error")` interleaved with a
        # `set_state("idle")` from another thread never leaves a
        # stale title on screen.
        self._lock = threading.Lock()

    def set_state(self, state: str) -> None:
        if state not in _STATES:
            raise ValueError(f"unknown feedback state: {state!r}")
        with self._lock:
            self._state = state
            print(f"[speakinput] {state}", file=sys.stderr, flush=True)

    def start(self) -> None:
        return

    def stop(self) -> None:
        return


def make_feedback() -> Feedback:
    """Pick the best feedback implementation available in this environment."""
    try:
        from speakinput._feedback_rumps import RumpsFeedback  # type: ignore[import-not-found]

        return RumpsFeedback()
    except Exception:
        return StderrFeedback()
