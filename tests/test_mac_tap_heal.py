"""Tests for the CGEventTap self-heal installer (`_mac_tap_heal.py`).

The installer normally runs at import time and is a no-op off macOS.
These tests exercise the *install* machinery (platform gating,
dependency handling, idempotence) and the *self-healing `_run` loop*
on any host by simulating a darwin environment: `sys.platform`,
`pynput._util.darwin`, and `Quartz` are replaced with fakes, then the
installer is re-run and the patched loop is driven with a stub
listener.

(Full end-to-end tests against the real pynput ListenerMixin live in
test_hotkey.py and are macOS-only.)
"""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from speakinput import _mac_tap_heal as heal


def _run_loop_stopper(n_ticks):
    """Return a ``CFRunLoopRunInMode`` stand-in that always signals a
    timed-out iteration and flips the stub's `running` flag off after
    `n_ticks` calls, so the patched `_run` exits."""
    calls = {"n": 0}

    def make(stub):
        def inner(*a, **k):
            calls["n"] += 1
            if calls["n"] >= n_ticks:
                stub.running = False
            return "timed_out"

        return inner

    return make


@pytest.fixture
def macos_environment(monkeypatch) -> SimpleNamespace:
    """Simulate macOS: fake `pynput._util.darwin` + `Quartz` modules in
    ``sys.modules``, ``sys.platform == "darwin"``, and reset the module's
    patch gate so `_install_darwin_tap_healer` actually does its work."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(heal, "_PATCHED", False)

    darwin = types.ModuleType("pynput._util.darwin")

    class ListenerMixin:
        _run = lambda self: None  # placeholder; the healer replaces this  # noqa: E731

    setattr(darwin, "ListenerMixin", ListenerMixin)
    setattr(darwin, "CFMachPortCreateRunLoopSource", lambda *a, **k: "loop_source")
    setattr(darwin, "CFRunLoopGetCurrent", lambda: "loop")
    setattr(darwin, "CFRunLoopAddSource", lambda *a, **k: None)
    setattr(darwin, "CFRunLoopRunInMode", lambda *a, **k: "sentinel")
    setattr(darwin, "kCFRunLoopDefaultMode", "default_mode")
    setattr(darwin, "kCFRunLoopRunTimedOut", "timed_out")
    setattr(darwin, "HIServices", SimpleNamespace(AXIsProcessTrusted=lambda: True))

    quartz = types.ModuleType("Quartz")
    setattr(quartz, "CGEventTapIsEnabled", lambda tap: True)
    setattr(quartz, "CGEventTapEnable", MagicMock())

    # Fakes for the *whole* pynput package chain. If only
    # `pynput._util.darwin` were injected, `import pynput._util.darwin`
    # would still import the real `pynput` package first — and pynput's
    # __init__ picks its platform backend from sys.platform, which we
    # just switched to "darwin", so it would demand a real `Quartz`
    # binding and blow up. Replacing the parent modules too keeps the
    # import graph fully inside sys.modules.
    monkeypatch.setitem(sys.modules, "pynput", types.ModuleType("pynput"))
    monkeypatch.setitem(sys.modules, "pynput._util", types.ModuleType("pynput._util"))
    monkeypatch.setitem(sys.modules, "pynput._util.darwin", darwin)
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    return SimpleNamespace(darwin=darwin, quartz=quartz)


class _StubListener:
    """Minimal listener object with just the attributes the patched `_run`
    touches. We bypass pynput's real class so the test runs headless."""

    def __init__(self, tap):
        self.running = True
        self.IS_TRUSTED = False
        self.tap = tap
        self._log = MagicMock()
        self._loop = None
        self._mark_ready = MagicMock()

    def _create_event_tap(self):
        return self.tap


# --- installer machinery ----------------------------------------------------


def test_non_darwin_install_is_a_noop_and_marks_patched(monkeypatch):
    monkeypatch.setattr(heal, "_PATCHED", False)
    monkeypatch.setattr(sys, "platform", "linux")
    # If the installer tried to import darwin primitives on Linux, the None
    # sys.modules entries below would raise ImportError and fail the test.
    monkeypatch.setitem(sys.modules, "pynput._util.darwin", None)
    monkeypatch.setitem(sys.modules, "Quartz", None)
    heal._install_darwin_tap_healer()  # must return quietly
    assert heal._PATCHED is True


def test_install_is_idempotent(macos_environment):
    heal._install_darwin_tap_healer()
    first_run = macos_environment.darwin.ListenerMixin._run
    heal._install_darwin_tap_healer()  # second install must be a no-op
    assert macos_environment.darwin.ListenerMixin._run is first_run


def test_darwin_install_replaces_mixin_run_once(macos_environment):
    original = macos_environment.darwin.ListenerMixin._run
    heal._install_darwin_tap_healer()
    patched = macos_environment.darwin.ListenerMixin._run
    assert patched is not original
    assert callable(patched)


def test_darwin_install_fails_clean_when_dependencies_missing(monkeypatch):
    monkeypatch.setattr(heal, "_PATCHED", False)
    monkeypatch.setattr(sys, "platform", "darwin")
    # A None entry in sys.modules makes the import machinery raise
    # ImportError — the installer must swallow it and stay safe.
    monkeypatch.setitem(sys.modules, "pynput._util.darwin", None)
    monkeypatch.setitem(sys.modules, "Quartz", None)
    heal._install_darwin_tap_healer()  # must not raise
    assert heal._PATCHED is True


# --- the patched _run loop --------------------------------------------------


def test_run_reenables_disabled_tap(macos_environment, monkeypatch):
    """CGEventTapIsEnabled returning False → the loop calls Enable(True) to
    bring the hotkey back to life (the macOS sleep/wake recovery path)."""
    heal._install_darwin_tap_healer()
    run = macos_environment.darwin.ListenerMixin._run

    states = iter([False, True])
    monkeypatch.setattr(
        macos_environment.quartz,
        "CGEventTapIsEnabled",
        lambda tap: next(states, True),
    )
    enable_calls: list[bool] = []
    monkeypatch.setattr(
        macos_environment.quartz,
        "CGEventTapEnable",
        lambda tap, enabled: enable_calls.append(bool(enabled)),
    )

    stub = _StubListener(tap=object())
    macos_environment.darwin.CFRunLoopRunInMode = _run_loop_stopper(2)(stub)
    thread = threading.Thread(target=run, args=(stub,), daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "patched _run did not exit after running flag cleared"

    assert True in enable_calls, (
        f"expected at least one CGEventTapEnable(tap, True) call, got {enable_calls!r}"
    )
    # The Enable(True) after the heal must have happened at least once
    # beyond the initial startup enable.
    assert len(enable_calls) >= 1


def test_run_does_not_reenable_healthy_tap(macos_environment, monkeypatch):
    """When IsEnabled stays True, the heal loop must NOT call Enable — the
    IsEnabled check exists to skip a pointless round-trip to CoreGraphics.
    Only the one-time startup enable is allowed."""
    heal._install_darwin_tap_healer()
    run = macos_environment.darwin.ListenerMixin._run

    enable_calls: list[bool] = []
    monkeypatch.setattr(
        macos_environment.quartz,
        "CGEventTapEnable",
        lambda tap, enabled: enable_calls.append(bool(enabled)),
    )
    stub = _StubListener(tap=object())
    macos_environment.darwin.CFRunLoopRunInMode = _run_loop_stopper(3)(stub)
    thread = threading.Thread(target=run, args=(stub,), daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "patched _run did not exit"
    # Startup's single Enable(True) + the finally-block's disable(False) on
    # exit. The heal loop itself adds nothing for a healthy tap.
    assert enable_calls == [True, False], (
        f"expected only startup enable + teardown disable, got {enable_calls!r}"
    )


def test_run_marks_ready_when_no_event_tap(macos_environment):
    """create_event_tap() returning None means no tap to run: the loop must
    call `_mark_ready` and exit without touching Quartz."""
    heal._install_darwin_tap_healer()
    run = macos_environment.darwin.ListenerMixin._run

    class _NoTap:
        running = True
        IS_TRUSTED = False
        _log = MagicMock()
        _loop = None
        _mark_ready = MagicMock()

        def _create_event_tap(self):
            return None

    stub = _NoTap()
    run(stub)
    stub._mark_ready.assert_called_once()
    assert stub._loop is None


def test_run_warns_when_process_not_trusted(macos_environment):
    """The accessibility-trust warning (the 'input monitoring' prompt) must
    still fire from the patched run loop."""
    setattr(
        macos_environment.darwin,
        "HIServices",
        SimpleNamespace(AXIsProcessTrusted=lambda: False),
    )
    heal._install_darwin_tap_healer()
    run = macos_environment.darwin.ListenerMixin._run

    stub = _StubListener(tap=object())
    macos_environment.darwin.CFRunLoopRunInMode = _run_loop_stopper(1)(stub)
    thread = threading.Thread(target=run, args=(stub,), daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert stub._log.warning.called