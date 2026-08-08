"""Tests for the optional status-indicator implementations.

Covers `NullFeedback` (headless no-op), `StderrFeedback` (stderr
fallback with state validation), and `make_feedback`'s graceful
degradation when the rumps menu-bar backend is present or absent.
"""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from speakinput import feedback as fb


# --- NullFeedback ----------------------------------------------------------


def test_null_feedback_is_a_noop():
    nf = fb.NullFeedback()
    assert nf.set_state("listening") is None
    assert nf.start() is None
    assert nf.stop() is None


def test_null_feedback_accepts_any_state():
    """The null implementation is a pure no-op — it must not validate or
    choke on states a future build might introduce."""
    nf = fb.NullFeedback()
    nf.set_state("some-future-state")
    nf.set_state("error")
    nf.set_state("idle")


# --- StderrFeedback --------------------------------------------------------


def test_stderr_set_state_prints_announcement(capsys):
    fb_inst = fb.StderrFeedback()
    fb_inst.set_state("listening")
    fb_inst.set_state("error")
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith("[speakinput]")]
    assert "[speakinput] listening" in lines
    assert "[speakinput] error" in lines


def test_stderr_set_state_unknown_state_raises(capsys):
    fb_inst = fb.StderrFeedback()
    with pytest.raises(ValueError, match="unknown feedback state"):
        fb_inst.set_state("bogus")
    # Nothing must have been printed for a rejected state.
    assert capsys.readouterr().err == ""


def test_stderr_initial_state_is_idle():
    fb_inst = fb.StderrFeedback()
    assert fb_inst._state == "idle"


def test_stderr_start_stop_are_noops():
    fb_inst = fb.StderrFeedback()
    assert fb_inst.start() is None
    assert fb_inst.stop() is None


def test_stderr_accepts_all_valid_states():
    fb_inst = fb.StderrFeedback()
    for state in fb._STATES:  # idle, listening, processing, error
        fb_inst.set_state(state)
        assert fb_inst._state == state


# --- make_feedback ---------------------------------------------------------


def _inject_rumps_module(monkeypatch, rumps_cls):
    """Put a fake ``speakinput._feedback_rumps`` module into sys.modules
    so ``make_feedback`` resolves ``RumpsFeedback`` from it."""
    fake = types.ModuleType("speakinput._feedback_rumps")
    setattr(fake, "RumpsFeedback", rumps_cls)
    monkeypatch.setitem(sys.modules, "speakinput._feedback_rumps", fake)
    return fake


def test_make_feedback_falls_back_to_stderr_when_rumps_missing(monkeypatch):
    """Importing ``speakinput._feedback_rumps`` must fail (module absent)
    → we get the StderrFeedback fallback, never a crash."""
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):  # noqa: ANN001, ANN002
        if name == "speakinput._feedback_rumps":
            raise ImportError("No module named rumps")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert isinstance(fb.make_feedback(), fb.StderrFeedback)


def test_make_feedback_returns_rumps_when_available(monkeypatch):
    cls = type("RumpsFeedback", (), {})
    _inject_rumps_module(monkeypatch, cls)
    assert isinstance(fb.make_feedback(), cls)


def test_make_feedback_falls_back_when_rumps_constructor_raises(monkeypatch):
    class _Broken:  # noqa: D401
        def __init__(self):
            raise RuntimeError("no app bundle")

    _inject_rumps_module(monkeypatch, _Broken)
    # The constructor failure is swallowed; the app stays headless.
    assert isinstance(fb.make_feedback(), fb.StderrFeedback)


def test_make_feedback_rumps_constructor_is_used(monkeypatch):
    called: list[str] = []

    class _Fake:
        def __init__(self):
            called.append("new")

        def set_state(self, state):  # noqa: ARG002
            called.append("set_state")

    _inject_rumps_module(monkeypatch, _Fake)
    feedback = fb.make_feedback()
    feedback.set_state("processing")
    assert called == ["new", "set_state"]