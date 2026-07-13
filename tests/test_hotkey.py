"""Tests for the hotkey listener. Mocks pynput.keyboard.Listener."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_pynput(monkeypatch):
    fake = MagicMock()
    fake_listener_instance = MagicMock()
    fake.Listener = MagicMock(return_value=fake_listener_instance)
    fake.Key = MagicMock()
    fake.Key.alt_r = "alt_r_key"
    fake.Key.ctrl_r = "ctrl_r_key"
    monkeypatch.setattr("speakinput.hotkey.keyboard", fake, raising=False)
    return fake, fake_listener_instance


def test_resolve_key_validates_name(fake_pynput):
    from speakinput.hotkey import resolve_key

    with pytest.raises(ValueError, match="unknown hotkey"):
        resolve_key("nonsense")


def test_resolve_key_returns_pynput_key(fake_pynput):
    fake, _ = fake_pynput
    from speakinput.hotkey import resolve_key

    k = resolve_key("alt_r")
    assert k == fake.Key.alt_r


def test_listener_start_creates_and_starts_pynput_listener(fake_pynput):
    fake, instance = fake_pynput
    from speakinput.hotkey import HotkeyListener

    presses, releases = [], []
    h = HotkeyListener(
        key="alt_r_key",
        on_press=lambda: presses.append(1),
        on_release=lambda: releases.append(1),
    )
    h.start()
    assert h.is_running()
    fake.Listener.assert_called_once()
    instance.start.assert_called_once()


def test_listener_start_is_idempotent(fake_pynput):
    fake, instance = fake_pynput
    from speakinput.hotkey import HotkeyListener

    h = HotkeyListener(
        key="alt_r_key",
        on_press=lambda: None,
        on_release=lambda: None,
    )
    h.start()
    h.start()
    assert fake.Listener.call_count == 1


def test_listener_stop_tears_down(fake_pynput):
    fake, instance = fake_pynput
    from speakinput.hotkey import HotkeyListener

    h = HotkeyListener(
        key="alt_r_key",
        on_press=lambda: None,
        on_release=lambda: None,
    )
    h.start()
    h.stop()
    instance.stop.assert_called_once()
    assert not h.is_running()


def test_press_release_invoke_callbacks_in_order(fake_pynput):
    fake, instance = fake_pynput
    from speakinput.hotkey import HotkeyListener

    order: list[str] = []
    h = HotkeyListener(
        key="alt_r_key",
        on_press=lambda: order.append("press"),
        on_release=lambda: order.append("release"),
    )
    h.start()
    on_press = fake.Listener.call_args.kwargs["on_press"]
    on_release = fake.Listener.call_args.kwargs["on_release"]

    on_press("alt_r_key")
    on_release("alt_r_key")
    assert order == ["press", "release"]


def test_latch_guards_against_key_repeat(fake_pynput):
    """macOS sends multiple press events for one physical hold; only one callback should fire."""
    fake, instance = fake_pynput
    from speakinput.hotkey import HotkeyListener

    counter = {"n": 0}
    h = HotkeyListener(
        key="alt_r_key",
        on_press=lambda: counter.__setitem__("n", counter["n"] + 1),
        on_release=lambda: None,
    )
    h.start()
    on_press = fake.Listener.call_args.kwargs["on_press"]
    for _ in range(5):
        on_press("alt_r_key")
    assert counter["n"] == 1


def test_release_without_press_is_noop(fake_pynput):
    fake, instance = fake_pynput
    from speakinput.hotkey import HotkeyListener

    fired = {"n": 0}
    h = HotkeyListener(
        key="alt_r_key",
        on_press=lambda: None,
        on_release=lambda: fired.__setitem__("n", fired["n"] + 1),
    )
    h.start()
    on_release = fake.Listener.call_args.kwargs["on_release"]
    # Release without prior press — should NOT fire.
    on_release("alt_r_key")
    assert fired["n"] == 0


def test_press_other_key_is_ignored(fake_pynput):
    fake, instance = fake_pynput
    from speakinput.hotkey import HotkeyListener

    fired = {"n": 0}
    h = HotkeyListener(
        key="alt_r_key",
        on_press=lambda: fired.__setitem__("n", fired["n"] + 1),
        on_release=lambda: fired.__setitem__("n", fired["n"] + 1),
    )
    h.start()
    on_press = fake.Listener.call_args.kwargs["on_press"]
    on_release = fake.Listener.call_args.kwargs["on_release"]
    on_press("ctrl_r_key")
    on_release("ctrl_r_key")
    assert fired["n"] == 0


# --- pynput-missing paths --------------------------------------------------


def test_resolve_key_raises_when_pynput_missing(monkeypatch):
    from speakinput import hotkey as hk

    monkeypatch.setattr(hk, "keyboard", None, raising=False)
    with pytest.raises(RuntimeError, match="pynput"):
        hk.resolve_key("alt_r")


def test_listener_raises_when_pynput_missing(monkeypatch):
    from speakinput import hotkey as hk

    monkeypatch.setattr(hk, "keyboard", None, raising=False)
    with pytest.raises(RuntimeError, match="pynput"):
        hk.HotkeyListener(key="alt_r", on_press=lambda: None, on_release=lambda: None)
