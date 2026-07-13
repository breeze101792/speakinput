"""Tests for the hotkey listener. Mocks pynput.keyboard.Listener."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_pynput(monkeypatch):
    """Stub out `pynput` (and its `keyboard` submodule) in `sys.modules` so
    the lazy `from pynput import keyboard` inside `speakinput.hotkey` picks
    up our fake. Both entries need swapping because `from pynput import
    keyboard` looks up `.keyboard` on the cached `pynput` module object."""
    import sys

    fake = MagicMock()
    fake_listener_instance = MagicMock()
    fake.Listener = MagicMock(return_value=fake_listener_instance)
    fake.Key = MagicMock()
    fake.Key.alt_r = "alt_r_key"
    fake.Key.ctrl_r = "ctrl_r_key"
    fake_pynput_mod = MagicMock()
    fake_pynput_mod.keyboard = fake
    monkeypatch.setitem(sys.modules, "pynput", fake_pynput_mod)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", fake)
    return fake, fake_listener_instance


def test_resolve_key_returns_pynput_key(fake_pynput):
    fake, _ = fake_pynput
    from speakinput.hotkey import resolve_key

    k = resolve_key("alt_r")
    assert k == fake.Key.alt_r


def test_resolve_key_rejects_unknown_name():
    """An unknown hotkey name must raise ValueError before any pynput access."""
    from speakinput.hotkey import resolve_key

    with pytest.raises(ValueError, match="unknown hotkey"):
        resolve_key("nonsense")


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
