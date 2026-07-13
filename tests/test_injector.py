"""Tests for the injector. Mocks pynput and pyperclip so the suite runs headless."""

import sys
import threading
import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_modules(monkeypatch):
    """Patch `pynput.keyboard` in `sys.modules` and `pyperclip` so the
    injector picks up a stub Controller. The injector lazy-imports pynput
    via `importlib.import_module("pynput.keyboard")` so swapping the
    module entry is enough — there are no module-level references in
    `speakinput.injector` to chase down."""
    from speakinput import injector as inj_mod

    fake_keyboard = MagicMock()
    fake_keyboard.Controller = MagicMock()
    fake_keyboard.Key = MagicMock()
    fake_keyboard.Key.cmd = "cmd"
    fake_keyboard.Key.ctrl = "ctrl"

    fake_pyperclip = MagicMock()
    fake_pyperclip.paste = MagicMock(return_value="prior-clipboard")
    fake_pyperclip.copy = MagicMock()

    monkeypatch.setitem(sys.modules, "pynput.keyboard", fake_keyboard)
    monkeypatch.setattr(inj_mod, "pyperclip", fake_pyperclip, raising=False)
    return fake_keyboard, fake_pyperclip


def test_is_ascii_safe_accepts_letters_digits_punctuation():
    from speakinput.injector import is_ascii_safe

    assert is_ascii_safe("hello world")
    assert is_ascii_safe("Hello, World! 123")
    assert is_ascii_safe("  leading and trailing  ")


def test_is_ascii_safe_rejects_non_ascii():
    from speakinput.injector import is_ascii_safe

    assert not is_ascii_safe("héllo")  # accented
    assert not is_ascii_safe("你好")  # CJK
    assert not is_ascii_safe("hello—world")  # em dash
    assert not is_ascii_safe("hi 👋")


def test_inject_empty_string_does_nothing(fake_modules):
    keyboard_mod, _ = fake_modules
    from speakinput.injector import TypingInjector

    inj = TypingInjector()
    inj.inject("")
    controller = keyboard_mod.Controller.return_value
    controller.type.assert_not_called()


def test_inject_ascii_uses_controller(fake_modules):
    keyboard_mod, _ = fake_modules
    from speakinput.injector import TypingInjector

    inj = TypingInjector(trailing_space=False)
    inj.inject("hello world")
    controller = keyboard_mod.Controller.return_value
    controller.type.assert_called_once_with("hello world")


def test_inject_ascii_appends_trailing_space_by_default(fake_modules):
    keyboard_mod, _ = fake_modules
    from speakinput.injector import TypingInjector

    inj = TypingInjector()  # trailing_space defaults to True
    inj.inject("hello world")
    controller = keyboard_mod.Controller.return_value
    controller.type.assert_called_once_with("hello world ")


def test_inject_ascii_skips_trailing_space_when_disabled(fake_modules):
    keyboard_mod, _ = fake_modules
    from speakinput.injector import TypingInjector

    inj = TypingInjector(trailing_space=False)
    inj.inject("hello world")
    controller = keyboard_mod.Controller.return_value
    controller.type.assert_called_once_with("hello world")


def test_inject_unicode_appends_trailing_space_by_default(fake_modules, monkeypatch):
    keyboard_mod, pyperclip_mod = fake_modules
    from speakinput.injector import TypingInjector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux", raising=False)

    inj = TypingInjector(restore_clipboard_ms=0)  # trailing_space defaults True
    inj.inject("你好")
    # Clipboard should receive the text + space.
    pyperclip_mod.copy.assert_any_call("你好 ")


def test_inject_unicode_skips_trailing_space_when_disabled(fake_modules, monkeypatch):
    keyboard_mod, pyperclip_mod = fake_modules
    from speakinput.injector import TypingInjector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux", raising=False)

    inj = TypingInjector(restore_clipboard_ms=0, trailing_space=False)
    inj.inject("你好")
    pyperclip_mod.copy.assert_any_call("你好")


def test_inject_empty_string_does_not_add_space(fake_modules):
    """Defensive: empty text short-circuits before trailing-space logic."""
    keyboard_mod, _ = fake_modules
    from speakinput.injector import TypingInjector

    inj = TypingInjector()
    inj.inject("")
    controller = keyboard_mod.Controller.return_value
    controller.type.assert_not_called()


def test_inject_unicode_uses_clipboard_path(fake_modules, monkeypatch):
    keyboard_mod, pyperclip_mod = fake_modules
    from speakinput.injector import TypingInjector

    # Simulate non-mac so we exercise the pyperclip path deterministically.
    monkeypatch.setattr("speakinput.injector.sys.platform", "linux", raising=False)

    inj = TypingInjector(restore_clipboard_ms=0, trailing_space=False)  # disable restore timer
    inj.inject("你好")
    controller = keyboard_mod.Controller.return_value

    # Clipboard write happened.
    pyperclip_mod.copy.assert_any_call("你好")
    # Controller pressed Ctrl (linux override).
    controller.pressed.assert_called_once_with(keyboard_mod.Key.ctrl)
    controller.tap.assert_called_once_with("v")
    # Did NOT call .type() (ASCII path).
    controller.type.assert_not_called()


def test_inject_unicode_restores_prior_clipboard(fake_modules, monkeypatch):
    """End-to-end: write prior, paste new, restore prior after delay."""
    keyboard_mod, pyperclip_mod = fake_modules
    from speakinput.injector import TypingInjector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux", raising=False)

    inj = TypingInjector(restore_clipboard_ms=10, trailing_space=False)
    inj.inject("αβγ")
    # The restore timer is fire-and-forget; wait for it.
    for _ in range(50):
        if pyperclip_mod.copy.call_count >= 2:
            break
        time.sleep(0.01)
    # First call: write new text. Second call: restore prior.
    assert pyperclip_mod.copy.call_args_list[0] == (("αβγ",),)
    assert pyperclip_mod.copy.call_args_list[1] == (("prior-clipboard",),)


def test_inject_concurrent_unicode_calls_are_serialized(fake_modules, monkeypatch):
    """Two overlapping Unicode injections should not race the clipboard."""
    keyboard_mod, _ = fake_modules
    from speakinput.injector import TypingInjector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux", raising=False)

    inj = TypingInjector(restore_clipboard_ms=0)

    def fire():
        inj.inject("α")  # non-ASCII forces the clipboard path

    threads = [threading.Thread(target=fire) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Controller.tap("v") called once per injection; no exceptions raised.
    assert keyboard_mod.Controller.return_value.tap.call_count == 5
