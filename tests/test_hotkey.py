"""Tests for the hotkey listener. Mocks pynput.keyboard.Listener and
the Linux evdev InputDevice for the Wayland backend."""

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


# --- evdev backend ---------------------------------------------------------
#
# The Wayland-side listener is tested with a fake InputDevice that
# yields canned InputEvent objects. This mirrors how the pynput tests
# inject a fake keyboard.Listener and lets us cover the press/release/
# repeat semantics without a real /dev/input/eventN.


def _make_evdev_event(code, value, type_=1, sec=0, usec=0):  # noqa: ANN001
    """Build a real evdev `InputEvent`.

    `type_=1` is EV_KEY (the only type the listener cares about).
    `value` follows the kernel's convention: 1=press, 0=release, 2=repeat.
    """
    from evdev.events import InputEvent

    return InputEvent(sec, usec, type_, code, value)


def _make_fake_keyboard():
    """Return a MagicMock standing in for an opened evdev.InputDevice.

    The listener calls `read_loop()` (a fresh iterator each invocation)
    and `close()`. Tests attach a generator via `side_effect = lambda:
    iter([...])` so each `read_loop()` call yields the canned event
    list. Tests then call `close()` (or stop the listener) to make the
    loop exit. The default fixture installs a hanging generator that
    blocks until `close()` is called — useful for tests that need the
    listener to stay "running" while they assert.
    """
    instance = MagicMock()
    instance.path = "/dev/input/eventFAKE"
    return instance


@pytest.fixture
def fake_evdev_keyboard():
    """Provides a fake InputDevice to inject into EvdevHotkeyListener.

    Returns (instance, ecodes_module) so tests can build events with
    real keycodes (e.g. ecodes.KEY_RIGHTCTRL) and inject them.
    """
    from speakinput import hotkey as hk

    instance = _make_fake_keyboard()
    # Default: a long-running generator that yields nothing until the
    # test calls `instance.close()`. The listener's `read_loop()` is
    # replaced per-test as needed.
    def _hanging_loop():
        # Block until close() is called by the listener's stop().
        # We use a threading.Event so the test thread can wait for
        # the listener thread to actually start, then for stop() to
        # close the device.
        from threading import Event
        ev = Event()
        instance._stop_event = ev
        ev.wait()
        return
        yield  # pragma: no cover - generator never returns

    instance.read_loop.side_effect = _hanging_loop
    return instance, hk._ecodes


def test_resolve_evdev_key_maps_every_valid_name(fake_evdev_keyboard):
    from speakinput.hotkey import resolve_evdev_key, _EVDEV_KEY_MAP

    # Every VALID_HOTKEYS entry must map to a sensible Linux keycode.
    from speakinput.config import VALID_HOTKEYS

    for name in VALID_HOTKEYS:
        code = resolve_evdev_key(name)
        assert isinstance(code, int)
        assert code > 0
        assert code == _EVDEV_KEY_MAP[name]


def test_resolve_evdev_key_rejects_unknown_name(fake_evdev_keyboard):
    from speakinput.hotkey import resolve_evdev_key

    with pytest.raises(ValueError, match="unknown hotkey"):
        resolve_evdev_key("nonsense")


def test_resolve_evdev_key_raises_when_evdev_missing(monkeypatch):
    from speakinput import hotkey as hk

    monkeypatch.setattr(hk, "evdev", None, raising=False)
    monkeypatch.setattr(hk, "_ecodes", None, raising=False)
    with pytest.raises(RuntimeError, match="evdev"):
        hk.resolve_evdev_key("alt_r")


def test_find_keyboard_device_picks_physical_keyboard_with_ev_rep(monkeypatch):
    """Auto-detect should pick a physical keyboard (non-empty `phys`,
    `EV_REP` capability, lots of keycodes) over a virtual device or a
    consumer-control sibling that happens to come first in the list."""
    from speakinput import hotkey as hk

    # Virtual device that LOOKS like a keyboard (lots of keys) — should be
    # demoted because it has no `phys` path.
    fake_dev_virtual = MagicMock()
    fake_dev_virtual.capabilities.return_value = {1: set(range(540))}  # 540 keys
    fake_dev_virtual.phys = ""  # virtual devices report no phys path
    fake_dev_virtual.path = "/dev/input/event0"

    # Consumer Control sibling of the real keyboard — has EV_KEY but no
    # EV_REP (real keyboards advertise auto-repeat, consumer controls don't).
    fake_dev_cc = MagicMock()
    fake_dev_cc.capabilities.return_value = {1: set(range(141)), 4: set()}  # EV_MSC=4
    fake_dev_cc.phys = "usb-0000:03:00.0-3.4.4/input1"
    fake_dev_cc.path = "/dev/input/event1"

    # The real physical keyboard — phys set, EV_REP advertised, lots of keys.
    fake_dev_kbd = MagicMock()
    fake_dev_kbd.capabilities.return_value = {
        1: set(range(163)),   # EV_KEY
        2: set(),              # EV_REL
        4: set(),              # EV_MSC
        20: set(range(1)),     # EV_REP=20 — the magic bit
    }
    fake_dev_kbd.phys = "usb-0000:03:00.0-3.4.4/input0"
    fake_dev_kbd.path = "/dev/input/event2"

    monkeypatch.setattr(
        hk.evdev,
        "list_devices",
        lambda: [
            "/dev/input/event0",  # virtual — listed first
            "/dev/input/event1",  # consumer control — listed second
            "/dev/input/event2",  # real keyboard — listed third
        ],
    )
    monkeypatch.setattr(
        hk.evdev,
        "InputDevice",
        lambda path: {
            "/dev/input/event0": fake_dev_virtual,
            "/dev/input/event1": fake_dev_cc,
            "/dev/input/event2": fake_dev_kbd,
        }[path],
    )

    dev = hk.find_keyboard_device()
    # Even though the virtual device is listed first AND has more keys,
    # the real physical keyboard wins because it has both `phys` and
    # `EV_REP` set.
    assert dev is fake_dev_kbd
    # All non-chosen candidates are closed after the scan, so we don't
    # leak file descriptors. The chosen device is left open for the caller.
    fake_dev_virtual.close.assert_called_once()
    fake_dev_cc.close.assert_called_once()
    fake_dev_kbd.close.assert_not_called()


def test_find_keyboard_device_raises_when_none_match(monkeypatch):
    from speakinput import hotkey as hk

    fake_dev = MagicMock()
    fake_dev.capabilities.return_value = {1: set(range(5))}
    monkeypatch.setattr(hk.evdev, "list_devices", lambda: ["/dev/input/event0"])
    monkeypatch.setattr(hk.evdev, "InputDevice", lambda path: fake_dev)

    with pytest.raises(hk.HotkeyError, match="no keyboard device found"):
        hk.find_keyboard_device()


def test_find_keyboard_device_raises_when_evdev_missing(monkeypatch):
    from speakinput import hotkey as hk

    monkeypatch.setattr(hk, "evdev", None, raising=False)
    with pytest.raises(hk.HotkeyError, match="evdev is not installed"):
        hk.find_keyboard_device()


def test_evdev_listener_press_release_in_order(fake_evdev_keyboard):
    from speakinput.hotkey import EvdevHotkeyListener

    instance, ecodes = fake_evdev_keyboard
    instance.read_loop.side_effect = lambda: iter(
        [
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 1),  # press
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 0),  # release
        ]
    )
    order: list[str] = []
    h = EvdevHotkeyListener(
        keycode=ecodes.KEY_RIGHTCTRL,
        on_press=lambda: order.append("press"),
        on_release=lambda: order.append("release"),
        device=instance,
    )
    h.start()
    h.stop()  # closes the device, unblocking the loop
    assert order == ["press", "release"]


def test_evdev_listener_latch_suppresses_repeat(fake_evdev_keyboard):
    """Linux evdev sends `value=2` (KEY_REPEATED) for held keys. The latch
    must suppress these so on_press fires exactly once per physical hold."""
    from speakinput.hotkey import EvdevHotkeyListener

    instance, ecodes = fake_evdev_keyboard
    instance.read_loop.side_effect = lambda: iter(
        [
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 1),  # press
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 2),  # repeat (suppress)
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 2),  # repeat (suppress)
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 0),  # release
        ]
    )
    presses: list[int] = []
    h = EvdevHotkeyListener(
        keycode=ecodes.KEY_RIGHTCTRL,
        on_press=lambda: presses.append(1),
        on_release=lambda: None,
        device=instance,
    )
    h.start()
    h.stop()
    assert presses == [1]


def test_evdev_listener_ignores_other_keys(fake_evdev_keyboard):
    from speakinput.hotkey import EvdevHotkeyListener

    instance, ecodes = fake_evdev_keyboard
    instance.read_loop.side_effect = lambda: iter(
        [
            _make_evdev_event(ecodes.KEY_A, 1),       # unrelated press
            _make_evdev_event(ecodes.KEY_A, 0),       # unrelated release
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 1),  # our key
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 0),
        ]
    )
    order: list[str] = []
    h = EvdevHotkeyListener(
        keycode=ecodes.KEY_RIGHTCTRL,
        on_press=lambda: order.append("press"),
        on_release=lambda: order.append("release"),
        device=instance,
    )
    h.start()
    h.stop()
    assert order == ["press", "release"]


def test_evdev_listener_release_without_press_is_noop(fake_evdev_keyboard):
    from speakinput.hotkey import EvdevHotkeyListener

    instance, ecodes = fake_evdev_keyboard
    instance.read_loop.side_effect = lambda: iter(
        [
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 0),  # release w/o prior press
        ]
    )
    fired: list[int] = []
    h = EvdevHotkeyListener(
        keycode=ecodes.KEY_RIGHTCTRL,
        on_press=lambda: fired.append(1),
        on_release=lambda: fired.append(1),
        device=instance,
    )
    h.start()
    h.stop()
    assert fired == []


def test_evdev_listener_ignores_non_key_events(fake_evdev_keyboard):
    """EV_REL / EV_ABS / EV_MSC events should be ignored."""
    from speakinput import hotkey as hk
    from speakinput.hotkey import EvdevHotkeyListener

    instance, ecodes = fake_evdev_keyboard
    instance.read_loop.side_effect = lambda: iter(
        [
            _make_evdev_event(0, 5, type_=hk._ecodes.EV_REL),    # mouse motion
            _make_evdev_event(0, 0, type_=hk._ecodes.EV_MSC),    # misc scan code
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 1),
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 0),
        ]
    )
    order: list[str] = []
    h = EvdevHotkeyListener(
        keycode=ecodes.KEY_RIGHTCTRL,
        on_press=lambda: order.append("press"),
        on_release=lambda: order.append("release"),
        device=instance,
    )
    h.start()
    h.stop()
    assert order == ["press", "release"]


def test_evdev_listener_start_is_idempotent(fake_evdev_keyboard):
    from speakinput.hotkey import EvdevHotkeyListener

    instance, ecodes = fake_evdev_keyboard
    # The fixture default is a hanging read_loop(); good for this test
    # because it keeps the listener "running" until we stop it.
    h = EvdevHotkeyListener(
        keycode=ecodes.KEY_RIGHTCTRL,
        on_press=lambda: None,
        on_release=lambda: None,
        device=instance,
    )
    h.start()
    h.start()  # second start should be a noop
    assert h.is_running()
    h.stop()


def test_evdev_listener_stop_tears_down(fake_evdev_keyboard):
    from speakinput.hotkey import EvdevHotkeyListener

    instance, ecodes = fake_evdev_keyboard
    h = EvdevHotkeyListener(
        keycode=ecodes.KEY_RIGHTCTRL,
        on_press=lambda: None,
        on_release=lambda: None,
        device=instance,
    )
    h.start()
    h.stop()
    instance.close.assert_called_once()
    assert not h.is_running()


def test_evdev_listener_raises_when_evdev_missing(monkeypatch):
    from speakinput import hotkey as hk

    monkeypatch.setattr(hk, "evdev", None, raising=False)
    with pytest.raises(RuntimeError, match="evdev"):
        hk.EvdevHotkeyListener(
            keycode=100,
            on_press=lambda: None,
            on_release=lambda: None,
        )
