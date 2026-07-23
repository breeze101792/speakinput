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


def test_find_keyboard_devices_handles_real_sway_wayland_hardware(monkeypatch):
    """Regression: the exact device list from the bug report.

    The reporter's sway/wayland box exposes:

      * /dev/input/event3  "MI Wireless Receiver"               — a
        Bluetooth receiver's HID keyboard interface (163 keys, has
        KEY_A/KEY_SPACE, has phys, NO EV_REP — Bluetooth keyboards
        commonly don't expose EV_REP through evdev).
      * /dev/input/event12 "SINO WEALTH Gaming KB "             — the
        real USB gaming keyboard the user actually types on (163 keys,
        has KEY_A/KEY_SPACE, has phys, NO EV_REP).
      * /dev/input/event15 "SINO WEALTH Gaming KB  Keyboard"    — a
        second event node on the same USB keyboard (108 keys).
      * /dev/input/event6  "MI Wireless Receiver Consumer Control"
        — media-key sibling of the Bluetooth receiver (158 keys, NO
        KEY_A/KEY_SPACE, has phys, NO EV_REP).
      * /dev/input/event14 "SINO WEALTH Gaming KB  Consumer Control"
        — media-key sibling of the gaming keyboard (141 keys, NO
        KEY_A/KEY_SPACE, has phys, NO EV_REP).
      * /dev/input/event265 "ydotoold virtual device"           — the
        ydotool daemon (540 keys, has KEY_A/KEY_SPACE, NO phys).

    The pre-fix `find_keyboard_device` ranked by (has_phys, has_rep,
    key_count). Every keyboard on this list reports EV_REP=False, so
    the tiebreaker collapsed to raw key count — and the algorithm
    picked /dev/input/event3 (the Bluetooth receiver) by accident.
    The user typed on the gaming keyboard (event12), so the hotkey
    silently never fired. This was reported as "key detection broken
    on Linux; works on Mac; sway/wayland".

    The fix has two parts, both exercised by this test:

      1. `find_keyboard_devices` returns ALL sanity-key-matching
         devices with a real `phys`, so the listener can attach to
         every real keyboard simultaneously. The user's hotkey fires
         no matter which one they type on.
      2. Consumer-control siblings (no KEY_A/KEY_SPACE) and virtual
         devices (no `phys` when a real phys exists) are filtered
         out, so we don't waste fds and kernel events on devices
         that will never produce the hotkey.

    This test reproduces the exact device set so a future refactor
    that reverts to single-device selection, drops the sanity-key
    check, or stops filtering virtual devices will fail here.
    """
    from speakinput import hotkey as hk

    # Two real keyboards. Both expose KEY_A (30) and KEY_SPACE (57),
    # both have a phys path, both LACK EV_REP (the bug-trigger: the
    # old EV_REP tiebreaker was dead on this hardware).
    fake_mi_receiver = MagicMock()
    fake_mi_receiver.capabilities.return_value = {1: set(range(163))}
    fake_mi_receiver.phys = "usb-0000:03:00.0-3.2/input0"
    fake_mi_receiver.path = "/dev/input/event3"
    fake_mi_receiver.name = "MI Wireless Receiver"

    fake_gaming_kb = MagicMock()
    fake_gaming_kb.capabilities.return_value = {1: set(range(163))}
    fake_gaming_kb.phys = "usb-0000:03:00.0-3.3.4/input0"
    fake_gaming_kb.path = "/dev/input/event12"
    fake_gaming_kb.name = "SINO WEALTH Gaming KB "

    # Second event node on the same gaming keyboard (fewer keys —
    # some keyboards split their key matrix across multiple event
    # nodes). Still has KEY_A/KEY_SPACE; still has phys.
    fake_gaming_kb_2 = MagicMock()
    fake_gaming_kb_2.capabilities.return_value = {1: set(range(108))}
    fake_gaming_kb_2.phys = "usb-0000:03:00.0-3.3.4/input1"
    fake_gaming_kb_2.path = "/dev/input/event15"
    fake_gaming_kb_2.name = "SINO WEALTH Gaming KB  Keyboard"

    # Consumer-control siblings. Both have 100+ KEY_* codes but
    # NEITHER has KEY_A or KEY_SPACE (keycodes 160..301 are all
    # media/brightness/etc. — well past KEY_SPACE=57 and KEY_Z=44).
    # The sanity-key check must reject these.
    fake_mi_cc = MagicMock()
    fake_mi_cc.capabilities.return_value = {1: set(range(160, 319))}
    fake_mi_cc.phys = "usb-0000:03:00.0-3.2/input1"
    fake_mi_cc.path = "/dev/input/event6"
    fake_mi_cc.name = "MI Wireless Receiver Consumer Control"

    fake_gaming_cc = MagicMock()
    fake_gaming_cc.capabilities.return_value = {1: set(range(160, 302))}
    fake_gaming_cc.phys = "usb-0000:03:00.0-3.3.4/input1"
    fake_gaming_cc.path = "/dev/input/event14"
    fake_gaming_cc.name = "SINO WEALTH Gaming KB  Consumer Control"

    # ydotoold virtual device. Advertises the full alphabet (so the
    # sanity-key check alone would keep it) but has NO phys path.
    # The virtual-device filter must drop it because real keyboards
    # are present in the pool.
    fake_ydotoold = MagicMock()
    fake_ydotoold.capabilities.return_value = {1: set(range(540))}
    fake_ydotoold.phys = ""
    fake_ydotoold.path = "/dev/input/event265"
    fake_ydotoold.name = "ydotoold virtual device"

    # Order the devices so the WRONG ones come first — this is the
    # worst case for any "pick the first" or "pick the highest score"
    # algorithm. The bug was that the pre-fix algorithm picked
    # event3 (MI Wireless Receiver) instead of the keyboard the user
    # was actually typing on.
    monkeypatch.setattr(
        hk.evdev,
        "list_devices",
        lambda: [
            "/dev/input/event6",    # MI consumer control
            "/dev/input/event14",   # gaming consumer control
            "/dev/input/event265",  # ydotoold
            "/dev/input/event3",    # MI Wireless Receiver (wrong pick)
            "/dev/input/event12",   # gaming keyboard (right pick)
            "/dev/input/event15",   # gaming keyboard event node 2
        ],
    )
    monkeypatch.setattr(
        hk.evdev,
        "InputDevice",
        lambda path: {
            "/dev/input/event3": fake_mi_receiver,
            "/dev/input/event6": fake_mi_cc,
            "/dev/input/event12": fake_gaming_kb,
            "/dev/input/event14": fake_gaming_cc,
            "/dev/input/event15": fake_gaming_kb_2,
            "/dev/input/event265": fake_ydotoold,
        }[path],
    )

    devs = hk.find_keyboard_devices()
    paths = {d.path for d in devs}

    # The three real keyboard event nodes are all returned — the
    # listener attaches to every one of them so the user's hotkey
    # fires regardless of which physical keyboard they type on.
    # This is the core fix: we no longer gamble on picking "the
    # right one" — we listen on all of them.
    assert paths == {
        "/dev/input/event3",
        "/dev/input/event12",
        "/dev/input/event15",
    }
    # The consumer-control siblings (no sanity keys) and the
    # ydotoold virtual device (no phys when real phys exists) are
    # filtered out and closed so we don't leak fds.
    fake_mi_cc.close.assert_called_once()
    fake_gaming_cc.close.assert_called_once()
    fake_ydotoold.close.assert_called_once()
    # Kept devices are left open for the listener to read from.
    fake_mi_receiver.close.assert_not_called()
    fake_gaming_kb.close.assert_not_called()
    fake_gaming_kb_2.close.assert_not_called()
    for d in devs:
        d.close()


def test_probe_evdev_available_returns_false_when_no_keyboard(monkeypatch):
    """probe_evdev_available is the gate App.run() uses to decide
    between the evdev and pynput backends on Linux. When no keyboard
    is found, it must return False (not raise) so App.run() can fall
    through to the pynput backend."""
    from speakinput import hotkey as hk

    fake_dev = MagicMock()
    fake_dev.capabilities.return_value = {1: set(range(5))}  # too few keys
    monkeypatch.setattr(hk.evdev, "list_devices", lambda: ["/dev/input/event0"])
    monkeypatch.setattr(hk.evdev, "InputDevice", lambda path: fake_dev)
    assert hk.probe_evdev_available() is False


def test_find_keyboard_devices_returns_all_real_keyboards(monkeypatch):
    """Multi-keyboard regression: when the box has two real keyboards
    (e.g. a Bluetooth receiver + a USB gaming keyboard), the listener
    must return BOTH so it can listen on each in parallel. The previous
    "pick one best" implementation attached to whichever happened to
    rank first and silently ignored the keyboard the user was actually
    typing on."""
    from speakinput import hotkey as hk

    # Two real keyboards — both have KEY_A and KEY_SPACE and a phys path.
    # set(range(163)) includes both KEY_A (30) and KEY_SPACE (57).
    fake_dev_a = MagicMock()
    fake_dev_a.capabilities.return_value = {1: set(range(163))}
    fake_dev_a.phys = "usb-0000:03:00.0-3.2/input0"
    fake_dev_a.path = "/dev/input/eventA"

    fake_dev_b = MagicMock()
    fake_dev_b.capabilities.return_value = {1: set(range(163))}
    fake_dev_b.phys = "usb-0000:03:00.0-3.3.4/input0"
    fake_dev_b.path = "/dev/input/eventB"

    # A ydotoold virtual device — advertises the full alphabet (so the
    # sanity-key check alone would keep it) but has no phys path. The
    # virtual-device filter must drop it once at least one real phys
    # path is in the pool.
    fake_dev_virtual = MagicMock()
    fake_dev_virtual.capabilities.return_value = {1: set(range(540))}
    fake_dev_virtual.phys = ""
    fake_dev_virtual.path = "/dev/input/eventV"

    # A consumer-control sibling — 141 KEY_* codes but no KEY_A or
    # KEY_SPACE (real consumer controls carry media/volume/brightness
    # keys, not letters). The sanity-key check rejects it before the
    # phys filter runs. Keycodes 160..300 are well past KEY_SPACE (57)
    # and KEY_Z (44), so the sanity check fails.
    fake_dev_cc = MagicMock()
    fake_dev_cc.capabilities.return_value = {1: set(range(160, 301))}
    fake_dev_cc.phys = "usb-0000:03:00.0-3.3.4/input1"
    fake_dev_cc.path = "/dev/input/eventC"

    monkeypatch.setattr(
        hk.evdev,
        "list_devices",
        lambda: [
            "/dev/input/eventV",  # virtual — listed first
            "/dev/input/eventC",  # consumer control
            "/dev/input/eventA",  # real keyboard A
            "/dev/input/eventB",  # real keyboard B
        ],
    )
    monkeypatch.setattr(
        hk.evdev,
        "InputDevice",
        lambda path: {
            "/dev/input/eventV": fake_dev_virtual,
            "/dev/input/eventC": fake_dev_cc,
            "/dev/input/eventA": fake_dev_a,
            "/dev/input/eventB": fake_dev_b,
        }[path],
    )

    devs = hk.find_keyboard_devices()
    paths = {d.path for d in devs}
    # Both real keyboards are returned; the virtual device and the
    # consumer-control sibling are filtered out.
    assert paths == {"/dev/input/eventA", "/dev/input/eventB"}
    # Filtered-out devices are closed so we don't leak fds.
    fake_dev_virtual.close.assert_called_once()
    fake_dev_cc.close.assert_called_once()
    # Kept devices are left open for the caller.
    fake_dev_a.close.assert_not_called()
    fake_dev_b.close.assert_not_called()
    for d in devs:
        d.close()


def test_find_keyboard_devices_keeps_virtual_when_only_virtual(monkeypatch):
    """If every candidate is virtual (no phys path), we still return
    the best one rather than failing — a headless CI box with only
    uinput devices should still get a listener instead of a hard
    failure. The phys filter only runs when at least one real phys
    path is in the pool."""
    from speakinput import hotkey as hk

    fake_dev_virtual = MagicMock()
    fake_dev_virtual.capabilities.return_value = {1: set(range(163))}
    fake_dev_virtual.phys = ""
    fake_dev_virtual.path = "/dev/input/event0"

    monkeypatch.setattr(hk.evdev, "list_devices", lambda: ["/dev/input/event0"])
    monkeypatch.setattr(hk.evdev, "InputDevice", lambda path: fake_dev_virtual)

    devs = hk.find_keyboard_devices()
    assert devs == [fake_dev_virtual]
    fake_dev_virtual.close.assert_not_called()
    fake_dev_virtual.close()


def test_evdev_listener_listens_on_multiple_devices(monkeypatch):
    """Regression: the listener must open every device returned by
    find_keyboard_devices and run a read_loop on each. The real-world
    motivation: a Linux box with two keyboards (e.g. a Bluetooth
    receiver + a USB gaming keyboard) needs the listener attached to
    both so the user's hotkey press is detected no matter which one
    they type on. The kernel routes each physical keypress to ONE
    event node, so only the device the user actually pressed emits
    events — but we don't know in advance which one that'll be, so we
    listen on all of them.

    This test verifies the listener correctly handles events coming
    from the SECOND device in the list (the one that would have been
    ignored by the old single-device implementation)."""
    from speakinput.hotkey import EvdevHotkeyListener

    dev1 = MagicMock()
    dev1.path = "/dev/input/eventA"
    dev2 = MagicMock()
    dev2.path = "/dev/input/eventB"

    from speakinput import hotkey as hk

    ecodes = hk._ecodes
    # dev1 emits nothing (it's not the keyboard the user typed on);
    # dev2 emits the press+release. The listener must catch dev2's
    # events even though dev1 was listed first.
    dev1.read_loop.side_effect = lambda: iter([])
    dev2.read_loop.side_effect = lambda: iter(
        [
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 1),
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 0),
        ]
    )

    order: list[str] = []
    h = EvdevHotkeyListener(
        keycode=ecodes.KEY_RIGHTCTRL,
        on_press=lambda: order.append("press"),
        on_release=lambda: order.append("release"),
        device=[dev1, dev2],
    )
    h.start()
    h.stop()

    # The listener caught the events from dev2, even though dev1 was
    # listed first and would have been the only one the old
    # single-device implementation opened.
    assert order == ["press", "release"]
    # Both devices are closed on stop, regardless of which emitted.
    dev1.close.assert_called_once()
    dev2.close.assert_called_once()


def test_evdev_listener_catches_hotkey_from_wrong_picked_keyboard(monkeypatch):
    """End-to-end regression for the sway/wayland bug report.

    Reproduces the exact failure: the user has two real keyboards
    plugged in (a Bluetooth receiver + a USB gaming keyboard). The
    pre-fix algorithm picked the Bluetooth receiver (event3) by
    accident; the user typed on the gaming keyboard (event12); the
    hotkey silently never fired.

    This test wires the listener with both keyboards (mimicking what
    `find_keyboard_devices` now returns) and verifies that a
    KEY_RIGHTCTRL press on the GAMING keyboard (the one the old
    algorithm would have ignored) is detected and fires on_press /
    on_release correctly.

    A future refactor that reverts to single-device selection will
    pick event3 (first-listed, equal score) and this test will fail
    because event3 emits nothing — the listener's read_loop on it
    would block forever and the press on event12 would be missed.
    """
    from speakinput.hotkey import EvdevHotkeyListener

    # The Bluetooth receiver — the one the old algorithm WRONGLY picked.
    # It emits nothing because the user isn't typing on it.
    fake_mi_receiver = MagicMock()
    fake_mi_receiver.path = "/dev/input/event3"
    fake_mi_receiver.read_loop.side_effect = lambda: iter([])

    # The gaming keyboard — the one the user is ACTUALLY typing on.
    # The old algorithm ignored this device entirely.
    fake_gaming_kb = MagicMock()
    fake_gaming_kb.path = "/dev/input/event12"
    from speakinput import hotkey as hk

    ecodes = hk._ecodes
    fake_gaming_kb.read_loop.side_effect = lambda: iter(
        [
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 1),  # user presses
            _make_evdev_event(ecodes.KEY_RIGHTCTRL, 0),  # user releases
        ]
    )

    order: list[str] = []
    # The listener is constructed with BOTH devices, matching what
    # find_keyboard_devices returns on this hardware.
    h = EvdevHotkeyListener(
        keycode=ecodes.KEY_RIGHTCTRL,
        on_press=lambda: order.append("press"),
        on_release=lambda: order.append("release"),
        device=[fake_mi_receiver, fake_gaming_kb],
    )
    h.start()
    h.stop()

    # The press on the gaming keyboard WAS detected — the bug
    # symptom was that this list stayed empty.
    assert order == ["press", "release"]
    # Both devices are closed on stop.
    fake_mi_receiver.close.assert_called_once()
    fake_gaming_kb.close.assert_called_once()


def test_probe_evdev_available_closes_device_when_available(monkeypatch):
    """When evdev CAN find a keyboard, the probe opens it (to validate)
    then immediately closes it so the real listener can re-open it
    later. The fd-leak guard is the whole point of this helper."""
    from speakinput import hotkey as hk

    fake_dev_kbd = MagicMock()
    fake_dev_kbd.capabilities.return_value = {
        1: set(range(163)),
        2: set(),
        4: set(),
        20: set(range(1)),
    }
    fake_dev_kbd.phys = "usb-0000:03:00.0-3.4.4/input0"
    fake_dev_kbd.path = "/dev/input/event0"
    monkeypatch.setattr(hk.evdev, "list_devices", lambda: ["/dev/input/event0"])
    monkeypatch.setattr(hk.evdev, "InputDevice", lambda path: fake_dev_kbd)
    assert hk.probe_evdev_available() is True
    # The probe opens the device to validate it, then closes it. If
    # close() wasn't called, an fd would leak on every app start.
    fake_dev_kbd.close.assert_called_once()


def test_probe_evdev_available_returns_false_when_evdev_missing(monkeypatch):
    from speakinput import hotkey as hk

    monkeypatch.setattr(hk, "evdev", None, raising=False)
    assert hk.probe_evdev_available() is False


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
