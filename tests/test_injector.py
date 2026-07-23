"""Tests for the injector. Mocks pynput and pyperclip so the suite runs headless."""

import subprocess
import threading
import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_modules(monkeypatch):
    """Patch the module-level `Controller`/`Key` and `pyperclip` references
    inside the injector module. Replacing only `sys.modules` is not enough
    because the import was already captured at module load time.
    """
    from speakinput import injector as inj_mod

    fake_keyboard = MagicMock()
    fake_keyboard.Controller = MagicMock()
    fake_keyboard.Key = MagicMock()
    fake_keyboard.Key.cmd = "cmd"
    fake_keyboard.Key.ctrl = "ctrl"

    fake_pyperclip = MagicMock()
    fake_pyperclip.paste = MagicMock(return_value="prior-clipboard")
    fake_pyperclip.copy = MagicMock()

    monkeypatch.setattr(inj_mod, "Controller", fake_keyboard.Controller, raising=False)
    monkeypatch.setattr(inj_mod, "Key", fake_keyboard.Key, raising=False)
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


# --- WtypeInjector / YdotoolInjector / select_injector ---------------------
#
# These tests patch the binary-discovery path (shutil.which / os.environ /
# os.path.exists) and subprocess.run so the tests run without a real wtype,
# ydotool, wl-copy, or running compositor. The ASCII / Unicode split logic
# is shared with TypingInjector; here we only verify the backend-specific
# subprocess commands and the auto-select dispatch.


@pytest.fixture
def _patch_subprocess(monkeypatch):
    """Replace subprocess.run with a recording MagicMock."""
    fake_run = MagicMock()
    monkeypatch.setattr("speakinput.injector.subprocess.run", fake_run)
    return fake_run


@pytest.fixture
def _wtype_available(monkeypatch):
    """Pretend wtype is on PATH and ydotool is not."""
    which = MagicMock(side_effect=lambda b: f"/usr/bin/{b}" if b == "wtype" else None)
    monkeypatch.setattr("speakinput.injector.shutil.which", which)
    return which


@pytest.fixture
def _ydotool_available(monkeypatch):
    """Pretend ydotool is on PATH with a reachable socket."""
    which = MagicMock(side_effect=lambda b: f"/usr/bin/{b}" if b == "ydotool" else None)
    monkeypatch.setattr("speakinput.injector.shutil.which", which)
    monkeypatch.setenv("YDOTOOL_SOCKET", "/tmp/ydotool.sock")
    return which


def test_wtype_injector_ascii_invokes_wtype(_wtype_available, _patch_subprocess):
    from speakinput.injector import WtypeInjector

    inj = WtypeInjector(trailing_space=False)
    inj.inject("hello world")
    _patch_subprocess.assert_called_once()
    args = _patch_subprocess.call_args.args[0]
    assert args[0] == "wtype"
    # The `--` separates options from positional text; payload is the last arg.
    assert args[-1] == "hello world"


def test_wtype_injector_ascii_appends_trailing_space_by_default(
    _wtype_available, _patch_subprocess
):
    from speakinput.injector import WtypeInjector

    inj = WtypeInjector()
    inj.inject("hi")
    args = _patch_subprocess.call_args.args[0]
    assert args[-1] == "hi "


def test_wtype_injector_unicode_uses_clipboard_paste(
    _wtype_available, _patch_subprocess, fake_modules, monkeypatch
):
    """Non-ASCII text writes to the clipboard, sends Ctrl+V via wtype, then
    schedules a restore of the prior clipboard contents."""
    from speakinput.injector import WtypeInjector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux")
    _, pyperclip_mod = fake_modules
    inj = WtypeInjector(trailing_space=False)
    inj.inject("你好")
    # Clipboard got the text.
    pyperclip_mod.copy.assert_any_call("你好")
    # wtype was called with a Ctrl+V sequence.
    paste_call = [
        c for c in _patch_subprocess.call_args_list
        if c.args and c.args[0] and c.args[0][0] == "wtype"
        and "-k" in c.args[0]
    ]
    assert paste_call, "expected a wtype paste invocation"
    args = paste_call[0].args[0]
    assert args == ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]


def test_wtype_injector_raises_when_wtype_missing(monkeypatch):
    from speakinput.injector import WtypeInjector

    monkeypatch.setattr("speakinput.injector.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="wtype is not installed"):
        WtypeInjector()


def test_ydotool_injector_ascii_uses_socket_env(_ydotool_available, _patch_subprocess):
    """ASCII path passes YDOTOOL_SOCKET through to the subprocess env."""
    from speakinput.injector import YdotoolInjector

    inj = YdotoolInjector(trailing_space=False)
    inj.inject("hello")
    _patch_subprocess.assert_called_once()
    args = _patch_subprocess.call_args.args[0]
    assert args[0] == "ydotool"
    assert args[1] == "type"
    assert args[-1] == "hello"
    env = _patch_subprocess.call_args.kwargs.get("env") or {}
    assert env.get("YDOTOOL_SOCKET") == "/tmp/ydotool.sock"


def test_ydotool_injector_unicode_sends_ctrl_v_keycodes(
    _ydotool_available, _patch_subprocess, fake_modules, monkeypatch
):
    """Unicode path uses `ydotool key` with Linux keycodes for left-Ctrl+V."""
    from speakinput.injector import YdotoolInjector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux")
    _, pyperclip_mod = fake_modules
    inj = YdotoolInjector(trailing_space=False)
    inj.inject("αβγ")
    pyperclip_mod.copy.assert_any_call("αβγ")
    paste_call = [
        c for c in _patch_subprocess.call_args_list
        if c.args and c.args[0] and c.args[0][0] == "ydotool"
        and c.args[0][1] == "key"
    ]
    assert paste_call, "expected a ydotool key invocation"
    args = paste_call[0].args[0]
    # KEY_LEFTCTRL=29, KEY_V=47: down, down, up, up.
    assert args == ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]


def test_ydotool_injector_raises_when_binary_missing(monkeypatch):
    from speakinput.injector import YdotoolInjector

    monkeypatch.setattr("speakinput.injector.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="ydotool is not installed"):
        YdotoolInjector()


def test_ydotool_injector_raises_when_socket_missing(monkeypatch):
    """ydotool is installed but ydotoold isn't running → no socket."""
    from speakinput.injector import YdotoolInjector

    # Pretend ydotool is on PATH but no YDOTOOL_SOCKET env and no
    # /run/user/<uid>/.ydotool_socket.
    monkeypatch.setattr(
        "speakinput.injector.shutil.which",
        lambda b: "/usr/bin/ydotool" if b == "ydotool" else None,
    )
    monkeypatch.delenv("YDOTOOL_SOCKET", raising=False)
    monkeypatch.setattr("speakinput.injector.os.getuid", lambda: 99999)
    monkeypatch.setattr("speakinput.injector.os.path.exists", lambda _: False)
    with pytest.raises(RuntimeError, match="YDOTOOL_SOCKET"):
        YdotoolInjector()


# --- select_injector dispatch --------------------------------------------


def test_select_injector_picks_pynput_on_macos(monkeypatch, fake_modules):
    from speakinput.config import InjectConfig
    from speakinput.injector import TypingInjector, select_injector

    monkeypatch.setattr("speakinput.injector.sys.platform", "darwin")
    inj = select_injector(InjectConfig())
    assert isinstance(inj, TypingInjector)


def test_select_injector_picks_pynput_on_windows(monkeypatch, fake_modules):
    from speakinput.config import InjectConfig
    from speakinput.injector import TypingInjector, select_injector

    monkeypatch.setattr("speakinput.injector.sys.platform", "win32")
    inj = select_injector(InjectConfig())
    assert isinstance(inj, TypingInjector)


def test_select_injector_picks_wtype_on_wayland(
    monkeypatch, _wtype_available, fake_modules
):
    from speakinput.config import InjectConfig
    from speakinput.injector import WtypeInjector, select_injector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    inj = select_injector(InjectConfig())
    assert isinstance(inj, WtypeInjector)


def test_select_injector_falls_back_to_ydotool_on_wayland(
    monkeypatch, _ydotool_available, fake_modules
):
    from speakinput.config import InjectConfig
    from speakinput.injector import YdotoolInjector, select_injector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    # wtype is NOT in the mock; only ydotool is.
    inj = select_injector(InjectConfig())
    assert isinstance(inj, YdotoolInjector)


def test_select_injector_falls_back_to_pynput_when_neither(
    monkeypatch, fake_modules
):
    """Pure Wayland with no wtype / no ydotool → pynput last-resort.

    pynput only works through XWayland; the user will get a clear
    error at injection time on a true pure-Wayland box, but the
    construction-time path must not crash."""
    from speakinput.config import InjectConfig
    from speakinput.injector import TypingInjector, select_injector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setattr("speakinput.injector.shutil.which", lambda _: None)
    inj = select_injector(InjectConfig())
    assert isinstance(inj, TypingInjector)


def test_select_injector_picks_pynput_on_linux_x11(monkeypatch, fake_modules):
    from speakinput.config import InjectConfig
    from speakinput.injector import TypingInjector, select_injector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux")
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    # Even when wtype is installed, X11 should not pick it (the user
    # explicitly chose X11 by being on an X session).
    monkeypatch.setattr(
        "speakinput.injector.shutil.which",
        lambda b: f"/usr/bin/{b}" if b in ("wtype", "ydotool") else None,
    )
    inj = select_injector(InjectConfig())
    assert isinstance(inj, TypingInjector)


def test_select_injector_explicit_backend_overrides_platform(
    monkeypatch, _ydotool_available, fake_modules
):
    """`backend = "ydotool"` on a macOS box still picks YdotoolInjector."""
    from speakinput.config import InjectConfig
    from speakinput.injector import YdotoolInjector, select_injector

    monkeypatch.setattr("speakinput.injector.sys.platform", "darwin")
    inj = select_injector(InjectConfig(backend="ydotool"))
    assert isinstance(inj, YdotoolInjector)


def test_select_injector_unknown_backend_raises(fake_modules):
    from speakinput.config import InjectConfig
    from speakinput.injector import select_injector

    with pytest.raises(ValueError, match="unknown inject.backend"):
        select_injector(InjectConfig(backend="bogus"))


def test_select_injector_explicit_wtype_on_non_wlroots_raises(monkeypatch, fake_modules):
    """User force-sets `backend = "wtype"` on macOS → WtypeInjector is
    constructed and will fail at injection time. The factory just
    constructs what was asked for."""
    from speakinput.config import InjectConfig
    from speakinput.injector import WtypeInjector, select_injector

    monkeypatch.setattr("speakinput.injector.sys.platform", "darwin")
    monkeypatch.setattr(
        "speakinput.injector.shutil.which",
        lambda b: "/usr/bin/wtype" if b == "wtype" else None,
    )
    inj = select_injector(InjectConfig(backend="wtype"))
    assert isinstance(inj, WtypeInjector)


# --- stability: clipboard restore, subprocess timeouts, lock-around-inject ---


def test_clipboard_restore_timer_is_daemon(monkeypatch):
    """A pending clipboard restore must not block process exit.

    The `threading.Timer` used to schedule a clipboard restore is
    created with `daemon=False` by default, which lets a user with
    `restore_clipboard_ms = 1000` (no upper cap on the config)
    block shutdown for a full second. The fix sets `daemon=True`
    on the timer; the kernel reaps it on exit and the user's next
    paste sees the last injected text.
    """
    from speakinput import injector as inj_mod

    fake = MagicMock()
    fake.daemon = None  # capture whatever the production code sets
    monkeypatch.setattr(inj_mod.threading, "Timer", lambda *a, **kw: fake)
    monkeypatch.setattr(inj_mod, "_pbcopy_restore", lambda *_a, **_kw: None)
    inj_mod._schedule_clipboard_restore("x", 50)
    # Production must have set daemon=True on the Timer.
    assert fake.daemon is True


def test_run_subprocess_propagates_called_process_error(monkeypatch):
    """A non-zero exit from pbcopy/wtype/ydotool surfaces as
    CalledProcessError so the unicode path can decide to bail out
    instead of pasting stale clipboard contents. Pre-fix, the bare
    `subprocess.run(check=True)` already raised this — keep the
    contract after we added the timeout."""
    from speakinput import injector as inj_mod

    def fake_run(*_a, **_kw):
        cp = MagicMock()
        cp.returncode = 1
        cp.stdout = b""
        cp.stderr = b"boom"
        raise subprocess.CalledProcessError(1, ["x"], stderr=b"boom")

    monkeypatch.setattr(inj_mod.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        inj_mod._run_subprocess(["pbcopy"])


def test_run_subprocess_timeout_does_not_freeze_caller(monkeypatch):
    """A subprocess that hangs longer than the timeout must NOT block
    the event worker indefinitely. The production `subprocess.run`
    with `timeout=X` is what enforces the cap — verify a fake that
    honors the timeout (raises TimeoutExpired) is what we get out,
    and that the exception is propagated to the caller.
    """
    from speakinput import injector as inj_mod

    def slow_run(*_a, **kw):
        # Mirror what the real subprocess.run does on timeout.
        timeout = kw.get("timeout")
        if timeout is None:
            raise RuntimeError("subprocess.run called without a timeout")
        raise subprocess.TimeoutExpired(cmd=kw.get("args", ["?"]), timeout=timeout)

    monkeypatch.setattr(inj_mod.subprocess, "run", slow_run)
    with pytest.raises(subprocess.TimeoutExpired):
        inj_mod._run_subprocess(["pbcopy"])


def test_pbcopy_timeout_does_not_freeze_unix_inject(monkeypatch, fake_modules, capsys):
    """End-to-end: a hung pbcopy on macOS raises TimeoutExpired on
    the TypingInjector unicode path. The injector must NOT proceed
    to send a Ctrl-V with whatever stale clipboard contents are
    there — the user would type the wrong text."""
    import subprocess as _sp
    keyboard_mod, _pyperclip_mod = fake_modules
    from speakinput.injector import TypingInjector

    monkeypatch.setattr("speakinput.injector.sys.platform", "darwin")

    def slow_pbcopy(*_a, **_kw):
        # Honor the timeout kwarg the production code passes.
        timeout = _kw.get("timeout")
        if timeout is not None:
            raise _sp.TimeoutExpired(cmd="pbcopy", timeout=timeout)
        raise RuntimeError("pbcopy did not receive a timeout kwarg")

    monkeypatch.setattr("speakinput.injector.subprocess.run", slow_pbcopy)

    inj = TypingInjector(restore_clipboard_ms=0, trailing_space=False)
    inj.inject("你好")
    # Ctrl+V must NOT have been sent — pbcopy failed.
    keyboard_mod.Controller.return_value.pressed.assert_not_called()
    keyboard_mod.Controller.return_value.tap.assert_not_called()
    # And the user got a warning, not a silent skip.
    out = capsys.readouterr().err
    assert "clipboard write failed" in out


def capsys_or_stderr() -> str:
    """Read whatever stderr the last test wrote.

    Kept for callers that don't want to thread `capsys` through;
    empty by default."""
    return ""


_STDERR_CAPTURE: dict = {}


def test_inject_lock_serializes_concurrent_inject_calls(fake_modules, monkeypatch):
    """The App-level `_inject_lock` must make two simultaneous
    injector.inject() calls run serially, not interleave. We
    simulate the chunked-body-vs-finalize race with a slow
    `Controller.type` and assert both calls' ordering is preserved
    in the recorded order.
    """
    from speakinput.app import App
    from speakinput.config import AudioConfig, Config

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux")
    config = Config(audio=AudioConfig(silence_threshold=0, auto_stop_seconds=0))
    recorder = MagicMock()
    feedback = MagicMock()
    app = App(
        config=config,
        recorder=recorder,
        transcribers={config.primary.key: MagicMock()},
        injector=MagicMock(),
        feedback=feedback,
    )

    # Wire the mock injector's `inject` to acquire the App's
    # `_inject_lock` around a slow type — exactly what the
    # production code does.
    order: list[str] = []
    type_evt = threading.Event()

    def slow_inject(text: str) -> None:
        with app._inject_lock:
            type_evt.set()
            time.sleep(0.1)
            order.append(text)
    app.injector.inject = slow_inject

    t1 = threading.Thread(target=app.injector.inject, args=("hello",))
    t2 = threading.Thread(target=app.injector.inject, args=("world",))
    t1.start()
    type_evt.wait(0.5)  # ensure t1 entered the lock first
    t2.start()
    t1.join()
    t2.join()
    # Each inject must have run to completion before the next
    # started — the lock prevented interleaving. The order
    # is deterministic because t1 entered first.
    assert order == ["hello", "world"]


def test_select_injector_warns_on_pure_wayland_fallback(monkeypatch, fake_modules, capsys):
    """Pure-Wayland sessions without wtype/ydotool fall back to
    pynput, but pynput only types through XWayland. Without the
    warning, the user types into the void with no error. Verify
    the warning fires."""
    from speakinput.config import InjectConfig
    from speakinput.injector import TypingInjector, select_injector

    monkeypatch.setattr("speakinput.injector.sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setattr("speakinput.injector.shutil.which", lambda _: None)
    inj = select_injector(InjectConfig())
    assert isinstance(inj, TypingInjector)
    out = capsys.readouterr().err
    assert "no Wayland typing backend" in out
    assert "wtype" in out
