"""pytest config: ensure `src/` is importable without an editable install,
and stub out `pynput` so its Carbon/HIToolbox background thread doesn't abort
the test process. Tests that need pynput behavior patch the stub."""

import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Stub pynput before any test module imports it. The real pynput starts a
# background thread on first import that talks to the macOS input source
# subsystem; in a test process without Input Monitoring permission, that
# thread can call abort() inside Carbon/HIToolbox. We give tests a MagicMock
# surface that they can patch further per-test if they need real behavior.
if "pynput" not in sys.modules:
    pynput = types.ModuleType("pynput")
    pynput_keyboard = types.ModuleType("pynput.keyboard")

    class _StubController:
        def __init__(self, *a, **kw):
            pass

        def type(self, *a, **kw):
            pass

        def press(self, *a, **kw):
            pass

        def release(self, *a, **kw):
            pass

        def tap(self, *a, **kw):
            pass

        def pressed(self, *a, **kw):
            class _Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *exc):
                    return False

            return _Ctx()

    class _StubKey:
        alt_r = "alt_r"
        alt_l = "alt_l"
        ctrl_r = "ctrl_r"
        ctrl_l = "ctrl_l"
        cmd_r = "cmd_r"
        cmd_l = "cmd_l"
        shift_r = "shift_r"
        shift_l = "shift_l"
        caps_lock = "caps_lock"
        f12 = "f12"
        cmd = "cmd"
        ctrl = "ctrl"

    class _StubListener:
        def __init__(self, *a, **kw):
            self.daemon = False

        def start(self):
            pass

        def stop(self):
            pass

        def join(self, *a, **kw):
            pass

        def is_alive(self):
            return False

    pynput_keyboard.Controller = _StubController
    pynput_keyboard.Key = _StubKey
    pynput_keyboard.Listener = _StubListener

    pynput.keyboard = pynput_keyboard
    sys.modules["pynput"] = pynput
    sys.modules["pynput.keyboard"] = pynput_keyboard
