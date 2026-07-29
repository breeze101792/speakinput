"""Self-healing pynput macOS CGEventTap.

pynput's macOS listener creates a `CGEventTap` and waits on a `CFRunLoop`.
When macOS disables the tap (brief sleep/wake, focus transitions to a
process that consumes the event, permission churn, certain app
interactions), pynput's run loop just times out and waits — pynput
never calls `CGEventTapIsEnabled` and never re-enables the tap. The
thread stays alive, so naive liveness checks see a healthy listener,
but no events ever arrive. The hotkey is dead until restart.

This module monkey-patches `pynput._util.darwin.ListenerMixin._run` at
import time so the run loop checks the tap's enabled state on every
iteration and re-enables it if macOS turned it off. The patch is
applied only on Darwin (no-op elsewhere) and only if pynput + Quartz
are importable.

The patched `_run` is behavior-preserving for the common case
(enabled tap → identical to upstream) and self-healing for the
disabled case (1s detection latency, since the run loop polls
`CFRunLoopRunInMode` with a 1s timeout). One `[info]` line per
recovery so the user can grep their log for the cause.
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

_PATCHED = False


def _install_darwin_tap_healer() -> None:
    """Wrap pynput's darwin ListenerMixin._run with a self-healing loop.

    Idempotent: subsequent calls are no-ops. Safe to import multiple
    times. Only does anything on macOS with pynput + Quartz available.
    """
    global _PATCHED
    if _PATCHED:
        return
    if sys.platform != "darwin":
        _PATCHED = True  # so we don't re-check on every import
        return
    try:
        import pynput._util.darwin as _darwin
        import Quartz
    except ImportError:
        # pynput or pyobjc missing — speakinput won't work on macOS
        # anyway, and the existing pynput-import guard in hotkey.py
        # surfaces the real error to the user. Don't patch.
        _PATCHED = True
        return

    # NOTE: We deliberately do NOT capture `Quartz.CGEventTapIsEnabled`
    # and `Quartz.CGEventTapEnable` as closure variables. The patched
    # `_run` resolves them lazily from the live `Quartz` module on
    # each iteration, so:
    #   - production code always uses the real macOS functions, and
    #   - tests can `monkeypatch.setattr(Quartz, "CGEventTapEnable", ...)`
    #     and the swap takes effect immediately.
    # If we'd captured them at install time, tests would have to
    # re-install the patch every time they swapped the functions, or
    # they'd hit the real (tap-touching) C entry point and hang.
    _HIServices = getattr(_darwin, "HIServices", None)
    _log = log

    def _self_healing_run(self):  # noqa: ANN001 - matches pynput signature
        # Mirror the upstream trust check so the warning still fires.
        if _HIServices is not None:
            try:
                self.IS_TRUSTED = _HIServices.AXIsProcessTrusted()
                if not self.IS_TRUSTED:
                    self._log.warning(
                        "This process is not trusted! Input event monitoring "
                        "will not be possible until it is added to "
                        "accessibility clients."
                    )
            except Exception:
                pass

        self._loop = None
        tap = None
        try:
            tap = self._create_event_tap()
            if tap is None:
                self._mark_ready()
                return

            loop_source = _darwin.CFMachPortCreateRunLoopSource(None, tap, 0)
            self._loop = _darwin.CFRunLoopGetCurrent()
            _darwin.CFRunLoopAddSource(
                self._loop, loop_source, _darwin.kCFRunLoopDefaultMode
            )
            Quartz.CGEventTapEnable(tap, True)

            self._mark_ready()

            # Bound the time we spend doing the self-heal check so a
            # pathological re-enable failure can't starve the run loop
            # (defensive — the check is a single ObjC BOOL read).
            _RECOVERED = False
            try:
                while self.running:
                    result = _darwin.CFRunLoopRunInMode(
                        _darwin.kCFRunLoopDefaultMode, 1, False
                    )
                    try:
                        if result != _darwin.kCFRunLoopRunTimedOut:
                            break
                    except AttributeError:
                        # Teardown of the virtual machine.
                        break
                    # Self-heal: macOS can disable a CGEventTap
                    # asynchronously (sleep/wake, focus churn, permission
                    # transition, certain app interactions). When that
                    # happens pynput's run loop just keeps timing out
                    # without delivering events. Re-enable the tap so
                    # events resume; the next loop iteration will see
                    # them. Rate-limited to one log line per stuck
                    # episode so a flapping tap doesn't spam the log.
                    try:
                        if not Quartz.CGEventTapIsEnabled(tap):
                            Quartz.CGEventTapEnable(tap, True)
                            if not _RECOVERED:
                                _log.info(
                                    "macOS disabled the hotkey event tap; "
                                    "re-enabled it (this can happen on "
                                    "sleep/wake or focus transitions)"
                                )
                                _RECOVERED = True
                        else:
                            _RECOVERED = False
                    except Exception:
                        # A rare ObjC bridge failure shouldn't kill the
                        # run loop. Worst case the next iteration tries
                        # again.
                        pass

            except Exception:
                # Pass through to the upstream behaviour (which also
                # swallows the exception); the thread exits and our
                # liveness watcher will see the dead thread and
                # restart the listener.
                pass

        finally:
            self._loop = None
            # Best-effort: disable the tap so we don't leak a CoreGraphics
            # event tap if the listener is being torn down. The tap will
            # be re-enabled by the next listener instance when needed.
            if tap is not None:
                try:
                    Quartz.CGEventTapEnable(tap, False)
                except Exception:
                    pass

    # Replace the method. We assign to the class (not an instance) so
    # every Listener created after this point gets the self-healing
    # behavior. This is the same pattern pynput uses internally for
    # its platform-specific mixins.
    _darwin.ListenerMixin._run = _self_healing_run  # type: ignore[assignment]
    _PATCHED = True


# Install on import. The hotkey module imports pynput at the top, so
# importing this module right after that is the right place to apply
# the patch — by the time any Listener is constructed, _run is the
# patched version.
_install_darwin_tap_healer()
