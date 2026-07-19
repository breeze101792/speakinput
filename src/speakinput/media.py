"""Media playback control: pause/resume via platform-native tools.

On Linux, uses ``playerctl`` (MPRIS D-Bus interface).
On macOS, uses ``osascript`` for Spotify and Music.app.
On Windows, uses PowerShell for SMTC.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys

log = logging.getLogger("speakinput")


class MediaController:
    """Pause/resume media playback on push-to-talk.

    Call ``pause()`` on hotkey press. If any media player was actively
    playing, it is paused and the method returns ``True``.
    Call ``resume()`` on hotkey release — it only resumes if we paused.
    """

    def __init__(self) -> None:
        self._paused_by_us = False
        self._backend = _detect_backend()

    @property
    def available(self) -> bool:
        return self._backend is not None

    def pause(self) -> bool:
        """Pause any playing media. Returns True if media was paused."""
        if self._backend is None:
            return False
        try:
            if _check_playing(self._backend):
                _pause_all(self._backend)
                self._paused_by_us = True
                return True
        except Exception:
            log.exception("failed to pause media")
        return False

    def resume(self) -> None:
        """Resume media that was paused by us."""
        if not self._paused_by_us or self._backend is None:
            return
        try:
            _resume_all(self._backend)
        except Exception:
            log.exception("failed to resume media")
        self._paused_by_us = False


def _detect_backend() -> str | None:
    if sys.platform == "linux" and shutil.which("playerctl"):
        return "playerctl"
    if sys.platform == "darwin":
        return "osascript"
    if sys.platform == "win32":
        return "powershell"
    return None


def _osascript(cmd: str, timeout: int = 5) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["osascript", "-e", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _app_is_running(app_name: str) -> bool:
    r = _osascript(
        f'tell application "System Events" to get exists of process "{app_name}"'
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def _check_playing(backend: str) -> bool:
    if backend == "playerctl":
        r = subprocess.run(
            ["playerctl", "-a", "status"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and "Playing" in r.stdout

    if backend == "osascript":
        for app_name in ("Spotify", "Music"):
            if not _app_is_running(app_name):
                continue
            r = _osascript(
                f'tell application "{app_name}" to get player state'
            )
            if r.returncode == 0 and r.stdout.strip().lower() == "playing":
                return True
        return False

    if backend == "powershell":
        script = (
            'Add-Type -AssemblyName System.Runtime.WindowsRuntime; '
            '$mgr = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]::RequestAsync().GetAwaiter().GetResult(); '
            '$s = $mgr.GetCurrentSession(); '
            'if ($s -eq $null) { exit 1 }; '
            '$pi = $s.GetPlaybackInfo(); '
            'exit (3 - $pi.PlaybackStatus)'
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0

    return False


def _pause_all(backend: str) -> None:
    if backend == "playerctl":
        subprocess.run(
            ["playerctl", "-a", "pause"],
            capture_output=True, timeout=5,
        )
    elif backend == "osascript":
        for app_name in ("Spotify", "Music"):
            if not _app_is_running(app_name):
                continue
            _osascript(f'tell application "{app_name}" to pause')
    elif backend == "powershell":
        script = (
            'Add-Type -AssemblyName System.Runtime.WindowsRuntime; '
            '$mgr = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]::RequestAsync().GetAwaiter().GetResult(); '
            '$s = $mgr.GetCurrentSession(); '
            'if ($s -ne $null) { $s.TryPauseAsync().GetAwaiter().GetResult() }'
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, timeout=5,
        )


def _resume_all(backend: str) -> None:
    if backend == "playerctl":
        subprocess.run(
            ["playerctl", "-a", "play"],
            capture_output=True, timeout=5,
        )
    elif backend == "osascript":
        for app_name in ("Spotify", "Music"):
            if not _app_is_running(app_name):
                continue
            _osascript(f'tell application "{app_name}" to play')
    elif backend == "powershell":
        script = (
            'Add-Type -AssemblyName System.Runtime.WindowsRuntime; '
            '$mgr = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]::RequestAsync().GetAwaiter().GetResult(); '
            '$s = $mgr.GetCurrentSession(); '
            'if ($s -ne $null) { $s.TryPlayAsync().GetAwaiter().GetResult() }'
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, timeout=5,
        )
