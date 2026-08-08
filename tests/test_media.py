"""Tests for media playback control (pause/resume on push-to-talk).

Mocks `sys.platform`, `shutil.which`, and `subprocess.run` so every
backend branch (playerctl / osascript / powershell) can be exercised
on any host without touching the real MPRIS bus, AppleScript, or
PowerShell.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from speakinput import media as m


def _completed(returncode: int, stdout: str = ""):
    """A stand-in for subprocess.CompletedProcess (text mode)."""
    return SimpleNamespace(returncode=returncode, stdout=stdout)


@pytest.fixture
def fake_subprocess(monkeypatch):
    """Record subprocess.run calls; route results by full argv, then by
    the executable name. Defaults to a clean zero-exit empty result."""
    recorded: list[list[str]] = []
    results: dict[str, SimpleNamespace] = {}
    default_result = _completed(0, "")

    def _run(args, **kwargs):
        recorded.append(list(args))
        return results.get(" ".join(args), results.get(args[0], default_result))

    monkeypatch.setattr(m.subprocess, "run", _run)
    return SimpleNamespace(recorded=recorded, results=results)


# --- backend detection -----------------------------------------------------


@pytest.mark.parametrize(
    ("platform", "which_result", "expected"),
    [
        ("linux", "/usr/bin/playerctl", "playerctl"),
        ("linux", None, None),
        ("darwin", None, "osascript"),
        ("win32", None, "powershell"),
        ("freebsd", None, None),
    ],
)
def test_detect_backend(monkeypatch, platform, which_result, expected):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(m.shutil, "which", lambda name: which_result)
    assert m._detect_backend() == expected


def test_detect_backend_checks_which_only_on_linux(monkeypatch):
    """Non-Linux platforms pick their backend without consulting
    shutil.which (playerctl isn't a thing on Windows/macOS)."""
    monkeypatch.setattr(sys, "platform", "win32")
    which_called = []

    def _which(name):
        which_called.append(name)
        return None

    monkeypatch.setattr(m.shutil, "which", _which)
    assert m._detect_backend() == "powershell"
    assert which_called == []


def test_controller_available(monkeypatch):
    monkeypatch.setattr(m, "_detect_backend", lambda: "playerctl")
    assert m.MediaController().available is True

    monkeypatch.setattr(m, "_detect_backend", lambda: None)
    assert m.MediaController().available is False


# --- playerctl backend -----------------------------------------------------


def test_playerctl_check_playing_when_someone_is_playing(fake_subprocess):
    fake_subprocess.results["playerctl -a status"] = _completed(0, "Playing\nPaused\n")
    assert m._check_playing("playerctl") is True


def test_playerctl_check_not_playing_when_all_paused(fake_subprocess):
    fake_subprocess.results["playerctl -a status"] = _completed(0, "Paused\nStopped\n")
    assert m._check_playing("playerctl") is False


def test_playerctl_check_false_when_binary_unreachable(fake_subprocess):
    """playerctl installed but no MPRIS session → returncode != 0. We must
    not treat that as 'playing' (else we'd pause nothing and claim media
    was paused for every press)."""
    fake_subprocess.results["playerctl -a status"] = _completed(1, "")
    assert m._check_playing("playerctl") is False


def test_playerctl_check_false_when_player_missing(fake_subprocess):
    """playerctl entirely absent → `check_call`-style failure. Same
    contract as above: 'unknown' != 'playing'."""
    fake_subprocess.results["playerctl -a status"] = _completed(127, "playerctl: not found\n")
    assert m._check_playing("playerctl") is False


# --- MediaController.pause / resume ---------------------------------------


def test_pause_returns_true_and_marks_paused_when_media_playing(monkeypatch, fake_subprocess):
    fake_subprocess.results["playerctl -a status"] = _completed(0, "Playing\n")
    monkeypatch.setattr(m, "_detect_backend", lambda: "playerctl")
    controller = m.MediaController()
    assert controller.pause() is True
    assert controller._paused_by_us is True
    assert any(cmd == ["playerctl", "-a", "pause"] for cmd in fake_subprocess.recorded)


def test_pause_returns_false_and_does_not_pause_when_nothing_playing(
    monkeypatch, fake_subprocess,
):
    fake_subprocess.results["playerctl -a status"] = _completed(0, "Paused\n")
    monkeypatch.setattr(m, "_detect_backend", lambda: "playerctl")
    controller = m.MediaController()
    assert controller.pause() is False
    assert controller._paused_by_us is False
    assert not any(cmd == ["playerctl", "-a", "pause"] for cmd in fake_subprocess.recorded)


def test_pause_returns_false_when_no_backend(monkeypatch):
    monkeypatch.setattr(m, "_detect_backend", lambda: None)
    controller = m.MediaController()
    assert controller.pause() is False
    assert controller._paused_by_us is False


def test_resume_is_noop_when_we_never_paused(monkeypatch, fake_subprocess):
    monkeypatch.setattr(m, "_detect_backend", lambda: "playerctl")
    controller = m.MediaController()
    controller.resume()
    # No play command, and nothing recorded at all.
    assert fake_subprocess.recorded == []


def test_resume_only_resumes_media_we_paused(monkeypatch, fake_subprocess):
    fake_subprocess.results["playerctl -a status"] = _completed(0, "Playing\n")
    monkeypatch.setattr(m, "_detect_backend", lambda: "playerctl")
    controller = m.MediaController()
    assert controller.pause() is True
    fake_subprocess.recorded.clear()
    controller.resume()
    assert [cmd for cmd in fake_subprocess.recorded] == [["playerctl", "-a", "play"]]
    assert controller._paused_by_us is False


def test_resume_noop_when_no_backend(monkeypatch):
    monkeypatch.setattr(m, "_detect_backend", lambda: None)
    controller = m.MediaController()
    controller._paused_by_us = True  # even if something paused before the backend died
    controller.resume()  # must not raise


def test_pause_swallows_subprocess_failure(monkeypatch, fake_subprocess):
    """A broken playerctl (exception, not just non-zero) must not crash
    the press path — pause() degrades to 'not paused'."""
    def _boom(args, **kwargs):
        raise RuntimeError("dbus connection refused")

    monkeypatch.setattr(m.subprocess, "run", _boom)
    monkeypatch.setattr(m, "_detect_backend", lambda: "playerctl")
    controller = m.MediaController()
    assert controller.pause() is False
    assert controller._paused_by_us is False


def test_resume_swallows_subprocess_failure_and_resets_flag(monkeypatch):
    """Even if the resume command fails, the flag must clear so we don't
    keep trying to resume a dead session forever."""
    def _run(args, **kwargs):
        joined = " ".join(args)
        if joined == "playerctl -a status":
            return _completed(0, "Playing\n")
        if joined == "playerctl -a pause":
            return _completed(0, "")
        raise RuntimeError("playerctl vanished")

    monkeypatch.setattr(m.subprocess, "run", _run)
    monkeypatch.setattr(m, "_detect_backend", lambda: "playerctl")
    controller = m.MediaController()
    assert controller.pause() is True
    assert controller._paused_by_us is True
    controller.resume()  # 'play' raises — swallowed, flag still cleared
    assert controller._paused_by_us is False


# --- osascript / macOS backend --------------------------------------------


def test_osascript_runs_the_binary(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        m.subprocess, "run",
        lambda args, **kw: calls.append(args) or _completed(0, ""),
    )
    m._osascript('tell application "Music" to pause', timeout=7)
    assert calls == [["osascript", "-e", 'tell application "Music" to pause']]


def test_app_is_running_true_when_process_exists(monkeypatch):
    monkeypatch.setattr(m, "_osascript", lambda cmd: _completed(0, "true"))
    assert m._app_is_running("Spotify") is True


def test_app_is_running_false_on_error_or_output(monkeypatch):
    monkeypatch.setattr(m, "_osascript", lambda cmd: _completed(0, "false"))
    assert m._app_is_running("Spotify") is False
    monkeypatch.setattr(m, "_osascript", lambda cmd: _completed(1, "false"))
    assert m._app_is_running("Spotify") is False


def test_check_playing_osascript_positive(monkeypatch):
    running = {"Spotify": False, "Music": True}

    def _fake_running(app):
        return running.get(app, False)

    monkeypatch.setattr(m, "_app_is_running", _fake_running)
    monkeypatch.setattr(m, "_osascript", lambda cmd: _completed(0, "playing"))
    assert m._check_playing("osascript") is True


def test_check_playing_osascript_negative(monkeypatch):
    monkeypatch.setattr(m, "_app_is_running", lambda app: True)
    monkeypatch.setattr(m, "_osascript", lambda cmd: _completed(0, "paused"))
    assert m._check_playing("osascript") is False


def test_check_playing_osascript_with_no_apps_running(monkeypatch):
    monkeypatch.setattr(m, "_app_is_running", lambda app: False)
    assert m._check_playing("osascript") is False


def test_pause_all_osascript_pauses_only_running_apps(monkeypatch):
    running = {"Spotify": False, "Music": True}
    monkeypatch.setattr(m, "_app_is_running", lambda app: running.get(app, False))
    cmds: list[str] = []
    monkeypatch.setattr(
        m, "_osascript", lambda cmd, timeout=5: cmds.append(cmd) or _completed(0, "")
    )
    m._pause_all("osascript")
    assert cmds == ['tell application "Music" to pause']


def test_resume_all_osascript_plays_running_apps(monkeypatch):
    monkeypatch.setattr(m, "_app_is_running", lambda app: True)
    cmds: list[str] = []
    monkeypatch.setattr(
        m, "_osascript", lambda cmd, timeout=5: cmds.append(cmd) or _completed(0, "")
    )
    m._resume_all("osascript")
    assert cmds == [
        'tell application "Spotify" to play',
        'tell application "Music" to play',
    ]


def test_resume_all_osascript_skips_apps_not_running(monkeypatch):
    """Only running apps get a resume command — a quit Music.app must not
    be sent an AppleScript play (it would just error)."""
    running = {"Spotify": True, "Music": False}
    monkeypatch.setattr(m, "_app_is_running", lambda app: running.get(app, False))
    cmds: list[str] = []
    monkeypatch.setattr(
        m, "_osascript", lambda cmd, timeout=5: cmds.append(cmd) or _completed(0, "")
    )
    m._resume_all("osascript")
    assert cmds == ['tell application "Spotify" to play']


# --- powershell / Windows backend -----------------------------------------


def test_check_playing_powershell_uses_exit_code(fake_subprocess):
    """The PowerShell script exits 0 when a session is playing; any other
    exit code means not playing (or nothing running)."""
    fake_subprocess.results["powershell"] = _completed(0)
    assert m._check_playing("powershell") is True

    fake_subprocess.results["powershell"] = _completed(53)
    assert m._check_playing("powershell") is False


def test_pause_all_powershell_runs_the_script(fake_subprocess):
    m._pause_all("powershell")
    assert len(fake_subprocess.recorded) == 1
    assert fake_subprocess.recorded[0][0] == "powershell"
    assert "TryPauseAsync" in fake_subprocess.recorded[0][3]


def test_resume_all_powershell_runs_the_script(fake_subprocess):
    m._resume_all("powershell")
    assert fake_subprocess.recorded[0][0] == "powershell"
    assert "TryPlayAsync" in fake_subprocess.recorded[0][3]


def test_unknown_backend_is_safe(fake_subprocess):
    assert m._check_playing("cinnamon") is False
    assert fake_subprocess.recorded == []
    m._pause_all("cinnamon")  # must not raise
    m._resume_all("cinnamon")  # must not raise