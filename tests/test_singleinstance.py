"""Tests for the single-instance lockfile guard."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


def test_acquire_succeeds_when_lock_free(tmp_path: Path, monkeypatch):
    """The first process to acquire gets a valid fd back."""
    from speakinput import singleinstance as si

    # Redirect the lockfile location to a temp dir so we don't pollute
    # the real runtime dir.
    monkeypatch.setattr(si, "_lockfile_path", lambda: tmp_path / "lock")

    fd = si.acquire()
    try:
        assert fd >= 0
        # The pid should have been written for debugging.
        contents = (tmp_path / "lock").read_bytes()
        assert str(os.getpid()).encode() in contents
    finally:
        si.release(fd)


def test_second_acquire_exits(monkeypatch, capsys):
    """Holding the lock blocks a second acquire and exits 3."""
    from speakinput import singleinstance as si

    fd_path = Path(tempfile.mkstemp()[1])
    monkeypatch.setattr(si, "_lockfile_path", lambda: fd_path)

    first_fd = si.acquire()
    try:
        with pytest.raises(SystemExit) as exc_info:
            si.acquire()
        assert exc_info.value.code == 3
        captured = capsys.readouterr()
        assert "another speakinput is already running" in captured.err
    finally:
        si.release(first_fd)


def test_release_is_idempotent():
    """Calling release on a closed/invalid fd must not raise."""
    from speakinput import singleinstance as si

    si.release(-1)  # bad fd
    si.release(99999)  # already-closed fd
