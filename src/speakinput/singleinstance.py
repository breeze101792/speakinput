"""Single-instance guard: refuse to start if another speakinput is running.

The symptom this prevents: the user starts speakinput, then runs start.sh
again (or has a leftover process from a previous session) and ends up with
N listeners all responding to the same push-to-talk key. The output then
appears N times in the focused field — every key release fans out to every
running process.

Mechanism: hold an exclusive `flock` on a per-user lockfile. If the lock
is already held, the new process exits with a clear error. The lock is
released automatically when the process exits (the kernel closes the fd).

The fd is intentionally inherited into the child Python process via
`pass_fds` so the lock survives the `exec` in start.sh / the CLI entry
point. We never close it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from platformdirs import user_runtime_dir

_LOCK_FILENAME = "speakinput.lock"


def _lockfile_path() -> Path:
    """Per-user lockfile. `user_runtime_dir` is the macOS-standard location
    for runtime artifacts like sockets, pipes, and lockfiles."""
    runtime = Path(user_runtime_dir("speakinput", appauthor=False))
    runtime.mkdir(parents=True, exist_ok=True)
    return runtime / _LOCK_FILENAME


def acquire() -> int:
    """Acquire the single-instance lock. Returns the open fd on success.

    Raises SystemExit(3) if another speakinput is already running.
    """
    path = _lockfile_path()
    # O_RDWR so the fd is writable, which flock requires on some systems.
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        # LOCK_EX = exclusive, LOCK_NB = non-blocking. If another process
        # holds the lock, flock returns immediately rather than waiting.
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        os.close(fd)
        print(
            f"error: another speakinput is already running "
            f"(lockfile: {path}). Stop it first, or run `pkill -f speakinput`.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    # Truncate and write the current pid for debugging. If you see a stale
    # pid in this file, the previous process exited without releasing the
    # lock — but since the lock is fd-based, the kernel clears it for us.
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def release(fd: int) -> None:
    """Release the single-instance lock. Safe to call multiple times."""
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
