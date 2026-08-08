"""Tests for the `python -m speakinput` entrypoint (`__main__.py`).

`__main__.py` is a two-line module: it binds `speakinput.cli.main` and
raises `SystemExit(main())` only when run as a script. These tests pin
that contract and add one real end-to-end subprocess check through the
installed package.
"""

from __future__ import annotations

import runpy
import sys

import pytest


def test_run_as_module_dispatches_to_cli_main(monkeypatch):
    """Running the module as `python -m speakinput` must invoke
    `speakinput.cli.main()` and translate its return code into a
    SystemExit."""
    from speakinput import cli

    captured: dict = {}

    def _fake_main(argv=None):
        captured["argv"] = argv
        return 42

    monkeypatch.setattr(cli, "main", _fake_main)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("speakinput.__main__", run_name="__main__")
    assert excinfo.value.code == 42
    # `main()` reads sys.argv itself when called with no arguments.
    assert captured["argv"] is None


def test_module_import_does_not_dispatch(monkeypatch):
    """Importing `speakinput.__main__` (as `python -m pytest ...` in some
    tooling does) must NOT start the CLI — only the `python -m` guard may."""
    from speakinput import cli

    def _explode(argv=None):
        raise AssertionError("cli.main ran during a plain import")

    monkeypatch.setattr(cli, "main", _explode)
    # run_name different from "__main__" → guard is False → no invocation.
    runpy.run_module("speakinput.__main__", run_name="speakinput.__main__")


def test_console_entrypoint_lists_models():
    """End-to-end through a fresh Python process: `python -m speakinput
    --list-models` must reach cli.main via __main__.py and exit 0. Runs
    from the repo root so Python resolves the `speakinput` package."""
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "speakinput", "--list-models"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=60,
    )
    assert result.returncode == 0
    assert "curated models" in result.stdout
    assert "tiny.en" in result.stdout