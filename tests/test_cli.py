"""Tests for CLI flags that don't require the full app to start.

The bulk of speakinput's CLI (model loading, hotkey, audio) is
exercised by tests/test_app.py with the bootstrap path mocked. This
file covers the standalone flags: -C / --edit-config, which must
short-circuit before model loading and before the single-instance
lock."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_xdg(monkeypatch, tmp_path):
    """Redirect platformdirs to a temp dir so tests don't touch the
    real user config (~/.config/speakinput or macOS's Application
    Support). Both XDG vars and HOME need overriding because
    platformdirs prefers XDG when set, and the user config dir is
    a subpath of HOME-derived dirs on macOS."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_config"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg_runtime"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def _run_cli(monkeypatch, *argv):
    """Invoke speakinput.cli.main with a list of argv-like strings and
    return the (returncode, captured stdout, captured stderr) triple.

    We stub out any side-effecting imports the CLI normally drags in
    (acquire_instance_lock) so tests can run without a working
    microphone, accessibility permission, or a unique runtime dir.
    """
    from speakinput import cli

    monkeypatch.setattr(cli, "acquire_instance_lock", lambda: None)
    with patch.object(sys, "argv", ["speakinput", *argv]):
        with patch("sys.stdout") as out, patch("sys.stderr") as err:
            rc = cli.main(list(argv))
    return rc, out, err


# --- the flag itself -------------------------------------------------------


def test_edit_config_flag_is_parsed():
    """argparse must accept both -C and --edit-config."""
    from speakinput.cli import _build_parser

    p = _build_parser()
    args = p.parse_args(["-C"])
    assert args.edit_config is True
    args = p.parse_args(["--edit-config"])
    assert args.edit_config is True
    args = p.parse_args([])
    assert args.edit_config is False


def test_edit_config_short_circuits_before_model_load(
    monkeypatch, isolated_xdg
):
    """-C must NOT go through the bootstrap or single-instance lock.
    The whole point is to be runnable while another speakinput is
    already running (or before any model is downloaded)."""
    from speakinput import cli

    # Sentinel: if either of these gets called, the short-circuit
    # failed. The CLI imports ensure_model at module load, so we patch
    # it on the module attribute the function looks up.
    def _explode(*a, **kw):
        raise AssertionError("ensure_model was called during -C")

    monkeypatch.setattr(cli, "ensure_model", _explode, raising=True)
    # Patching ensure_model above would also affect a real run; since
    # the short-circuit must happen before any of it, _explode will
    # never fire in the success case.
    monkeypatch.setattr(cli, "acquire_instance_lock", _explode, raising=False)
    # Simulate the editor.
    monkeypatch.setenv("VISUAL", "true")  # POSIX `true` is a no-op zero-exit
    rc, _out, _err = _run_cli(monkeypatch, "-C")
    assert rc == 0


# --- path resolution and seeding ------------------------------------------


def test_edit_config_opens_default_path_when_no_flag(monkeypatch, isolated_xdg):
    """With no -c, -C opens default_config_path() and seeds from the
    bundled example if the file doesn't exist yet."""
    from speakinput import cli
    from speakinput.config import default_config_path

    target = default_config_path()
    assert not target.exists()

    captured: dict = {}

    def _fake_run(args, check=False):
        captured["args"] = args
        captured["check"] = check
        # Don't write — the editor would, but we want to observe the
        # seed put in place by `_seed_example` before the launch.
        from unittest.mock import MagicMock

        m = MagicMock()
        m.returncode = 0
        return m

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    rc, _out, err = _run_cli(monkeypatch, "-C")
    assert rc == 0
    # Editor was called with the default config path.
    assert captured["args"][1] == str(target)
    # And the example was seeded into that path.
    assert target.exists()
    assert target.read_text()  # non-empty
    # Stderr told the user what's happening.
    written = "".join(
        call.args[0] if call.args else "" for call in err.write.call_args_list
    )
    assert "opening" in written


def test_edit_config_seeds_from_bundled_example(monkeypatch, isolated_xdg):
    """The seeded file must be the bundled config.example.toml, not an
    empty file or a generic stub."""
    from speakinput import cli
    from speakinput.config import default_config_path

    # The bundled example ships in the repo root.
    repo_root = Path(__file__).resolve().parent.parent
    example = repo_root / "config.example.toml"
    assert example.is_file(), "config.example.toml must exist for this test"

    target = default_config_path()

    def _fake_run(args, check=False):
        # Don't write — let the seeded bytes survive to the assert below.
        from unittest.mock import MagicMock

        m = MagicMock()
        m.returncode = 0
        return m

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    rc, _out, _err = _run_cli(monkeypatch, "-C")
    assert rc == 0
    assert target.read_text() == example.read_text()


def test_edit_config_does_not_overwrite_existing(monkeypatch, isolated_xdg):
    """If the config file already exists, -C must NOT replace it with
    the example — the user would lose their customizations."""
    from speakinput import cli
    from speakinput.config import default_config_path

    target = default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# user-customized — must NOT be clobbered\n")

    def _fake_run(args, check=False):
        from unittest.mock import MagicMock

        m = MagicMock()
        m.returncode = 0
        return m

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    rc, _out, _err = _run_cli(monkeypatch, "-C")
    assert rc == 0
    assert target.read_text() == "# user-customized — must NOT be clobbered\n"


def test_edit_config_honors_explicit_config_flag(monkeypatch, isolated_xdg, tmp_path):
    """-C -c /path/to/my.toml opens /path/to/my.toml, creating the
    parent dir on demand (the example would go in ~/.config/speakinput/
    and that path might not exist yet)."""
    from speakinput import cli

    explicit = tmp_path / "nested" / "subdir" / "config.toml"
    assert not explicit.parent.exists()

    captured: dict = {}

    def _fake_run(args, check=False):
        captured["args"] = args
        Path(args[1]).write_text("seeded")
        from unittest.mock import MagicMock

        m = MagicMock()
        m.returncode = 0
        return m

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    rc, _out, _err = _run_cli(monkeypatch, "-C", "-c", str(explicit))
    assert rc == 0
    assert captured["args"][1] == str(explicit)
    assert explicit.parent.is_dir()


# --- editor selection + exit code ----------------------------------------


def test_edit_config_uses_visual_over_editor(monkeypatch, isolated_xdg):
    """$VISUAL wins over $EDITOR — it's the standard Unix convention
    (GUI vs terminal editor)."""
    from speakinput import cli

    monkeypatch.setenv("VISUAL", "code")
    monkeypatch.setenv("EDITOR", "nano")
    captured: dict = {}

    def _fake_run(args, check=False):
        captured["editor"] = args[0]
        from unittest.mock import MagicMock

        m = MagicMock()
        m.returncode = 0
        return m

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    rc, _, _ = _run_cli(monkeypatch, "-C")
    assert rc == 0
    assert captured["editor"] == "code"


def test_edit_config_falls_back_to_editor_then_vi(monkeypatch, isolated_xdg):
    """No $VISUAL/$EDITOR → vi. Set and unset the vars to make the
    fallback deterministic across developer machines."""
    from speakinput import cli

    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    captured: dict = {}

    def _fake_run(args, check=False):
        captured["editor"] = args[0]
        from unittest.mock import MagicMock

        m = MagicMock()
        m.returncode = 0
        return m

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    rc, _, _ = _run_cli(monkeypatch, "-C")
    assert rc == 0
    assert captured["editor"] == "vi"


def test_edit_config_returns_editor_exit_code(monkeypatch, isolated_xdg):
    """Whatever the editor returns, we propagate it. A non-zero exit
    means the user (or their editor) signalled a problem — a script
    wrapping -C can detect that."""
    from speakinput import cli
    from unittest.mock import MagicMock

    def _fake_run(args, check=False):
        m = MagicMock()
        m.returncode = 42  # arbitrary non-zero, the editor's choice
        return m

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    rc, _, _ = _run_cli(monkeypatch, "-C")
    assert rc == 42


def test_edit_config_returns_1_when_editor_missing(monkeypatch, isolated_xdg):
    """A typo'd $VISUAL must NOT silently succeed. The user needs to
    know their env var points at a non-existent binary."""
    from speakinput import cli

    monkeypatch.setenv("VISUAL", "totally-not-an-editor-xyz")
    # Force the OSError that subprocess.run raises when the binary
    # can't be found.
    def _missing(args, check=False):
        raise FileNotFoundError(2, "No such file or directory", args[0])

    monkeypatch.setattr(cli.subprocess, "run", _missing)
    rc, _, err = _run_cli(monkeypatch, "-C")
    assert rc == 1
    err_str = "".join(
        call.args[0] if call.args else "" for call in err.write.call_args_list
    )
    assert "not found" in err_str or "not found" in (err_str + "")
