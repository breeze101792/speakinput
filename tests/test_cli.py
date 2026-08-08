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
    # failed. The CLI doesn't import ensure_model directly (it's
    # called via App.run), so we patch it on the models module in
    # case the bootstrap path is reached. We also patch the
    # single-instance lock.
    from speakinput import models as _models

    def _explode(*a, **kw):
        raise AssertionError("ensure_model was called during -C")

    monkeypatch.setattr(_models, "ensure_model", _explode, raising=False)
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


# --- -l / -L: device and model listing ------------------------------------


def test_list_devices_flag_parses_as_short_and_long():
    """argparse must accept both -l and --list-devices."""
    from speakinput.cli import _build_parser

    p = _build_parser()
    assert p.parse_args(["-l"]).list_devices is True
    assert p.parse_args(["--list-devices"]).list_devices is True
    assert p.parse_args([]).list_devices is False


def test_list_devices_prints_the_device_table(monkeypatch):
    from speakinput import cli

    devices = [
        {
            "index": 0,
            "max_input_channels": 2,
            "default_samplerate": 48000.0,
            "name": "Built-in Microphone",
        },
        {
            "index": 1,
            "max_input_channels": 1,
            "default_samplerate": 16000.0,
            "name": "USB Headset",
        },
    ]
    monkeypatch.setattr(cli, "list_input_devices", lambda: devices)
    rc, out, _err = _run_cli(monkeypatch, "-l")
    assert rc == 0
    written = "".join(
        call.args[0] if call.args else "" for call in out.write.call_args_list
    )
    assert "idx" in written and "channels" in written and "rate" in written
    assert "Built-in Microphone" in written
    assert "USB Headset" in written


def test_list_devices_no_input_devices_exits_1(monkeypatch):
    from speakinput import cli

    monkeypatch.setattr(cli, "list_input_devices", lambda: [])
    rc, _out, err = _run_cli(monkeypatch, "-l")
    assert rc == 1
    err_str = "".join(
        call.args[0] if call.args else "" for call in err.write.call_args_list
    )
    assert "no input devices" in err_str


def test_list_devices_broken_audio_stack_exits_1(monkeypatch):
    from speakinput import cli

    def _boom():
        raise RuntimeError("no PulseAudio daemon running")

    monkeypatch.setattr(cli, "list_input_devices", _boom)
    rc, _out, err = _run_cli(monkeypatch, "-l")
    assert rc == 1
    err_str = "".join(
        call.args[0] if call.args else "" for call in err.write.call_args_list
    )
    assert "no PulseAudio daemon running" in err_str


def test_list_models_prints_the_curated_set(monkeypatch):
    rc, out, _err = _run_cli(monkeypatch, "-L")
    assert rc == 0
    written = "".join(
        call.args[0] if call.args else "" for call in out.write.call_args_list
    )
    assert "curated models" in written
    assert "tiny.en" in written
    assert "large-v3" in written
    assert "English-only" in written


def test_list_flags_do_not_acquire_single_instance_lock(monkeypatch, isolated_xdg):
    """-l and -L are standalone actions — they must NOT touch the single-
    instance lock (a user might list devices while the app is running)."""
    from speakinput import cli

    def _explode(*a, **kw):
        raise AssertionError("single-instance lock acquired while listing")

    monkeypatch.setattr(cli, "acquire_instance_lock", _explode, raising=False)
    monkeypatch.setattr(cli, "list_input_devices", lambda: [])
    with patch.object(sys, "argv", ["speakinput"]):
        with patch("sys.stdout"), patch("sys.stderr"):
            assert cli.main(["-l"]) == 1  # empty device table, still no lock
            assert cli.main(["-L"]) == 0


# --- main(): the app-launch path ------------------------------------------


def test_main_config_load_failure_prints_error_and_exits_1(monkeypatch):
    from speakinput import cli

    def _boom(path):
        raise ValueError("invalid TOML on line 3")

    monkeypatch.setattr(cli, "load_config", _boom)
    rc, _out, err = _run_cli(monkeypatch)
    assert rc == 1
    err_str = "".join(
        call.args[0] if call.args else "" for call in err.write.call_args_list
    )
    assert "invalid TOML on line 3" in err_str


def test_main_constructs_app_with_cli_overrides(monkeypatch):
    """Every override flag must flow into the Config handed to App:
    model, language, trailing_space, pause_media, gpu_device, threads
    and the initial prompt."""
    from speakinput import cli
    from unittest.mock import MagicMock

    from speakinput.config import Config

    monkeypatch.setattr(cli, "load_config", lambda path: (Config(), "fake"))
    fake_app = MagicMock()
    app_cls = MagicMock(return_value=fake_app)
    monkeypatch.setattr(cli, "App", app_cls)

    rc, _out, _err = _run_cli(
        monkeypatch,
        "-m", "base.en",
        "-g", "zh",
        "-T",
        "--no-pause-media",
        "--gpu-device", "1",
        "--threads", "2",
        "-P", "kubectl",
    )
    assert rc == 0
    cfg = app_cls.call_args.args[0]
    assert cfg.primary.model == "base.en"
    assert cfg.primary.language == "zh"
    assert cfg.primary.initial_prompt == "kubectl"
    assert cfg.inject.trailing_space is False
    assert cfg.audio.pause_media is False
    assert cfg.transcribe.gpu_device == 1
    assert cfg.transcribe.n_threads == 2
    fake_app.run.assert_called_once()


def test_main_forward_dry_run_flag(monkeypatch):
    from speakinput import cli
    from unittest.mock import MagicMock

    from speakinput.config import Config

    monkeypatch.setattr(cli, "load_config", lambda path: (Config(), "fake"))
    app_cls = MagicMock()
    monkeypatch.setattr(cli, "App", app_cls)
    rc, _out, _err = _run_cli(monkeypatch, "-n")
    assert rc == 0
    assert app_cls.call_args.kwargs["dry_run"] is True


def test_main_passes_config_source_through(monkeypatch):
    from speakinput import cli
    from unittest.mock import MagicMock

    from speakinput.config import Config

    source = MagicMock()
    source.is_file.return_value = True
    monkeypatch.setattr(cli, "load_config", lambda path: (Config(), source))
    app_cls = MagicMock()
    monkeypatch.setattr(cli, "App", app_cls)
    rc, _out, _err = _run_cli(monkeypatch)
    assert rc == 0
    assert app_cls.call_args.kwargs["config_source"] is source


def test_main_diagnose_short_circuits_before_app(monkeypatch):
    """--diagnose records + prints RMS and exits before constructing App."""
    from speakinput import cli

    monkeypatch.setattr(cli, "load_config", lambda path: (None, None))
    monkeypatch.setattr(cli, "_diagnose", lambda config: 7)
    rc, _out, _err = _run_cli(monkeypatch, "-D")
    assert rc == 7


def test_main_rejects_invalid_model_choice():
    """argparse choices must reject bogus model names with exit code 2."""
    from speakinput.cli import _build_parser

    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args(["-m", "not-a-model"])
    assert excinfo.value.code == 2


def test_gpu_flags_map_to_use_gpu(monkeypatch):
    """--gpu and --no-gpu must both land on the same destination the app
    reads (use_gpu). A past bug left --gpu writing to a dead `gpu` dest,
    so the "force GPU on" flag silently did nothing."""
    from speakinput import cli
    from unittest.mock import MagicMock

    from speakinput.config import Config
    from speakinput.cli import _build_parser

    # Parsing: both flags share the use_gpu destination.
    assert _build_parser().parse_args(["--gpu"]).use_gpu is True
    assert _build_parser().parse_args(["--no-gpu"]).use_gpu is False
    assert _build_parser().parse_args([]).use_gpu is None

    # End-to-end: --gpu must reach the Config handed to App.
    monkeypatch.setattr(cli, "load_config", lambda path: (Config(), "fake"))
    app_cls = MagicMock()
    monkeypatch.setattr(cli, "App", app_cls)
    rc, _out, _err = _run_cli(monkeypatch, "--gpu")
    assert rc == 0
    assert app_cls.call_args.args[0].transcribe.use_gpu is True


def test_parser_accepts_invalid_model_choice():
    """-m accepts every name in the curated list (and rejects others)."""
    from speakinput.cli import _build_parser
    from speakinput.config import VALID_MODELS

    p = _build_parser()
    for name in VALID_MODELS:
        assert p.parse_args(["-m", name]).model == name


# --- example-config seeding ------------------------------------------------


def test_seed_example_copies_when_target_missing(monkeypatch, tmp_path):
    from speakinput import cli

    target = tmp_path / "config.toml"
    example = tmp_path / "example.toml"
    example.write_text("x = 1\n")
    monkeypatch.setattr(cli, "_example_config_path", lambda: example)
    cli._seed_example(target)
    assert target.read_text() == "x = 1\n"


def test_seed_example_never_overwrites_existing_target(monkeypatch, tmp_path):
    from speakinput import cli

    target = tmp_path / "config.toml"
    target.write_text("# user file")

    def _explode():
        raise AssertionError("example lookup attempted for an existing config")

    monkeypatch.setattr(cli, "_example_config_path", _explode)
    cli._seed_example(target)
    assert target.read_text() == "# user file"


def test_seed_example_skips_when_no_example_bundled(monkeypatch, tmp_path):
    from speakinput import cli

    target = tmp_path / "config.toml"
    monkeypatch.setattr(cli, "_example_config_path", lambda: None)
    cli._seed_example(target)  # must not crash
    assert not target.exists()


def test_example_config_path_finds_a_file():
    """Whatever the resolution order, `_example_config_path` must return a
    path to an existing config.example.toml (repo root in this checkout)."""
    from speakinput import cli

    found = cli._example_config_path()
    assert found is not None
    assert found.is_file()
    assert found.name == "config.example.toml"


def test_example_config_path_returns_none_when_unfindable(monkeypatch, tmp_path):
    """If the package and CWD both lack a config.example.toml, report None
    so the caller can fall back to opening a blank file."""
    import speakinput
    from speakinput import cli

    fake_init = tmp_path / "pkg" / "speakinput" / "__init__.py"
    fake_init.parent.mkdir(parents=True)
    monkeypatch.setattr(speakinput, "__file__", str(fake_init))
    monkeypatch.chdir(tmp_path)  # CWD also has no config.example.toml
    assert cli._example_config_path() is None
