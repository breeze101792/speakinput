"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from speakinput.app import App
from speakinput.audio import list_input_devices
from speakinput.config import (
    Config,
    default_config_path,
    load_config,
)
from speakinput.models import (
    ModelDownloadError,
    ModelNotFoundError,
)
from speakinput.singleinstance import acquire as acquire_instance_lock


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speakinput",
        description="Push-to-talk voice transcription typed into the focused field.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help=f"Path to config.toml (default: {default_config_path()})",
    )
    parser.add_argument(
        "-C",
        "--edit-config",
        action="store_true",
        help=(
            f"Open the config file in $VISUAL (else $EDITOR, else vi) and exit. "
            f"Uses --config if given, otherwise {default_config_path()}. "
            f"Seeds the file from config.example.toml if it doesn't exist yet."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=(
            "tiny.en",
            "base.en",
            "small.en",
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
        ),
        default=None,
        help="Override the primary profile's whisper model (default: small)",
    )
    parser.add_argument(
        "-l",
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "-L",
        "--list-models",
        action="store_true",
        help="List available whisper models and exit",
    )
    parser.add_argument(
        "-g",
        "--language",
        choices=("auto", "en", "zh"),
        default=None,
        help="Override the primary profile's language (auto | en | zh). "
        "'auto' detects per utterance; explicit values skip the language "
        "ID pass. Secondary profile's language is configured in config.toml.",
    )
    parser.add_argument(
        "-D",
        "--diagnose",
        action="store_true",
        help="Record for 2s and print the audio RMS without injecting text",
    )
    parser.add_argument(
        "-n",
        "--no-inject",
        action="store_true",
        help="In dry-run mode, print transcribed text to stderr instead of typing it",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Log every key event and transcript to stderr (useful for permission issues)",
    )
    parser.add_argument(
        "-t",
        "--trailing-space",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append a single space after each transcript (default: on; override with -T / --no-trailing-space)",
    )
    # Register the short form of the negation. BooleanOptionalAction only
    # attaches a short flag to the positive form; we mirror it here.
    parser.add_argument(
        "-T",
        action="store_false",
        dest="trailing_space",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=None,
        help="Force GPU acceleration on. If the pywhispercpp wheel is "
        "CPU-only, logs a warning and falls back to CPU. See README "
        "→ 'GPU acceleration' for the wheel rebuild instructions.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="use_gpu",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=None,
        metavar="N",
        help="GPU device index when multiple GPUs are present (default: 0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="Number of CPU threads for the CPU path (default: 0 = auto, "
        "min(4, hardware_concurrency()))",
    )
    parser.add_argument(
        "-S",
        "--silence-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Skip transcribe when audio RMS is below this floor (0 disables, default: 0.005)",
    )
    parser.add_argument(
        "-A",
        "--auto-stop-seconds",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Auto-stop after this many seconds of consecutive silence while the "
        "key is held (0 disables, default: 0.8). Trailing silence is also "
        "trimmed from the buffer before transcribe.",
    )
    parser.add_argument(
        "-P",
        "--initial-prompt",
        type=str,
        default=None,
        metavar="TEXT",
        help='Override the primary profile\'s whisper initial_prompt — '
        'bias the decoder toward specific vocabulary (e.g. names, jargon, '
        'acronyms). Empty for no prompt. Secondary profile is configured '
        'via config.toml.',
    )
    parser.add_argument(
        "--pause-media",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pause playing media on hotkey press, resume on release "
        "(default: on; override with --no-pause-media)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def _list_devices() -> int:
    try:
        devices = list_input_devices()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not devices:
        print("no input devices found", file=sys.stderr)
        return 1
    print(f"{'idx':>4}  {'channels':>8}  {'rate':>8}  name")
    for d in devices:
        print(
            f"{d['index']:>4}  {d['max_input_channels']:>8}  "
            f"{d['default_samplerate']:>8.0f}  {d['name']}"
        )
    return 0


def _list_models() -> int:
    """List models curated for v1. Full pywhispercpp list is also accepted via
    a path to a .bin file, but the curated set is what we recommend."""
    from speakinput.config import VALID_MODELS

    print("curated models (default whitelist):")
    for name in VALID_MODELS:
        suffix = "  # English-only" if name.endswith(".en") else "  # multilingual"
        print(f"  {name}{suffix}")
    print()
    print("English-only models are faster. Multilingual models support")
    print("Chinese and other languages via profile.*.language in config.toml.")
    return 0


def _example_config_path() -> Path | None:
    """Locate the bundled config.example.toml.

    Resolves in this order:
      1. Relative to the installed `speakinput` package (editable install
         puts the example at the repo root, two levels above `__init__.py`).
      2. CWD-relative — useful for `python -m speakinput` runs from the
         repo checkout.

    Returns None if neither exists; the caller surfaces a clear error.
    """
    try:
        import speakinput

        pkg = Path(speakinput.__file__).resolve().parent
        candidates = [pkg.parent.parent / "config.example.toml", pkg / "config.example.toml"]
    except Exception:
        candidates = []
    candidates.append(Path.cwd() / "config.example.toml")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _resolve_config_target(arg: Path | None) -> Path:
    """Pick which config file `-C` should open.

    Honors an explicit `--config` if given; otherwise uses
    `default_config_path()`. The directory is created on demand so the
    first run of `-C` doesn't fail with "No such file or directory".
    """
    target = arg if arg is not None else default_config_path()
    target = target.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _seed_example(target: Path) -> None:
    """Copy the bundled example config to `target` if it's missing.

    Mirrors what start.sh does on first run, so `-C` "just works" the
    first time a user wants to customize. Skips silently if the
    example can't be located (pip installs without package data), so
    `-C` on a missing config still tries to open a blank file.
    """
    if target.exists():
        return
    example = _example_config_path()
    if example is None:
        # No example bundled — fall through; the editor will open an
        # empty file. Better than hard-failing for a missing data file.
        return
    shutil.copyfile(example, target)


def _edit_config(target: Path | None) -> int:
    """Open the resolved config in the user's editor and return 0/non-zero.

    Editor selection matches the Unix convention: `$VISUAL` (GUI editor
    like VS Code / Sublime) → `$EDITOR` (terminal editor) → `vi` as a
    last resort. We launch via subprocess.run(check=True) so an editor
    that can't be found (ENOENT) or returns non-zero bubbles up as a
    proper exit code rather than masquerading as success.
    """
    path = _resolve_config_target(target)
    _seed_example(path)
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    print(f"opening {path} in {editor}", file=sys.stderr, flush=True)
    try:
        result = subprocess.run([editor, str(path)], check=False)
    except FileNotFoundError:
        print(
            f"error: editor {editor!r} not found on PATH "
            f"(set $VISUAL or $EDITOR to your editor of choice)",
            file=sys.stderr,
        )
        return 1
    return result.returncode


def _diagnose(config: Config) -> int:
    import time

    import numpy as np

    from speakinput.app import _build_transcribers
    from speakinput.audio import AudioRecorder

    # Bootstrap the primary profile's model the same way the live app
    # does, so a misconfigured `profile.primary.model` surfaces here
    # rather than at first transcribe(). We build transcribers for all
    # profiles so a secondary-only misconfiguration also surfaces (the
    # `_build_transcribers` helper applies the same auto-upgrade path).
    try:
        transcribers = _build_transcribers(
            [config.primary] + ([config.secondary] if config.secondary else [])
        )
    except ModelNotFoundError as exc:
        print(f"model error: {exc}", file=sys.stderr)
        return 2
    except ModelDownloadError as exc:
        print(f"model error: {exc}", file=sys.stderr)
        return 2

    print("loading model into memory...", file=sys.stderr)
    transcriber = transcribers[config.primary.key]
    # Force a warmup pass so the user can see if the model actually works.
    transcriber.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    print("model loaded", file=sys.stderr)

    print("recording 2s of audio...", file=sys.stderr)
    recorder = AudioRecorder(
        sample_rate=config.audio.sample_rate,
        device=config.audio.device,
    )
    recorder.start()
    time.sleep(2.0)
    audio = recorder.stop()
    if audio.size == 0:
        print("no audio captured (device=)", file=sys.stderr)
        return 1
    rms = float(np.sqrt(np.mean(audio * audio)))
    print(
        f"captured {audio.size} samples ({audio.size / config.audio.sample_rate:.2f}s) rms={rms:.4f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose or args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args.list_devices:
        return _list_devices()
    if args.list_models:
        return _list_models()
    # `-C` is an independent action — it must NOT acquire the single-
    # instance lock (we're not running the app) and must NOT load the
    # config (we're about to open it in an editor, not parse it).
    # Short-circuit ahead of both.
    if args.edit_config:
        return _edit_config(args.config)

    # Acquire the single-instance lock before any heavy work. If another
    # speakinput is already running, this exits 3 immediately. The fd is
    # intentionally held for the rest of the process lifetime so the OS
    # releases the lock when we exit (or crash).
    acquire_instance_lock()

    try:
        config, config_source = load_config(args.config)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    if args.model:
        config = config.with_overrides(model=args.model)
    if args.language:
        config = config.with_overrides(language=args.language)
    if args.trailing_space is not None:
        config = config.with_overrides(trailing_space=args.trailing_space)
    if args.pause_media is not None:
        config = config.with_overrides(pause_media=args.pause_media)
    if args.silence_threshold is not None:
        config = config.with_overrides(silence_threshold=args.silence_threshold)
    if args.auto_stop_seconds is not None:
        config = config.with_overrides(auto_stop_seconds=args.auto_stop_seconds)
    if args.initial_prompt is not None:
        config = config.with_overrides(initial_prompt=args.initial_prompt)
    if args.use_gpu is not None:
        config = config.with_overrides(use_gpu=args.use_gpu)
    if args.gpu_device is not None:
        config = config.with_overrides(gpu_device=args.gpu_device)
    if args.threads is not None:
        config = config.with_overrides(n_threads=args.threads)

    if args.diagnose:
        return _diagnose(config)

    app = App(
        config,
        dry_run=args.no_inject,
        debug=args.debug,
        config_source=config_source,
    )
    # `app.run()` installs its own SIGINT handler that sets the
    # shutdown event and lets the `finally` block in `run()` call
    # `app.shutdown()`. Python's `signal.signal` replaces the
    # default SIGINT handler that would have raised
    # `KeyboardInterrupt`, so a plain `except KeyboardInterrupt`
    # here would never fire — it just looked like a second layer
    # of defense. The previous shape is replaced with a direct
    # call so the program's behavior on Ctrl-C is documented in
    # exactly one place (app.py:run()).
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
