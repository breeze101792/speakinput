"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
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
    ensure_model,
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
        ),
        default=None,
        help="Override the whisper model from config (default: small)",
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
        help="Override stt.language from config (auto | en | zh). 'auto' "
        "detects per utterance; explicit values skip the language ID pass.",
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
        "-S",
        "--silence-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Skip transcribe when audio RMS is below this floor (0 disables, default: 0.005)",
    )
    parser.add_argument(
        "-P",
        "--initial-prompt",
        type=str,
        default=None,
        metavar="TEXT",
        help='Whisper initial_prompt — bias the decoder toward specific '
        'vocabulary (e.g. names, jargon, acronyms). Empty for no prompt.',
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
    print("Chinese and other languages via stt.language in config.toml.")
    return 0


def _diagnose(config: Config) -> int:
    import time

    import numpy as np

    from speakinput.audio import AudioRecorder
    from speakinput.transcriber import WhisperCppTranscriber

    # Bootstrap the model the same way the live app does, so a misconfigured
    # `stt.model` surfaces here rather than at first transcribe().
    try:
        model_path = ensure_model(config.stt.model)
    except ModelNotFoundError as exc:
        print(f"model error: {exc}", file=sys.stderr)
        return 2
    except ModelDownloadError as exc:
        print(f"model error: {exc}", file=sys.stderr)
        return 2

    print("loading model into memory...", file=sys.stderr)
    transcriber = WhisperCppTranscriber(
        model=model_path,
        language=config.stt.language,
        beam_size=config.stt.beam_size,
        initial_prompt=config.stt.initial_prompt,
    )
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
    if args.silence_threshold is not None:
        config = config.with_overrides(silence_threshold=args.silence_threshold)
    if args.initial_prompt is not None:
        config = config.with_overrides(initial_prompt=args.initial_prompt)

    if args.diagnose:
        return _diagnose(config)

    app = App(
        config,
        dry_run=args.no_inject,
        debug=args.debug,
        config_source=config_source,
    )
    try:
        app.run()
    except KeyboardInterrupt:
        app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
