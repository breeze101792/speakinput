"""Synthetic audio signals shared by the test modules.

Speech has a 10-20dB envelope swing between syllables; noise does not.
The speech gate keys on that difference, so tests that want a buffer
treated as speech must use a modulated signal, not a constant tone.
"""

import numpy as np

SAMPLE_RATE = 16000

# Envelope period. 100ms on / 100ms off is a syllable-rate swing and
# gives runs of five 20ms frames, long enough for the gate's sustained-
# run requirement to see them as speech rather than as transients.
_ENVELOPE_HZ = 5.0
_TONE_HZ = 220.0


def speech_like(
    duration_ms: int,
    level: float = 0.2,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Tone burst at a speech-like level with a syllabic envelope."""
    n = int(sample_rate * duration_ms / 1000)
    t = np.arange(n) / sample_rate
    tone = np.sin(2 * np.pi * _TONE_HZ * t)
    envelope = (np.sin(2 * np.pi * _ENVELOPE_HZ * t) > 0) * 0.8 + 0.2
    return (tone * envelope * level).astype(np.float32)


def room_tone(
    duration_ms: int | None = None,
    level: float = 0.02,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 1234,
    *,
    samples: int | None = None,
) -> np.ndarray:
    """Stationary white noise — stands in for a fan or a noisy room.

    Pass `samples` instead of `duration_ms` when the test needs a
    frame-aligned length.
    """
    if samples is None:
        if duration_ms is None:
            raise ValueError("pass duration_ms or samples")
        samples = int(sample_rate * duration_ms / 1000)
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(samples) * level).astype(np.float32)
