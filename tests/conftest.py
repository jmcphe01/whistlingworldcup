"""Shared synthetic-signal helpers.

No test in this suite touches a microphone, a robot or a broker. Audio is
synthesized, hardware is faked and time is passed in explicitly, so the whole
suite runs anywhere in a couple of seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

SAMPLE_RATE = 44100
FRAME = 2048


@pytest.fixture
def sample_rate() -> int:
    return SAMPLE_RATE


@pytest.fixture
def frame_size() -> int:
    return FRAME


def sine(frequency: float, amplitude: float = 0.3, n: int = FRAME,
         sample_rate: int = SAMPLE_RATE, phase: float = 0.0) -> np.ndarray:
    """A pure tone, which is what a whistle very nearly is."""
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t + phase)).astype(np.float64)


def noise(amplitude: float = 0.3, n: int = FRAME, seed: int = 0) -> np.ndarray:
    """Broadband noise: the loud room, with no whistle in it."""
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(n)).astype(np.float64)


def babble(n: int = FRAME, sample_rate: int = SAMPLE_RATE, seed: int = 1,
           amplitude: float = 0.3) -> np.ndarray:
    """Crude stand-in for a noisy room: several low voiced fundamentals with
    harmonics, plus broadband noise. Deliberately includes energy inside the
    whistle search band so the tonality gate has real work to do."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    signal = np.zeros(n)
    for fundamental in (98.0, 131.0, 165.0, 220.0):
        for harmonic in range(1, 12):
            signal += (rng.uniform(0.3, 1.0) / harmonic) * np.sin(
                2 * np.pi * fundamental * harmonic * t + rng.uniform(0, 2 * np.pi)
            )
    signal += 0.6 * rng.standard_normal(n)
    return (amplitude * signal / np.abs(signal).max()).astype(np.float64)


@pytest.fixture
def make_sine():
    return sine


@pytest.fixture
def make_noise():
    return noise


def reading(frequency: float | None = 880.0, *, voiced: bool | None = None,
            level_db: float = -20.0, peak_to_median_db: float = 40.0,
            peak_to_second_db: float = 50.0, subharmonic_db: float = 80.0,
            noise_floor_db: float = -60.0, reject_reason: str | None = None):
    """Build a PitchReading without pinning tests to its field order."""
    from whistle.pitch import PitchReading

    return PitchReading(
        frequency=frequency,
        level_db=level_db,
        peak_to_median_db=peak_to_median_db,
        peak_to_second_db=peak_to_second_db,
        subharmonic_db=subharmonic_db,
        voiced=(frequency is not None) if voiced is None else voiced,
        noise_floor_db=noise_floor_db,
        reject_reason=reject_reason,
    )
