"""Pitch detection and the gates that decide whether a frame is a whistle.

Everything here is pure: buffers in, readings out, no audio device and no state
beyond the smoothing history. That is what lets the tests drive it with
synthesized tones.

Why a plain FFT peak rather than YIN or autocorrelation: a whistle is very close
to a pure sine with almost no harmonic content. The usual pitch trackers exist
to resolve which harmonic is the fundamental, a problem a whistle does not have,
and they cost latency to solve it. An FFT peak refined by parabolic
interpolation is both more accurate here and far cheaper.

Raw bin spacing at 44.1 kHz with a 2048-point window is 21.5 Hz, which is too
coarse -- A5 to A#5 is only 52 Hz. Fitting a parabola through the peak bin and
its two neighbours in the log-magnitude domain recovers the true peak to about
1 Hz on a clean tone, roughly 2 cents.

Three shape gates then decide whether the peak is a whistle at all. Peak-to-
median alone is not enough: voiced speech harmonics landing inside the search
band are narrow and tonal too, and they will pass it. What separates them is
that a whistle is a *lone* peak while a voice is a comb of comparable peaks, so
peak-to-second-peak and a sub-harmonic check carry most of the load. See
`GateConfig` for the measured margins behind each threshold.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from config import GateConfig
from whistle.notes import cents_between, describe

_EPSILON = 1e-12

# Sub-harmonic search tolerance, and the lowest frequency worth looking at.
_SUBHARMONIC_TOLERANCE = 0.03
_SUBHARMONIC_FLOOR_HZ = 60.0


@dataclass(frozen=True)
class PitchReading:
    """One analysis frame, with every gate metric kept for the live monitor."""

    frequency: float | None     # Hz, or None when the frame is not a whistle
    level_db: float             # in-band RMS, dBFS
    peak_to_median_db: float    # is the spectrum spiky, or broadband?
    peak_to_second_db: float    # is it one spike, or a harmonic comb?
    subharmonic_db: float       # how far the peak sits above its own f/2, f/3
    voiced: bool                # did the frame pass every gate
    noise_floor_db: float       # floor the level gate compared against
    reject_reason: str | None = None   # which gate said no, for the monitor

    @property
    def note(self) -> str | None:
        return None if self.frequency is None else describe(self.frequency)[0]

    @property
    def cents_off(self) -> float | None:
        return None if self.frequency is None else describe(self.frequency)[1]


def _to_db(amplitude: float) -> float:
    return 20.0 * math.log10(max(amplitude, _EPSILON))


class PitchDetector:
    """Single-frame analysis. The window and frequency axis are cached; nothing
    else is remembered between frames except the calibrated noise floor."""

    def __init__(self, sample_rate: int, frame_size: int, gates: GateConfig | None = None):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.gates = gates or GateConfig()
        self.noise_floor_db = self.gates.noise_floor_db

        # Hann window: a rectangular window smears a sine across many bins,
        # which flattens exactly the peak the shape gates are measuring.
        self._window = np.hanning(frame_size).astype(np.float64)
        self._window_scale = 2.0 / self._window.sum()
        self._freqs = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)

        band = (self._freqs >= self.gates.min_hz) & (self._freqs <= self.gates.max_hz)
        self._band = band
        self._band_indices = np.flatnonzero(band)
        if self._band_indices.size < 3:
            raise ValueError(
                f"search band {self.gates.min_hz}-{self.gates.max_hz} Hz is narrower "
                f"than 3 bins at {sample_rate} Hz / {frame_size} samples"
            )

    def set_noise_floor(self, level_db: float) -> None:
        """Set the floor the level gate works from (see `measure_noise_floor`)."""
        self.noise_floor_db = level_db

    @property
    def band_frequencies(self) -> np.ndarray:
        """Frequency axis of the search band, for the live monitor."""
        return self._freqs[self._band]

    def analyse(self, samples: np.ndarray) -> PitchReading:
        """Analyse one frame of mono float samples in [-1, 1]."""
        return self.analyse_with_spectrum(samples)[0]

    def analyse_with_spectrum(self, samples: np.ndarray) -> tuple[PitchReading, np.ndarray]:
        """As `analyse`, and also the in-band spectrum in dB for the monitor.

        Same single FFT either way, so drawing the spectrum costs nothing extra.
        """
        frame = np.asarray(samples, dtype=np.float64)
        if frame.size != self.frame_size:
            raise ValueError(f"expected {self.frame_size} samples, got {frame.size}")

        spectrum = np.abs(np.fft.rfft(frame * self._window))
        spectrum_db = 20.0 * np.log10(np.maximum(spectrum, _EPSILON))
        in_band = spectrum[self._band]

        # Window-gain normalised, so level_db reflects the signal's own amplitude
        # rather than being scaled by the window and the transform length.
        scaled = in_band * self._window_scale
        level_db = _to_db(float(np.sqrt(np.sum(scaled ** 2) / 2.0)))

        peak_bin = int(self._band_indices[int(np.argmax(in_band))])
        peak_db = float(spectrum_db[peak_bin])

        peak_to_median_db = peak_db - _to_db(float(np.median(in_band)))
        peak_to_second_db = peak_db - self._second_peak_db(spectrum_db, peak_bin)
        subharmonic_db = peak_db - self._subharmonic_db(spectrum_db, self._freqs[peak_bin])

        reject = self._first_failing_gate(
            level_db, peak_to_median_db, peak_to_second_db, subharmonic_db
        )

        frequency = None
        if reject is None:
            frequency = self._refine(spectrum_db, peak_bin)
            if frequency is None or not (self.gates.min_hz <= frequency <= self.gates.max_hz):
                frequency, reject = None, "out of band"

        reading = PitchReading(
            frequency=frequency,
            level_db=level_db,
            peak_to_median_db=peak_to_median_db,
            peak_to_second_db=peak_to_second_db,
            subharmonic_db=subharmonic_db,
            voiced=reject is None,
            noise_floor_db=self.noise_floor_db,
            reject_reason=reject,
        )
        return reading, spectrum_db[self._band]

    def _first_failing_gate(self, level_db: float, peak_to_median_db: float,
                            peak_to_second_db: float, subharmonic_db: float) -> str | None:
        if level_db < self.noise_floor_db + self.gates.noise_margin_db:
            return "too quiet"
        if peak_to_median_db < self.gates.peak_to_median_db:
            return "broadband"
        if peak_to_second_db < self.gates.peak_to_second_db:
            return "not a lone peak"
        if subharmonic_db < self.gates.min_subharmonic_db:
            return "harmonic of a lower tone"
        return None

    def _second_peak_db(self, spectrum_db: np.ndarray, peak_bin: int) -> float:
        """Loudest in-band bin outside a narrow exclusion zone around the peak.

        The zone has to be wide enough to clear the Hann window's own skirt,
        or the peak's shoulder would be mistaken for a second peak.
        """
        skirt = max(4, int(round(peak_bin * (2.0 ** (100.0 / 1200.0) - 1.0))))
        mask = self._band.copy()
        mask[max(0, peak_bin - skirt):peak_bin + skirt + 1] = False
        if not mask.any():
            return -np.inf
        return float(spectrum_db[mask].max())

    def _subharmonic_db(self, spectrum_db: np.ndarray, peak_hz: float) -> float:
        """Loudest energy near f/2 or f/3, i.e. evidence the peak is a harmonic.

        Returns -inf when there is nothing to check, which makes the gate a
        no-op rather than a rejection.
        """
        best = -np.inf
        for divisor in (2.0, 3.0):
            target = peak_hz / divisor
            if target < _SUBHARMONIC_FLOOR_HZ:
                continue
            window = (
                (self._freqs >= target * (1.0 - _SUBHARMONIC_TOLERANCE))
                & (self._freqs <= target * (1.0 + _SUBHARMONIC_TOLERANCE))
            )
            if window.any():
                best = max(best, float(spectrum_db[window].max()))
        return best

    def _refine(self, spectrum_db: np.ndarray, peak: int) -> float | None:
        """Parabolic interpolation over log magnitudes around the peak bin."""
        if peak <= 0 or peak >= spectrum_db.size - 1:
            return None

        left, centre, right = (float(spectrum_db[peak + o]) for o in (-1, 0, 1))
        denominator = left - 2.0 * centre + right
        if denominator == 0:
            offset = 0.0
        else:
            offset = 0.5 * (left - right) / denominator
            offset = max(-0.5, min(0.5, offset))  # a real peak never moves a whole bin

        return (peak + offset) * self.sample_rate / self.frame_size


def measure_noise_floor(frames: list[np.ndarray], detector: PitchDetector,
                        percentile: float = 50.0) -> float:
    """In-band level of the quiet room, in dBFS.

    Taking a percentile rather than a mean keeps a cough or a slammed door during
    calibration from lifting the floor and deafening the gate for the whole match.
    """
    if not frames:
        raise ValueError("need at least one frame to measure the noise floor")
    levels = [detector.analyse(frame).level_db for frame in frames]
    return float(np.percentile(levels, percentile))


class PitchTracker:
    """Smooths a stream of readings into a trustworthy pitch.

    Two jobs: a median filter over the last few frames, which removes the
    single-frame outliers that FFT peak-picking occasionally produces, and a
    persistence requirement, so a pitch has to hold steady for several frames
    before the robot acts on it. Both trade a few milliseconds of latency for not
    lurching at every stray transient.
    """

    def __init__(self, gates: GateConfig | None = None):
        self.gates = gates or GateConfig()
        self._history: deque[float] = deque(maxlen=max(1, self.gates.median_window))
        self._agreeing = 0
        self._stable: float | None = None

    @property
    def stable_frequency(self) -> float | None:
        """The pitch that has survived smoothing, or None if not whistling."""
        return self._stable

    def reset(self) -> None:
        self._history.clear()
        self._agreeing = 0
        self._stable = None

    def update(self, reading: PitchReading) -> float | None:
        """Feed one reading; return the smoothed pitch, or None."""
        if not reading.voiced or reading.frequency is None:
            self.reset()
            return None

        self._history.append(reading.frequency)
        smoothed = float(np.median(self._history))

        if self._stable is not None and \
                abs(cents_between(smoothed, self._stable)) > self.gates.agreement_cents:
            # A jump this large restarts the count rather than being followed.
            self._agreeing = 0
            self._stable = None

        self._agreeing += 1
        if self._agreeing >= self.gates.persistence_frames:
            self._stable = smoothed
        elif self._stable is not None:
            self._stable = smoothed
        return self._stable
