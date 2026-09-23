"""Pitch detection against synthesized signals.

The accuracy tests deliberately use frequencies that fall *between* FFT bins
(bin spacing here is 21.5 Hz). Hitting an on-bin frequency would pass even
without interpolation, so it would not test the thing that matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import GateConfig
from tests.conftest import babble, noise, reading, sine
from whistle.notes import cents_between, note_to_hz
from whistle.pitch import PitchDetector, PitchTracker, measure_noise_floor


@pytest.fixture
def detector(sample_rate, frame_size):
    d = PitchDetector(sample_rate, frame_size)
    d.set_noise_floor(-70.0)
    return d


# --- accuracy ---------------------------------------------------------------

@pytest.mark.parametrize("frequency", [261.63, 440.0, 739.99, 880.0, 1760.0, 2349.32, 3000.0])
def test_detects_note_frequencies_within_five_cents(detector, frequency):
    reading = detector.analyse(sine(frequency))
    assert reading.voiced
    assert abs(cents_between(reading.frequency, frequency)) < 5.0


@pytest.mark.parametrize("frequency", [873.3, 1001.7, 1500.4, 2117.9])
def test_off_bin_frequencies_need_interpolation_and_still_land(detector, frequency):
    bin_width = detector.sample_rate / detector.frame_size
    nearest_bin = round(frequency / bin_width) * bin_width
    assert abs(frequency - nearest_bin) > 4.0, "frequency must be off-bin to be a real test"

    reading = detector.analyse(sine(frequency))
    assert reading.frequency == pytest.approx(frequency, abs=1.5)
    # Interpolation must beat simply naming the nearest bin.
    assert abs(reading.frequency - frequency) < abs(nearest_bin - frequency)


def test_accuracy_is_independent_of_phase(detector):
    for phase in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        reading = detector.analyse(sine(1234.5, phase=phase))
        assert reading.frequency == pytest.approx(1234.5, abs=2.0)


def test_quiet_whistle_is_still_measured_accurately(detector):
    reading = detector.analyse(sine(880.0, amplitude=0.01))
    assert reading.voiced
    assert reading.frequency == pytest.approx(880.0, abs=1.5)


# --- gates ------------------------------------------------------------------

def test_a_whistle_is_tonal_and_broadband_noise_is_not(detector):
    whistle = detector.analyse(sine(880.0))
    room = detector.analyse(noise())

    assert whistle.peak_to_median_db > 30.0
    assert room.peak_to_median_db < detector.gates.peak_to_median_db
    assert whistle.voiced
    assert not room.voiced
    assert room.frequency is None


def test_loud_room_babble_is_rejected(detector):
    """The point of the tonality gate: babble louder than the whistle is still
    rejected, because the gate keys on spectral shape rather than level."""
    room = detector.analyse(babble(amplitude=0.5))
    quiet_whistle = detector.analyse(sine(1200.0, amplitude=0.05))

    assert room.level_db > quiet_whistle.level_db, "babble should be the louder signal"
    assert not room.voiced
    assert quiet_whistle.voiced


def test_whistle_survives_babble_at_moderate_snr(detector):
    mixed = sine(1200.0, amplitude=0.25) + babble(amplitude=0.25)
    reading = detector.analyse(mixed)
    assert reading.voiced
    assert reading.frequency == pytest.approx(1200.0, abs=3.0)


def test_silence_is_not_voiced(detector):
    reading = detector.analyse(np.zeros(detector.frame_size))
    assert not reading.voiced
    assert reading.frequency is None


def test_level_gate_rejects_a_tone_buried_under_the_noise_floor(detector):
    detector.set_noise_floor(-20.0)  # pretend the room is very loud
    reading = detector.analyse(sine(880.0, amplitude=0.001))
    assert reading.peak_to_median_db > detector.gates.peak_to_median_db, "still tonal"
    assert not reading.voiced, "but too quiet to act on"


def test_out_of_band_tones_are_ignored(detector):
    """A 120 Hz hum and a 6 kHz squeal are both outside the whistle band."""
    for frequency in (120.0, 6000.0):
        reading = detector.analyse(sine(frequency, amplitude=0.4))
        assert not reading.voiced or reading.frequency != pytest.approx(frequency, abs=5.0)


def test_speech_fundamental_does_not_masquerade_as_a_whistle(detector):
    reading = detector.analyse(babble(amplitude=0.4, seed=7))
    assert not reading.voiced


# --- housekeeping -----------------------------------------------------------

def test_wrong_frame_length_is_an_error(detector):
    with pytest.raises(ValueError):
        detector.analyse(np.zeros(detector.frame_size + 1))


def test_impossibly_narrow_band_is_an_error(sample_rate, frame_size):
    with pytest.raises(ValueError):
        PitchDetector(sample_rate, frame_size,
                      GateConfig(min_hz=1000.0, max_hz=1005.0))


def test_reading_reports_note_name_and_deviation():
    sharp = reading(445.0)
    assert sharp.note == "A4"
    assert sharp.cents_off == pytest.approx(19.56, abs=0.1)

    silent = reading(None)
    assert silent.note is None and silent.cents_off is None


# --- noise floor ------------------------------------------------------------

def test_noise_floor_tracks_the_room(detector):
    quiet = [noise(amplitude=0.001, seed=s) for s in range(20)]
    loud = [noise(amplitude=0.2, seed=s) for s in range(20)]
    assert measure_noise_floor(loud, detector) > measure_noise_floor(quiet, detector) + 20.0


def test_noise_floor_ignores_a_cough_during_calibration(detector):
    """A median, not a mean: one loud frame must not deafen the gate."""
    frames = [noise(amplitude=0.001, seed=s) for s in range(19)]
    clean = measure_noise_floor(frames, detector)
    frames.append(sine(900.0, amplitude=0.9))
    assert measure_noise_floor(frames, detector) == pytest.approx(clean, abs=1.0)


def test_noise_floor_needs_frames(detector):
    with pytest.raises(ValueError):
        measure_noise_floor([], detector)


# --- tracker ----------------------------------------------------------------

def voiced(frequency: float) -> PitchReading:
    return reading(frequency)


UNVOICED = reading(None)


def test_tracker_waits_for_persistence_before_trusting_a_pitch():
    tracker = PitchTracker(GateConfig(persistence_frames=3))
    assert tracker.update(voiced(880.0)) is None
    assert tracker.update(voiced(880.0)) is None
    assert tracker.update(voiced(880.0)) == pytest.approx(880.0)


def test_tracker_reports_stable_pitch_while_the_whistle_holds():
    tracker = PitchTracker(GateConfig(persistence_frames=2))
    for _ in range(5):
        tracker.update(voiced(880.0))
    assert tracker.stable_frequency == pytest.approx(880.0)


def test_tracker_clears_the_moment_whistling_stops():
    tracker = PitchTracker(GateConfig(persistence_frames=2))
    for _ in range(4):
        tracker.update(voiced(880.0))
    assert tracker.update(UNVOICED) is None
    assert tracker.stable_frequency is None


def test_tracker_median_filters_a_single_octave_jump():
    """One bad frame between good ones must not reach the robot."""
    tracker = PitchTracker(GateConfig(persistence_frames=2, median_window=3))
    for _ in range(4):
        tracker.update(voiced(880.0))
    result = tracker.update(voiced(1760.0))  # a single spurious octave jump
    assert result == pytest.approx(880.0), "median of [880, 880, 1760] is 880"


def test_tracker_follows_a_genuine_pitch_change():
    tracker = PitchTracker(GateConfig(persistence_frames=2, median_window=3))
    for _ in range(4):
        tracker.update(voiced(880.0))
    for _ in range(4):
        result = tracker.update(voiced(1760.0))
    assert result == pytest.approx(1760.0)


# --- the harmonic-comb gates ------------------------------------------------
#
# These exist because peak-to-median alone is not sufficient. Voiced speech
# harmonics that land inside the whistle search band are narrow and tonal, so
# they pass a spikiness test. What distinguishes them is that a whistle is a
# lone peak while a voice is a comb of peaks of comparable height.

def test_a_whistle_is_a_lone_peak_and_a_voice_is_a_comb(detector):
    whistle = detector.analyse(sine(880.0))
    voice = detector.analyse(babble(amplitude=0.5))

    assert whistle.peak_to_second_db > 40.0
    assert voice.peak_to_second_db < detector.gates.peak_to_second_db


def test_harmonic_comb_is_rejected_even_though_it_looks_spiky(detector):
    """A regression test for the real failure this design hit: a 220 Hz voice
    puts a narrow, tonal peak at 440 Hz, inside the whistle band."""
    voice = detector.analyse(babble(amplitude=0.4, seed=7))

    assert voice.peak_to_median_db > detector.gates.peak_to_median_db, \
        "it does pass the spikiness gate, which is the whole problem"
    assert not voice.voiced
    assert voice.reject_reason in {"not a lone peak", "harmonic of a lower tone"}


def test_a_tone_with_a_strong_subharmonic_is_rejected(detector):
    """A peak with energy at f/2 is a harmonic of something lower, not a whistle."""
    combined = sine(440.0, amplitude=0.30) + sine(220.0, amplitude=0.30)
    result = detector.analyse(combined)
    assert result.subharmonic_db < detector.gates.min_subharmonic_db
    assert not result.voiced


def test_a_genuine_whistle_keeps_a_wide_subharmonic_margin(detector):
    """The sub-harmonic gate must not reject real whistles: there is simply
    nothing at f/2 when you whistle."""
    for frequency in (300.0, 880.0, 1760.0, 2349.32):
        assert detector.analyse(sine(frequency)).subharmonic_db > 50.0


@pytest.mark.parametrize(
    "signal, reason",
    [
        (np.zeros(2048), "too quiet"),
        (noise(0.3), "broadband"),
        (babble(amplitude=0.5), "not a lone peak"),
    ],
)
def test_rejections_say_which_gate_failed(detector, signal, reason):
    """The monitor shows this, so you can tell *why* a whistle did not take."""
    assert detector.analyse(signal).reject_reason == reason
