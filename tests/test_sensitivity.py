"""The sensitivity slider: one knob that scales every gate threshold."""

from __future__ import annotations

import pytest

from config import Config, GateConfig
from tests.conftest import babble, sine
from whistle.pitch import (
    SENSITIVITY_DEFAULT,
    SENSITIVITY_MAX,
    SENSITIVITY_MIN,
    PitchDetector,
    scaled_gates,
    sensitivity_to_scale,
)


@pytest.fixture
def detector():
    d = PitchDetector(44100, 2048, Config().gates)
    d.set_noise_floor(-70.0)
    return d


# --- the mapping ------------------------------------------------------------

def test_the_middle_of_the_slider_is_exactly_what_config_says():
    """So the centre position is always the tuned baseline."""
    assert sensitivity_to_scale(SENSITIVITY_DEFAULT) == pytest.approx(1.0)


def test_more_sensitive_means_lower_thresholds():
    assert sensitivity_to_scale(SENSITIVITY_MAX) < sensitivity_to_scale(SENSITIVITY_DEFAULT)


def test_less_sensitive_means_higher_thresholds():
    assert sensitivity_to_scale(SENSITIVITY_MIN) > sensitivity_to_scale(SENSITIVITY_DEFAULT)


def test_the_mapping_is_monotonic():
    values = [sensitivity_to_scale(s / 2) for s in range(0, 21)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))


@pytest.mark.parametrize("sensitivity", [-5.0, 15.0, 1e9])
def test_out_of_range_values_are_clamped(sensitivity):
    scale = sensitivity_to_scale(sensitivity)
    assert sensitivity_to_scale(SENSITIVITY_MAX) <= scale <= sensitivity_to_scale(SENSITIVITY_MIN)


def test_scaling_moves_every_threshold():
    base = GateConfig()
    tightened = scaled_gates(base, 2.0)
    assert tightened.noise_margin_db == base.noise_margin_db * 2
    assert tightened.peak_to_median_db == base.peak_to_median_db * 2
    assert tightened.peak_to_second_db == base.peak_to_second_db * 2
    assert tightened.min_subharmonic_db == base.min_subharmonic_db * 2


def test_scaling_never_touches_the_search_band():
    """The band edges are baked into the FFT masks when the detector is built,
    so moving them here would silently disagree with the spectrum analysed."""
    base = GateConfig()
    for scale in (0.15, 1.0, 2.0):
        scaled = scaled_gates(base, scale)
        assert (scaled.min_hz, scaled.max_hz) == (base.min_hz, base.max_hz)


# --- the detector -----------------------------------------------------------

def test_the_detector_starts_at_the_default(detector):
    assert detector.sensitivity == SENSITIVITY_DEFAULT
    assert detector.gates == Config().gates


# The signals below were measured rather than guessed: each one sits on a known
# side of the default gates, so the tests pin real behaviour instead of luck.
FAINT = sine(1200.0, amplitude=0.05) + babble(amplitude=0.35, seed=5)   # under default
ORDINARY = sine(1200.0, amplitude=0.10) + babble(amplitude=0.25, seed=5)  # over default


def test_turning_it_up_accepts_a_whistle_the_default_rejects(detector):
    """The whole point of the control: a faint whistle in a loud room gets
    through when you ask for more sensitivity."""
    assert not detector.analyse(FAINT).voiced

    detector.set_sensitivity(SENSITIVITY_MAX)
    assert detector.analyse(FAINT).voiced


def test_turning_it_down_rejects_a_whistle_the_default_accepts(detector):
    """The other direction, for when the room is setting the car off."""
    assert detector.analyse(ORDINARY).voiced

    detector.set_sensitivity(SENSITIVITY_MIN)
    assert not detector.analyse(ORDINARY).voiced


def test_the_slider_is_a_continuum_not_a_switch(detector):
    """Sweeping it should never make a signal harder to hear."""
    accepted = []
    for step in range(0, 21):
        detector.set_sensitivity(step / 2.0)
        accepted.append(detector.analyse(FAINT).voiced)
    assert any(accepted) and not all(accepted)
    assert accepted == sorted(accepted), "acceptance only ever increases"


def test_a_clean_loud_whistle_survives_the_fussiest_setting(detector):
    """Turning sensitivity all the way down must not make the car undriveable."""
    detector.set_sensitivity(SENSITIVITY_MIN)
    assert detector.analyse(sine(880.0, amplitude=0.3)).voiced


def test_silence_is_still_rejected_at_the_most_eager_setting(detector):
    """The top of the slider must not make the car drive itself."""
    import numpy as np

    detector.set_sensitivity(SENSITIVITY_MAX)
    assert not detector.analyse(np.zeros(2048)).voiced


def test_returning_to_the_middle_restores_the_configured_gates(detector):
    """Scaling compounds from the base, never from the last value, so dragging
    back and forth cannot drift."""
    original = detector.gates
    for value in (0.0, 10.0, 3.0, 9.5):
        detector.set_sensitivity(value)
    detector.set_sensitivity(SENSITIVITY_DEFAULT)
    assert detector.gates == original


def test_the_reported_marks_follow_the_slider(detector):
    """The monitor draws these, so they are what makes the slider legible."""
    loose = (detector.set_sensitivity(SENSITIVITY_MAX), detector.gate_thresholds)[1]
    tight = (detector.set_sensitivity(SENSITIVITY_MIN), detector.gate_thresholds)[1]
    assert all(t > l for t, l in zip(tight, loose))


def test_there_is_one_mark_per_gate_bar(detector):
    assert len(detector.gate_thresholds) == 4


def test_the_marks_are_in_the_order_the_panel_draws_them(detector):
    gates = detector.gates
    assert detector.gate_thresholds == (
        gates.noise_margin_db, gates.peak_to_median_db,
        gates.peak_to_second_db, gates.min_subharmonic_db,
    )
