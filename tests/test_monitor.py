"""Monitor helpers. Nothing here opens a window."""

from __future__ import annotations

import math

import numpy as np
import pytest

from monitor import Snapshot, TrackHistory, pool_centres, pool_max


def test_pooling_keeps_the_peak():
    """Max-pooling, not averaging: the panel exists to show peak height, and
    averaging is precisely the operation that would hide a narrow whistle."""
    spectrum = np.full(100, -90.0)
    spectrum[42] = -10.0
    assert pool_max(spectrum, 10).max() == -10.0


def test_averaging_would_have_lost_it():
    spectrum = np.full(100, -90.0)
    spectrum[42] = -10.0
    assert spectrum.reshape(10, 10).mean(axis=1).max() < -80.0


def test_pooling_returns_the_requested_size():
    assert pool_max(np.arange(1000.0), 250).size == 250


def test_pooling_a_short_spectrum_leaves_it_alone():
    values = np.arange(5.0)
    assert np.array_equal(pool_max(values, 400), values)


def test_the_axis_pools_to_match():
    spectrum = np.arange(1000.0)
    assert pool_centres(spectrum, 250).size == pool_max(spectrum, 250).size


def test_pooled_axis_stays_ordered_and_in_range():
    axis = pool_centres(np.linspace(240.0, 4200.0, 1000), 250)
    assert axis[0] >= 240.0 and axis[-1] <= 4200.0
    assert np.all(np.diff(axis) > 0)


def test_zero_bins_is_an_error():
    with pytest.raises(ValueError):
        pool_max(np.arange(10.0), 0)


# --- history ----------------------------------------------------------------

def test_history_forgets_what_scrolled_off():
    history = TrackHistory(seconds=1.0)
    for step in range(200):
        history.add(step * 0.02, 880.0)
    assert history.times[0] >= history.times[-1] - 1.0001
    assert len(history.times) < 60


def test_silences_break_the_line_rather_than_bridging_it():
    """A straight line drawn across a pause would read as a glide that never
    happened."""
    history = TrackHistory()
    history.add(0.0, 880.0)
    history.add(0.1, None)
    history.add(0.2, 880.0)
    assert math.isnan(history.pitches[1])
    assert not math.isnan(history.pitches[0])


def test_history_starts_empty():
    assert TrackHistory().times == []


# --- the snapshot -----------------------------------------------------------

def test_a_snapshot_survives_pickling():
    """It crosses a process boundary, so this is a real requirement."""
    import pickle

    snapshot = Snapshot(
        t=1.5, spectrum_db=np.linspace(-90, -10, 128), frequency=880.0, note="A5",
        cents_off=3.2, level_db=-22.0, noise_floor_db=-60.0, peak_to_median_db=40.0,
        peak_to_second_db=50.0, subharmonic_db=80.0, reject_reason=None,
        drive="forward", steering=False, phase="running", role="ball",
    )
    restored = pickle.loads(pickle.dumps(snapshot))
    assert restored.note == "A5"
    assert np.array_equal(restored.spectrum_db, snapshot.spectrum_db)
