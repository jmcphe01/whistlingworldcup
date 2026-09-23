"""Proximity detection on synthetic reflection readings."""

from __future__ import annotations

import pytest

from config import SensorConfig
from whistle.sensor import ProximityWatch


@pytest.fixture
def watch():
    w = ProximityWatch(SensorConfig(trigger_delta=12.0, persistence_reads=5))
    w.set_baseline(30.0)
    return w


def feed(watch, values):
    return [watch.update(value) for value in values]


def test_the_goalie_arriving_triggers(watch):
    assert any(feed(watch, [80.0] * 5))


def test_it_takes_several_reads_to_confirm(watch):
    """Persistence: a quarter of a second, in exchange for immunity to spikes."""
    assert feed(watch, [80.0] * 4) == [False] * 4
    assert watch.update(80.0) is True


def test_a_single_bright_spike_is_ignored(watch):
    """A camera flash or a passing shadow must not end the match."""
    assert not any(feed(watch, [30.0, 31.0, 95.0, 30.0, 29.0, 31.0, 30.0]))


def test_an_interrupted_rise_restarts_the_count(watch):
    assert not any(feed(watch, [80.0, 80.0, 80.0, 80.0, 30.0, 80.0, 80.0]))


def test_a_steady_room_never_triggers(watch):
    assert not any(feed(watch, [30.0, 31.0, 29.0, 30.5, 28.0] * 20))


def test_readings_just_under_the_threshold_do_not_trigger(watch):
    assert not any(feed(watch, [41.9] * 20))


def test_readings_at_the_threshold_do_trigger(watch):
    assert any(feed(watch, [42.0] * 5))


def test_it_triggers_only_once(watch):
    """So the caller cannot end the match twice."""
    results = feed(watch, [80.0] * 20)
    assert sum(results) == 1
    assert watch.triggered


def test_reset_allows_another_match(watch):
    feed(watch, [80.0] * 5)
    watch.reset()
    assert any(feed(watch, [80.0] * 5))


# --- baseline ---------------------------------------------------------------

def test_the_baseline_is_the_median_of_the_samples():
    watch = ProximityWatch()
    for value in [20.0, 21.0, 19.0, 20.5, 20.0]:
        watch.add_baseline_sample(value)
    assert watch.finish_baseline() == pytest.approx(20.0)


def test_one_bad_baseline_read_cannot_skew_the_threshold():
    """Median, not mean: if the goalie leans over the sensor during calibration
    the threshold must not be lifted out of reach for the whole match."""
    watch = ProximityWatch()
    for value in [20.0, 21.0, 19.0, 20.0, 20.0, 99.0]:
        watch.add_baseline_sample(value)
    assert watch.finish_baseline() < 25.0


def test_the_threshold_follows_the_baseline():
    """A dark venue and a bright one both work, which an absolute threshold
    would not."""
    for baseline in (5.0, 30.0, 70.0):
        watch = ProximityWatch(SensorConfig(trigger_delta=12.0))
        watch.set_baseline(baseline)
        assert watch.threshold == pytest.approx(baseline + 12.0)


def test_an_empty_baseline_is_an_error():
    with pytest.raises(ValueError):
        ProximityWatch().finish_baseline()


def test_updating_without_a_baseline_is_an_error():
    with pytest.raises(RuntimeError):
        ProximityWatch().update(50.0)
