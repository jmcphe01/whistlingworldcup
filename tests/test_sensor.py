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


# --- the armable monitor ----------------------------------------------------
#
# These run a real background thread against a fake sensor and poll for the
# outcome, so they wait on conditions rather than sleeping a fixed time.

import time as _time

from whistle.sensor import SensorMonitor


def wait_until(condition, seconds=3.0):
    deadline = _time.monotonic() + seconds
    while _time.monotonic() < deadline:
        if condition():
            return True
        _time.sleep(0.005)
    return False


class ScriptedSensor:
    """A sensor whose reading the test can change."""

    def __init__(self, value=20.0):
        self.value = value

    def __call__(self):
        return self.value


@pytest.fixture
def monitor():
    sensor = ScriptedSensor(20.0)
    m = SensorMonitor(sensor, SensorConfig(baseline_seconds=0.05, poll_interval=0.003,
                                           trigger_delta=12.0, persistence_reads=3),
                      log=lambda _: None)
    m.sensor = sensor
    yield m
    m.disarm()


def test_it_measures_a_baseline_then_trips_on_a_sustained_rise(monitor):
    monitor.arm()
    assert wait_until(lambda: monitor.watch.baseline is not None)
    assert monitor.watch.baseline == pytest.approx(20.0)
    assert not monitor.tripped
    monitor.sensor.value = 80.0
    assert wait_until(lambda: monitor.tripped)


def test_a_trip_can_be_acknowledged_and_reported_again(monitor):
    monitor.arm()
    wait_until(lambda: monitor.watch.baseline is not None)
    monitor.sensor.value = 80.0
    assert wait_until(lambda: monitor.tripped)
    monitor.acknowledge()
    assert not monitor.tripped
    assert wait_until(lambda: monitor.tripped), "the thread kept watching"


def test_a_steady_room_never_trips(monitor):
    monitor.arm()
    wait_until(lambda: monitor.watch.baseline is not None)
    _time.sleep(0.1)
    assert not monitor.tripped


def test_disarming_stops_the_thread(monitor):
    monitor.arm()
    assert wait_until(lambda: monitor.armed)
    monitor.disarm()
    assert wait_until(lambda: not monitor.armed)


def test_it_can_be_armed_again_after_being_disarmed(monitor):
    monitor.arm()
    wait_until(lambda: monitor.armed)
    monitor.disarm()
    wait_until(lambda: not monitor.armed)
    monitor.arm()
    assert wait_until(lambda: monitor.armed)


def test_a_failing_read_does_not_kill_the_watch():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise OSError("bluetooth hiccup")
        return 20.0

    m = SensorMonitor(flaky, SensorConfig(baseline_seconds=0.05, poll_interval=0.003),
                      log=lambda _: None)
    m.arm()
    try:
        assert wait_until(lambda: m.watch.baseline is not None)
        assert m.armed
    finally:
        m.disarm()


def test_the_description_says_what_it_sees(monitor):
    assert "not armed" in monitor.describe()
    monitor.arm()
    wait_until(lambda: monitor.watch.baseline is not None)
    assert "baseline" in monitor.describe()
