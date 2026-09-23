"""Throttle zones and their hysteresis."""

from __future__ import annotations

import pytest

from config import ThrottleConfig
from whistle.commands import Drive
from whistle.notes import note_to_hz
from whistle.throttle import ThrottleMapper, build_bands


@pytest.fixture
def mapper():
    return ThrottleMapper()


@pytest.mark.parametrize(
    "note, expected",
    [
        ("C4", Drive.BACKWARD),        # the lowest note Jake can whistle
        ("A4", Drive.BACKWARD),
        ("F5", Drive.BACKWARD),
        ("G5", Drive.BACKWARD),
        ("C6", Drive.FORWARD),
        ("A5", Drive.FORWARD),
        ("A#6", Drive.FORWARD_FAST),
        ("D7", Drive.FORWARD_FAST),    # the "go fast" target note
    ],
)
def test_notes_land_in_the_intended_zone(mapper, note, expected):
    assert mapper.update(note_to_hz(note)) is expected


def test_silence_stops_the_car(mapper):
    mapper.update(note_to_hz("A5"))
    assert mapper.update(None) is Drive.STOP


def test_stopping_clears_the_active_band(mapper):
    """So the next whistle is judged fresh rather than against a stale band."""
    mapper.update(note_to_hz("A5"))
    mapper.update(None)
    assert mapper.current_drive is Drive.STOP
    assert mapper.update(note_to_hz("C4")) is Drive.BACKWARD


def test_the_three_zones_cover_the_whole_axis():
    bands = build_bands(ThrottleConfig())
    assert bands[0].low_hz == float("-inf")
    assert bands[-1].high_hz == float("inf")
    for lower, upper in zip(bands, bands[1:]):
        assert lower.high_hz == upper.low_hz, "no gaps and no overlaps"


def test_zone_order_is_validated():
    with pytest.raises(ValueError):
        build_bands(ThrottleConfig(backward_top="A6", forward_top="F#5"))


# --- hysteresis -------------------------------------------------------------

def test_a_pitch_sitting_on_a_boundary_does_not_chatter(mapper):
    """The failure this prevents: the car flipping between two commands several
    times a second while you hold a note near a zone edge."""
    boundary = note_to_hz("A5")
    mapper.update(boundary * 0.97)
    assert mapper.current_drive is Drive.BACKWARD

    commands = {mapper.update(boundary * factor)
                for factor in (0.999, 1.001, 0.998, 1.002, 1.0)}
    assert commands == {Drive.BACKWARD}, "wobbling across the edge changed nothing"


def test_a_decisive_move_past_the_boundary_does_switch(mapper):
    boundary = note_to_hz("A5")
    mapper.update(boundary * 0.97)
    assert mapper.update(boundary * 1.10) is Drive.FORWARD


def test_hysteresis_applies_in_both_directions(mapper):
    boundary = note_to_hz("A6")
    mapper.update(boundary * 1.05)
    assert mapper.current_drive is Drive.FORWARD_FAST
    assert mapper.update(boundary * 0.995) is Drive.FORWARD_FAST, "held by hysteresis"
    assert mapper.update(boundary * 0.90) is Drive.FORWARD


def test_the_hysteresis_margin_is_honoured_exactly():
    mapper = ThrottleMapper(ThrottleConfig(hysteresis_cents=100.0))
    boundary = note_to_hz("A5")
    mapper.update(boundary * 0.9)

    just_inside = boundary * 2 ** (99.0 / 1200.0)
    just_outside = boundary * 2 ** (101.0 / 1200.0)
    assert mapper.update(just_inside) is Drive.BACKWARD
    assert mapper.update(just_outside) is Drive.FORWARD


def test_zero_hysteresis_switches_at_the_edge():
    mapper = ThrottleMapper(ThrottleConfig(hysteresis_cents=0.0))
    boundary = note_to_hz("A5")
    assert mapper.update(boundary - 0.5) is Drive.BACKWARD
    assert mapper.update(boundary + 0.5) is Drive.FORWARD
