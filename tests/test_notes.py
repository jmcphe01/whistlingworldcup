import math

import pytest

from whistle.notes import (
    cents_above_a4,
    cents_between,
    describe,
    hz_to_midi,
    midi_to_hz,
    note_to_hz,
    note_to_midi,
)


@pytest.mark.parametrize(
    "name, midi",
    [("C4", 60), ("A4", 69), ("F#5", 78), ("A6", 93), ("D7", 98), ("Bb3", 58), ("c4", 60)],
)
def test_note_to_midi(name, midi):
    assert note_to_midi(name) == midi


@pytest.mark.parametrize(
    "name, hz",
    [("A4", 440.0), ("C4", 261.626), ("F#5", 739.989), ("A6", 1760.0), ("D7", 2349.318)],
)
def test_note_to_hz(name, hz):
    assert note_to_hz(name) == pytest.approx(hz, abs=0.01)


@pytest.mark.parametrize("name", ["", "H4", "A", "Az", "4A"])
def test_bad_note_names_are_rejected(name):
    with pytest.raises(ValueError):
        note_to_midi(name)


def test_midi_and_hz_round_trip():
    for midi in range(48, 108):
        assert hz_to_midi(midi_to_hz(midi)) == pytest.approx(midi)


def test_an_octave_is_1200_cents():
    assert cents_between(880.0, 440.0) == pytest.approx(1200.0)
    assert cents_between(220.0, 440.0) == pytest.approx(-1200.0)


def test_a_semitone_is_100_cents():
    assert cents_between(note_to_hz("A#4"), note_to_hz("A4")) == pytest.approx(100.0)


def test_cents_above_a4_is_zero_at_a4():
    assert cents_above_a4(440.0) == pytest.approx(0.0)


def test_describe_names_the_note_and_the_deviation():
    name, cents = describe(440.0)
    assert (name, cents) == ("A4", pytest.approx(0.0))

    name, cents = describe(445.0)
    assert name == "A4"
    assert cents == pytest.approx(19.56, abs=0.1)  # 5 Hz sharp at A4

    # Past the halfway point it should name the neighbour, flat, not A4 sharp.
    name, cents = describe(note_to_hz("A4") * 2 ** (0.6 / 12))
    assert name == "A#4"
    assert cents < 0


def test_describe_crosses_octaves_correctly():
    assert describe(note_to_hz("C5"))[0] == "C5"
    assert describe(note_to_hz("B4"))[0] == "B4"


@pytest.mark.parametrize("hz", [0.0, -1.0])
def test_non_positive_frequencies_are_rejected(hz):
    with pytest.raises(ValueError):
        hz_to_midi(hz)
    with pytest.raises(ValueError):
        cents_between(hz, 440.0)
