"""Song rendering. Pure buffer maths -- nothing here opens an audio device."""

from __future__ import annotations

import numpy as np
import pytest

from whistle.notes import note_to_hz
from whistle.songs import (
    MELODIES,
    Note,
    Song,
    duration,
    render,
    render_note,
    render_song,
)

SAMPLE_RATE = 44100


def dominant_frequency(buffer, sample_rate=SAMPLE_RATE):
    spectrum = np.abs(np.fft.rfft(buffer * np.hanning(buffer.size)))
    return float(np.fft.rfftfreq(buffer.size, 1 / sample_rate)[np.argmax(spectrum)])


def test_a_rendered_note_has_the_right_pitch():
    buffer = render_note(Note("A4", 0.5))
    assert dominant_frequency(buffer) == pytest.approx(440.0, abs=3.0)


@pytest.mark.parametrize("name", ["C4", "A4", "C5", "G5", "C6"])
def test_every_note_in_range_renders_at_its_own_pitch(name):
    buffer = render_note(Note(name, 0.4))
    assert dominant_frequency(buffer) == pytest.approx(note_to_hz(name), rel=0.02)


def test_a_rendered_note_has_the_right_length():
    assert render_note(Note("A4", 0.25)).size == pytest.approx(0.25 * SAMPLE_RATE, abs=2)


def test_a_rest_is_silence():
    assert not render_note(Note(None, 0.2)).any()


def test_notes_start_and_end_at_silence():
    """The envelope exists because a hard edge clicks, and on a laptop speaker
    the click is louder than the note."""
    buffer = render_note(Note("A4", 0.3))
    assert abs(buffer[0]) < 0.01
    assert abs(buffer[-1]) < 0.01


def test_output_stays_inside_the_usable_range():
    """Harmonics are summed, so the result has to be checked for clipping."""
    for song in Song:
        assert np.abs(render_song(song)).max() <= 1.0


def test_rendering_respects_amplitude():
    quiet = np.abs(render_note(Note("A4", 0.2), amplitude=0.1)).max()
    loud = np.abs(render_note(Note("A4", 0.2), amplitude=0.4)).max()
    assert loud > quiet * 3


@pytest.mark.parametrize("song", list(Song))
def test_each_song_renders_to_its_stated_length(song):
    expected = duration(MELODIES[song]) * SAMPLE_RATE
    assert render_song(song).size == pytest.approx(expected, rel=0.001)


@pytest.mark.parametrize("song", list(Song))
def test_each_song_is_short_enough_to_sit_through(song):
    assert 0.5 < duration(MELODIES[song]) < 3.0


def test_there_is_a_song_for_every_outcome():
    assert set(MELODIES) == set(Song)


def test_the_death_song_falls_and_the_victory_song_rises():
    """They have to be tellable apart from across a noisy room."""
    def pitches(song):
        return [n.frequency for n in MELODIES[song] if n.frequency is not None]

    death = pitches(Song.DEATH)
    victory = pitches(Song.VICTORY)
    assert death[0] > death[-1]
    assert victory[0] < victory[-1]


def test_an_empty_melody_renders_to_nothing():
    assert render([]).size == 0


def test_a_melody_is_the_concatenation_of_its_notes():
    melody = (Note("C5", 0.1), Note(None, 0.1), Note("G5", 0.1))
    assert render(melody).size == render_note(melody[0]).size * 3
