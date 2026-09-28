"""Song rendering. Pure buffer maths -- nothing here opens an audio device."""

from __future__ import annotations

import math

import numpy as np
import pytest

from whistle.notes import note_to_hz
from whistle.songs import (
    MELODIES,
    BeepPlayer,
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
    assert 0.5 < duration(MELODIES[song]) < 6.0


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


# --- the songs themselves -----------------------------------------------------

def names(song):
    return [n.name for n in MELODIES[song] if n.name is not None]


def test_the_winning_song_opens_with_the_chorus_hook():
    """Scale degrees 1 2 4 2 6 6 5, then 1 2 4 2 5 5 4, in C major."""
    assert names(Song.VICTORY) == [
        "C5", "D5", "F5", "D5", "A5", "A5", "G5",
        "C5", "D5", "F5", "D5", "G5", "G5", "F5",
    ]


def test_the_losing_song_is_four_descending_half_steps():
    pitches = [n.frequency for n in MELODIES[Song.DEATH] if n.frequency is not None]
    assert len(pitches) == 4
    for higher, lower in zip(pitches, pitches[1:]):
        assert 1200 * math.log2(higher / lower) == pytest.approx(100.0, abs=0.5)


def test_the_last_womp_is_held():
    womps = [n for n in MELODIES[Song.DEATH] if n.name is not None]
    assert womps[-1].seconds > 2 * womps[0].seconds


def test_every_note_fits_the_beepers_range():
    for song in Song:
        for note in MELODIES[song]:
            if note.frequency is not None:
                assert 0 < note.frequency <= BeepPlayer.MAX_HZ


# --- playing on the robot's beeper --------------------------------------------

class FakeBeeper:
    """Records beeps, and advances a fake clock when the player sleeps."""

    def __init__(self):
        self.events = []
        self.fail = False

    def beep(self, *, pattern, frequency, blocking):
        if self.fail:
            raise OSError("bluetooth hiccup")
        self.events.append(("beep", frequency, blocking))

    def stop_beep(self, *, blocking):
        self.events.append(("stop", blocking))


def play(song, sustain=0.3):
    robot, slept = FakeBeeper(), []
    player = BeepPlayer(robot, sustain, sleep=slept.append, log=lambda _: None)
    player.play(song)
    return robot, slept, player


def beeps(robot):
    return [e[1] for e in robot.events if e[0] == "beep"]


def test_the_victory_song_beeps_its_notes_in_order():
    robot, _, _ = play(Song.VICTORY, sustain=10.0)      # no retriggering
    assert beeps(robot) == [round(note_to_hz(n)) for n in names(Song.VICTORY)]


def test_the_death_song_beeps_four_falling_notes():
    robot, _, _ = play(Song.DEATH, sustain=10.0)
    heard = beeps(robot)
    assert len(heard) == 4 and heard == sorted(heard, reverse=True)


def test_each_note_is_cut_off_after_its_length():
    """The beep has no duration, so stop_beep is what ends a note."""
    robot, _, _ = play(Song.DEATH, sustain=10.0)
    kinds = [e[0] for e in robot.events]
    assert kinds == ["beep", "stop"] * 4


def test_beeps_do_not_wait_on_bluetooth():
    """Waiting on each response would stretch every note by a round trip."""
    robot, _, _ = play(Song.DEATH)
    assert all(e[-1] is False for e in robot.events)


def test_rests_are_silent_but_take_their_time():
    robot, slept, _ = play(Song.DEATH, sustain=10.0)
    assert any(t == pytest.approx(0.05) for t in slept)
    assert len(robot.events) == 8, "a rest sends nothing"


def test_a_held_note_is_kept_sounding_by_rebeeping():
    robot, _, _ = play(Song.DEATH, sustain=0.3)
    assert len(beeps(robot)) > 4, "the 1 s final note needs several beeps"


def test_the_song_takes_as_long_as_it_says():
    _, slept, _ = play(Song.VICTORY)
    assert sum(slept) == pytest.approx(duration(MELODIES[Song.VICTORY]))


def test_a_failing_beeper_does_not_crash_the_end_of_a_match():
    robot, slept, player = FakeBeeper(), [], None
    robot.fail = True
    player = BeepPlayer(robot, 0.3, sleep=slept.append, log=lambda _: None)
    player.play(Song.DEATH)
    assert player.failures > 0
    assert sum(slept) == pytest.approx(duration(MELODIES[Song.DEATH]))


def test_a_failure_is_reported_once_not_for_every_note():
    robot, messages = FakeBeeper(), []
    robot.fail = True
    BeepPlayer(robot, 0.3, sleep=lambda _: None, log=messages.append).play(Song.VICTORY)
    assert len(messages) == 1
