"""Sweep and chirp recognition.

Both detectors take an explicit timestamp, so these tests play whole gestures
through them instantly. `HOP` is the real frame interval (11.6 ms at 44.1 kHz
with a 512-sample hop), so the frame counts here match what the live stream
produces.
"""

from __future__ import annotations

import pytest

from config import GestureConfig
from whistle.gestures import FALLING, RISING, ChirpSequenceDetector, SweepDetector
from whistle.notes import note_to_hz

HOP = 512 / 44100.0     # 11.6 ms


def play(detector, frames, start=0.0, hop=HOP):
    """Feed (frequency or None) frames at `hop` spacing; collect what fires."""
    results = []
    for index, frequency in enumerate(frames):
        outcome = detector.update(start + index * hop, frequency)
        if outcome:
            results.append((start + index * hop, outcome))
    return results


def glide(start_hz, end_hz, seconds, hop=HOP):
    """A whistle sliding smoothly from one pitch to another."""
    count = max(2, int(round(seconds / hop)))
    step = (end_hz - start_hz) / (count - 1)
    return [start_hz + step * i for i in range(count)]


def hold(hz, seconds, hop=HOP):
    return [hz] * max(1, int(round(seconds / hop)))


def silence(seconds, hop=HOP):
    return [None] * max(1, int(round(seconds / hop)))


# --- sweeps: the two turns --------------------------------------------------

def test_a_rising_sweep_turns_right():
    config = GestureConfig()
    fired = play(SweepDetector(config), glide(note_to_hz("A4"), note_to_hz("A5"), 0.6))
    assert len(fired) == 1
    when, sweep = fired[0]
    assert sweep.direction is RISING
    assert sweep.cents >= config.sweep_min_cents
    assert sweep.duration >= config.sweep_min_duration
    assert when < 0.6, "the turn starts during the sweep, not after it"


def test_a_falling_sweep_turns_left():
    fired = play(SweepDetector(), glide(note_to_hz("A5"), note_to_hz("A4"), 0.6))
    assert len(fired) == 1
    assert fired[0][1].direction is FALLING
    assert fired[0][1].cents < 0


def test_a_held_note_never_looks_like_a_sweep():
    """This is the important non-interference case: driving forward is a held
    note, and it must not steer."""
    assert play(SweepDetector(), hold(note_to_hz("A5"), 3.0)) == []


def test_a_wobbly_held_note_is_not_a_sweep():
    detector = SweepDetector()
    frames = []
    for i in range(240):
        frames.append(note_to_hz("A5") * (1.0 + 0.02 * (1 if i % 2 else -1)))
    assert play(detector, frames) == []


def test_a_short_pitch_change_is_not_a_sweep():
    """"Long whistle" is part of the definition -- a quick blip must not steer."""
    assert play(SweepDetector(), glide(note_to_hz("A4"), note_to_hz("A5"), 0.15)) == []


def test_a_small_pitch_change_is_not_a_sweep():
    assert play(SweepDetector(), glide(880.0, 950.0, 1.0)) == []   # ~130 cents


def test_a_non_monotonic_scribble_is_not_a_sweep():
    """Net travel alone is not enough; the whistle has to actually go one way."""
    detector = SweepDetector(GestureConfig(sweep_monotonic_fraction=0.9))
    frames = []
    base = note_to_hz("A4")
    for i in range(60):
        frames.append(base * (1.0 + 0.02 * i) * (1.10 if i % 2 else 0.90))
    assert play(detector, frames) == []


def test_a_sweep_does_not_fire_on_every_frame():
    """Without the reset-on-fire, a two-octave glide would latch a turn on each
    of its ~130 frames. It should produce a handful of turns, not a hundred."""
    frames = glide(note_to_hz("A4"), note_to_hz("A6"), 1.5)
    fired = play(SweepDetector(), frames)
    assert 1 <= len(fired) <= 4


def test_a_longer_sweep_turns_further():
    """Keeping the sweep going earns more turning, which is what makes the
    gesture controllable rather than all-or-nothing."""
    short = play(SweepDetector(), glide(note_to_hz("A4"), note_to_hz("A5"), 0.6))
    long = play(SweepDetector(), glide(note_to_hz("A4"), note_to_hz("A6"), 1.2))
    assert len(long) > len(short)
    assert all(sweep.direction is RISING for _, sweep in long)


def test_two_separate_sweeps_both_fire():
    frames = (glide(note_to_hz("A4"), note_to_hz("A5"), 0.6)
              + silence(0.4)
              + glide(note_to_hz("A4"), note_to_hz("A5"), 0.6))
    assert len(play(SweepDetector(), frames)) == 2


def test_a_brief_dropout_does_not_break_a_sweep():
    """Whistles flicker below the gate for a frame or two; a sweep should survive."""
    frames = glide(note_to_hz("A4"), note_to_hz("A5"), 0.7)
    frames[20] = None       # one dropped frame, well under the gap timeout
    assert len(play(SweepDetector(), frames)) == 1


def test_a_long_silence_splits_one_sweep_into_nothing():
    """Half a sweep, a pause, then the other half is not a gesture."""
    frames = (glide(note_to_hz("A4"), note_to_hz("C#5"), 0.3)
              + silence(0.5)
              + glide(note_to_hz("C#5"), note_to_hz("A5"), 0.3))
    assert play(SweepDetector(), frames) == []


def test_only_the_recent_window_counts():
    """A very slow drift across 10 seconds is not a gesture, because only the
    last `sweep_max_duration` seconds are considered."""
    detector = SweepDetector(GestureConfig(sweep_max_duration=1.0))
    assert play(detector, glide(note_to_hz("A4"), note_to_hz("A5"), 10.0)) == []


def test_travel_is_reported_while_the_sweep_is_in_progress():
    """The monitor shows this so you can see the gesture building."""
    detector = SweepDetector()
    for index, frequency in enumerate(glide(note_to_hz("A4"), note_to_hz("A5"), 0.30)):
        detector.update(index * HOP, frequency)
    assert detector.travel_cents > 500.0


# --- chirps: the goal command ----------------------------------------------

CHIRP_HZ = note_to_hz("C7")      # comfortably above the A6 chirp floor


def chirp_sequence(count=3, on=0.12, gap=0.15, hz=CHIRP_HZ):
    frames = []
    for _ in range(count):
        frames += hold(hz, on) + silence(gap)
    return frames


def test_three_high_chirps_claim_the_goal():
    fired = play(ChirpSequenceDetector(), chirp_sequence())
    assert len(fired) == 1


def test_two_chirps_are_not_enough():
    assert play(ChirpSequenceDetector(), chirp_sequence(count=2)) == []


def test_a_sustained_high_note_is_not_a_chirp_sequence():
    """Whistling D7 to go fast must never be read as claiming the goal."""
    assert play(ChirpSequenceDetector(), hold(note_to_hz("D7"), 4.0) + silence(0.5)) == []


def test_low_chirps_are_ignored():
    """Chirping in the drive range does not claim the goal."""
    assert play(ChirpSequenceDetector(), chirp_sequence(hz=note_to_hz("A5"))) == []


def test_a_sustained_note_breaks_a_partial_sequence():
    frames = (chirp_sequence(count=2)
              + hold(CHIRP_HZ, 1.0) + silence(0.2)
              + chirp_sequence(count=2))
    assert play(ChirpSequenceDetector(), frames) == []


def test_chirps_spaced_too_far_apart_do_not_count():
    assert play(ChirpSequenceDetector(), chirp_sequence(gap=1.2)) == []


def test_chirps_must_be_long_enough_to_be_deliberate():
    """A single stray frame above the gate is not a chirp."""
    frames = []
    for _ in range(5):
        frames += [CHIRP_HZ] + silence(0.15)
    assert play(ChirpSequenceDetector(), frames) == []


def test_the_sequence_fires_once_and_resets():
    fired = play(ChirpSequenceDetector(), chirp_sequence(count=6))
    assert len(fired) == 2, "six chirps is two complete sequences, not four"


def test_partial_progress_is_visible():
    detector = ChirpSequenceDetector()
    play(detector, chirp_sequence(count=2))
    assert detector.chirps_so_far == 2


def test_a_sweep_does_not_claim_the_goal():
    """The steering gesture passes through the chirp band; it must not fire."""
    frames = glide(note_to_hz("A5"), note_to_hz("D7"), 0.8) + silence(0.3)
    assert play(ChirpSequenceDetector(), frames) == []
