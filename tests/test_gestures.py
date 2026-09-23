"""Motion classification and chirp recognition.

Both detectors take an explicit timestamp, so these tests play whole gestures
through them instantly. `HOP` is the real frame interval (11.6 ms at 44.1 kHz
with a 512-sample hop), so the frame counts here match the live stream.
"""

from __future__ import annotations

import pytest

from config import GestureConfig
from whistle.gestures import ChirpSequenceDetector, Motion, MotionClassifier
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


def motions(frames, config=None, hop=HOP):
    classifier = MotionClassifier(config)
    return [classifier.update(i * hop, f).motion for i, f in enumerate(frames)]


def settled(frames, config=None, skip=8):
    """Motions after the classifier has enough history to judge from."""
    return set(motions(frames, config)[skip:])


# --- held versus sliding ----------------------------------------------------
#
# This split is the whole control scheme: a steady note is throttle, a moving
# one is steering, and the two must never overlap.

def test_a_steady_note_reads_as_held():
    assert settled(hold(note_to_hz("A5"), 2.0)) == {Motion.HELD}


def test_sliding_up_reads_as_rising():
    assert Motion.RISING in settled(glide(note_to_hz("A4"), note_to_hz("A6"), 1.0))


def test_sliding_down_reads_as_falling():
    assert Motion.FALLING in settled(glide(note_to_hz("A6"), note_to_hz("A4"), 1.0))


def test_a_slide_is_never_also_held():
    """The point of the change: a slide must not drive the car forward on its
    way through the forward zone."""
    for frames in (glide(note_to_hz("A4"), note_to_hz("A6"), 1.0),
                   glide(note_to_hz("A6"), note_to_hz("A4"), 1.0)):
        assert Motion.HELD not in settled(frames)


def test_the_start_of_a_slide_is_not_mistaken_for_a_held_note():
    """Its spread is still tiny in the first frames; only the rate test catches
    it, and without that the car would lurch forward before turning."""
    assert Motion.HELD not in set(motions(glide(note_to_hz("A4"), note_to_hz("A6"), 1.0)))


def test_silence_reads_as_silent():
    assert motions(silence(0.5))[-1] is Motion.SILENT


def test_a_wobbly_note_still_counts_as_held():
    """Whistling is not laboratory-steady; vibrato must not stop the car."""
    base = note_to_hz("A5")
    frames = [base * (1.0 + 0.012 * (1 if i % 2 else -1)) for i in range(120)]
    assert settled(frames) == {Motion.HELD}


def test_a_slow_drift_is_not_a_slide():
    """Failing to hold a note perfectly should not be read as steering."""
    assert not any(m.is_slide for m in settled(glide(880.0, 930.0, 2.0)))


def test_a_scribble_is_neither_and_the_car_stops():
    frames = []
    base = note_to_hz("A5")
    for i in range(80):
        frames.append(base * (1.30 if i % 2 else 0.77))
    assert not any(m is Motion.HELD or m.is_slide for m in settled(frames))


def test_steering_lasts_as_long_as_the_slide():
    """A longer slide means a longer turn, which is what makes it controllable."""
    short = [m for m in motions(glide(note_to_hz("A4"), note_to_hz("A5"), 0.5)) if m.is_slide]
    long = [m for m in motions(glide(note_to_hz("A4"), note_to_hz("A6"), 1.5)) if m.is_slide]
    assert len(long) > len(short)


def test_a_brief_dropout_does_not_break_a_slide():
    frames = glide(note_to_hz("A4"), note_to_hz("A6"), 1.0)
    frames[30] = None
    assert Motion.RISING in settled(frames)


def test_a_long_silence_starts_a_new_gesture():
    frames = (glide(note_to_hz("A4"), note_to_hz("C5"), 0.2)
              + silence(0.5)
              + hold(note_to_hz("C5"), 0.5))
    assert motions(frames)[-1] is Motion.HELD


def test_the_rate_is_reported_for_the_monitor():
    classifier = MotionClassifier()
    for index, frequency in enumerate(glide(note_to_hz("A4"), note_to_hz("A6"), 1.0)):
        classifier.update(index * HOP, frequency)
    assert classifier.rate_cents > 0


def test_a_faster_slide_reports_a_higher_rate():
    def rate(seconds):
        classifier = MotionClassifier()
        for index, frequency in enumerate(glide(note_to_hz("A4"), note_to_hz("A6"), seconds)):
            classifier.update(index * HOP, frequency)
        return classifier.rate_cents

    assert rate(0.5) > rate(2.0)


def test_the_thresholds_are_configurable():
    lazy = GestureConfig(slide_min_rate_cents=5000.0)
    assert not any(m.is_slide for m in settled(glide(note_to_hz("A4"), note_to_hz("A6"), 1.0), lazy))


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
