"""Motion classification and the warble that claims the goal.

Both detectors take an explicit timestamp, so these tests play whole gestures
through them instantly. `HOP` is the real frame interval (11.6 ms at 44.1 kHz
with a 512-sample hop), so the frame counts here match the live stream.
"""

from __future__ import annotations

import pytest

from config import GestureConfig
from whistle.gestures import (
    GoalWhistleDetector,
    Motion,
    MotionClassifier,
    describe_pattern,
    parse_pattern,
)
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


# --- the warble: the goal command -------------------------------------------
#
# "left" is a falling slide and "right" a rising one, so left-right-left is a
# whistle that falls, rises, then falls again without breaking.

DOWN, UP = -1, 1


def warble(directions, cents=600.0, leg_seconds=0.3, start_hz=None, hop=HOP):
    """One unbroken whistle made of legs, each falling (-1) or rising (+1)."""
    start_hz = start_hz or note_to_hz("A5")
    frames = [start_hz] * 3          # a moment of steady whistling first
    offset = 0.0
    for direction in directions:
        steps = max(2, int(round(leg_seconds / hop)))
        for i in range(1, steps + 1):
            frames.append(start_hz * 2 ** ((offset + direction * cents * i / steps) / 1200))
        offset += direction * cents
    return frames


def test_left_right_left_claims_the_goal():
    assert len(play(GoalWhistleDetector(), warble([DOWN, UP, DOWN]))) == 1


def test_it_fires_when_the_last_leg_has_travelled_far_enough_not_after_it_ends():
    """The final leg has no reversal after it, so waiting for one would mean the
    goal was never claimed."""
    frames = warble([DOWN, UP, DOWN])
    when, _ = play(GoalWhistleDetector(), frames)[0]
    assert when < (len(frames) - 1) * HOP


def test_extra_legs_before_the_pattern_do_not_matter():
    assert len(play(GoalWhistleDetector(), warble([UP, DOWN, UP, DOWN]))) == 1


def test_the_wrong_order_does_not_claim_the_goal():
    assert play(GoalWhistleDetector(), warble([UP, DOWN, UP])) == []


def test_two_legs_are_not_enough():
    assert play(GoalWhistleDetector(), warble([DOWN, UP])) == []


def test_a_single_slide_never_claims_the_goal():
    """A lone slide is a steering command, and must stay one."""
    assert play(GoalWhistleDetector(), warble([DOWN])) == []
    assert play(GoalWhistleDetector(), warble([UP])) == []


def test_a_held_note_never_claims_the_goal():
    assert play(GoalWhistleDetector(), hold(note_to_hz("C6"), 5.0)) == []


def test_vibrato_is_invisible_to_it():
    """Tens of cents of wobble must not read as a leg."""
    base = note_to_hz("C6")
    frames = [base * (1.0 + 0.015 * (1 if i % 2 else -1)) for i in range(400)]
    assert play(GoalWhistleDetector(), frames) == []


def test_small_swings_are_not_legs():
    assert play(GoalWhistleDetector(), warble([DOWN, UP, DOWN], cents=150.0)) == []


def test_a_pause_abandons_the_attempt():
    """Steering commands are separated by silences; the warble is one breath."""
    frames = warble([DOWN, UP]) + silence(0.5) + warble([DOWN])
    assert play(GoalWhistleDetector(), frames) == []


def test_steering_left_right_left_with_pauses_does_not_score():
    frames = (warble([DOWN]) + silence(0.5) + warble([UP]) + silence(0.5)
              + warble([DOWN]))
    assert play(GoalWhistleDetector(), frames) == []


def test_slow_drifting_legs_are_not_a_warble():
    assert play(GoalWhistleDetector(), warble([DOWN, UP, DOWN], leg_seconds=1.6)) == []


def test_a_longer_pattern_can_be_configured():
    config = GestureConfig(goal_pattern=("left", "right", "left", "right", "left"))
    assert play(GoalWhistleDetector(config), warble([DOWN, UP, DOWN])) == []
    assert len(play(GoalWhistleDetector(config), warble([DOWN, UP, DOWN, UP, DOWN]))) == 1


def test_it_fires_once_and_then_starts_over():
    frames = warble([DOWN, UP, DOWN]) + silence(0.5) + warble([DOWN, UP, DOWN])
    assert len(play(GoalWhistleDetector(), frames)) == 2


def test_progress_is_visible_for_the_monitor():
    detector = GoalWhistleDetector()
    play(detector, warble([DOWN, UP]))
    assert detector.legs_matched == 2


def test_progress_is_zero_after_the_wrong_start():
    detector = GoalWhistleDetector()
    play(detector, warble([UP]))
    assert detector.legs_matched == 0


@pytest.mark.parametrize("names, expected", [
    (("left", "right", "left"), (DOWN, UP, DOWN)),
    (("DOWN", "Up", "down"), (DOWN, UP, DOWN)),
    (("right", "left", "right"), (UP, DOWN, UP)),
])
def test_patterns_accept_either_vocabulary(names, expected):
    assert parse_pattern(names) == expected


@pytest.mark.parametrize("names", [
    ("left",), ("left", "right"),               # too short to be a series
    ("left", "left", "right"),                   # cannot alternate
    ("left", "sideways", "left"),                # not a direction
])
def test_unsafe_patterns_are_refused(names):
    with pytest.raises(ValueError):
        parse_pattern(names)


def test_a_bad_pattern_in_config_is_refused_by_the_detector():
    with pytest.raises(ValueError):
        GoalWhistleDetector(GestureConfig(goal_pattern=("left",)))


def test_the_pattern_is_described_for_the_monitor():
    assert describe_pattern((DOWN, UP, DOWN)) == "left > right > left"
