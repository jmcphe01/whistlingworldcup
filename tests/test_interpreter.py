"""The whole control scheme, driven by synthetic pitch streams.

These are the integration tests for the control half of the project: a sequence
of readings goes in and a sequence of drive commands comes out, with no audio
device, no robot and no clock.
"""

from __future__ import annotations

import pytest

from config import Config
from tests.conftest import reading
from whistle.commands import Drive
from whistle.interpreter import Interpreter
from whistle.notes import note_to_hz

HOP = 512 / 44100.0


def run(interpreter, frames, start=0.0, hop=HOP):
    """Feed (frequency or None) frames; return the Intent from each."""
    return [
        interpreter.update(start + i * hop, reading(f))
        for i, f in enumerate(frames)
    ]


def hold(hz, seconds, hop=HOP):
    return [hz] * max(1, int(round(seconds / hop)))


def silence(seconds, hop=HOP):
    return [None] * max(1, int(round(seconds / hop)))


def glide(start_hz, end_hz, seconds, hop=HOP):
    count = max(2, int(round(seconds / hop)))
    step = (end_hz - start_hz) / (count - 1)
    return [start_hz + step * i for i in range(count)]


@pytest.fixture
def interpreter():
    return Interpreter(Config())


def final(intents):
    return intents[-1].drive


# --- throttle through the full stack ---------------------------------------

@pytest.mark.parametrize(
    "note, expected",
    [
        ("D7", Drive.FORWARD_FAST),
        ("C6", Drive.FORWARD),
        ("C4", Drive.BACKWARD),
    ],
)
def test_holding_a_note_drives(interpreter, note, expected):
    assert final(run(interpreter, hold(note_to_hz(note), 0.3))) is expected


def test_silence_stops_the_car(interpreter):
    intents = run(interpreter, hold(note_to_hz("C6"), 0.3) + silence(0.2))
    assert final(intents) is Drive.STOP


def test_the_car_stops_the_moment_the_whistle_ends(interpreter):
    """Hold-to-drive: stopping must not wait for a timeout."""
    frames = hold(note_to_hz("C6"), 0.3) + silence(0.2)
    intents = run(interpreter, frames)
    first_silent = len(hold(note_to_hz("C6"), 0.3))
    assert intents[first_silent].drive is Drive.STOP


def test_driving_starts_within_a_few_frames(interpreter):
    """Responsiveness: persistence costs frames, but only a handful."""
    intents = run(interpreter, hold(note_to_hz("C6"), 0.5))
    moving = next(i for i, intent in enumerate(intents) if intent.drive is not Drive.STOP)
    assert moving * HOP < 0.06, "moving within 60 ms of the whistle starting"


# --- steering ---------------------------------------------------------------

def test_a_rising_sweep_pivots_right(interpreter):
    intents = run(interpreter, glide(note_to_hz("A4"), note_to_hz("A5"), 0.6))
    assert any(intent.drive is Drive.TURN_RIGHT for intent in intents)


def test_a_falling_sweep_pivots_left(interpreter):
    intents = run(interpreter, glide(note_to_hz("A5"), note_to_hz("A4"), 0.6))
    assert any(intent.drive is Drive.TURN_LEFT for intent in intents)


def test_the_turn_outlives_the_whistle_that_asked_for_it(interpreter):
    """The sweep is only recognised once it is over, by which point the throttle
    has already fallen back to STOP. Without the latch the turn would be
    cancelled in the frame it was requested."""
    frames = glide(note_to_hz("A4"), note_to_hz("A5"), 0.6) + silence(0.3)
    intents = run(interpreter, frames)
    tail = intents[len(frames) - len(silence(0.3)):]
    assert any(intent.drive is Drive.TURN_RIGHT for intent in tail)
    assert any(intent.steering for intent in tail)


def test_the_turn_eventually_releases(interpreter):
    hold_seconds = Config().gestures.steer_release_seconds
    frames = (glide(note_to_hz("A4"), note_to_hz("A5"), 0.6)
              + silence(hold_seconds + 0.4))
    intents = run(interpreter, frames)
    assert final(intents) is Drive.STOP
    assert intents[-1].steering is False


def test_a_held_note_never_steers(interpreter):
    """Driving forward is a held note; it must not turn."""
    intents = run(interpreter, hold(note_to_hz("C6"), 3.0))
    assert not any(intent.drive.is_turn for intent in intents)


def test_steering_reports_the_slide_that_caused_it(interpreter):
    intents = run(interpreter, glide(note_to_hz("A4"), note_to_hz("A5"), 0.6))
    turning = [intent for intent in intents if intent.steering]
    assert turning and turning[-1].slide_rate > 0


def test_throttle_resumes_after_a_turn(interpreter):
    hold_seconds = Config().gestures.steer_release_seconds
    frames = (glide(note_to_hz("A4"), note_to_hz("A5"), 0.6)
              + silence(hold_seconds + 0.3)
              + hold(note_to_hz("C6"), 0.3))
    assert final(run(interpreter, frames)) is Drive.FORWARD


# --- the goal whistle -------------------------------------------------------

def humps(start_hz=None):
    """Three humps, kept up in the forward and fast zones."""
    from tests.test_gestures import warble

    return warble([1, -1] * 3, start_hz=start_hz or note_to_hz("C6"))


def test_three_humps_claim_the_goal(interpreter):
    intents = run(interpreter, humps())
    assert sum(1 for intent in intents if intent.goal_whistle) == 1


def test_the_same_humps_below_the_reverse_ceiling_do_not(interpreter):
    """Only the forward and fast zones count, so a voice or a low hum cannot add up
    to a score."""
    intents = run(interpreter, humps(start_hz=note_to_hz("C4")))
    assert not any(intent.goal_whistle for intent in intents)


def test_the_region_can_be_switched_off():
    from dataclasses import replace

    config = Config()
    open_ = replace(config, gestures=replace(config.gestures, goal_floor_margin_cents=None))
    intents = run(Interpreter(open_), humps(start_hz=note_to_hz("C4")))
    assert sum(1 for intent in intents if intent.goal_whistle) == 1


def test_a_dip_just_under_the_ceiling_is_forgiven(interpreter):
    """The floor sits 150 cents under the reverse ceiling, so a trough that sags a
    little below it does not break the warble."""
    ceiling = note_to_hz(Config().throttle.backward_top)
    intents = run(interpreter, humps(start_hz=ceiling * 2 ** (-100 / 1200)))
    assert sum(1 for intent in intents if intent.goal_whistle) == 1


def test_ordinary_driving_never_claims_the_goal(interpreter):
    """The whole scheme, exercised at once: nothing you do to drive may score."""
    frames = (hold(note_to_hz("C6"), 1.0) + silence(0.3)
              + hold(note_to_hz("D7"), 1.0) + silence(0.3)
              + hold(note_to_hz("C4"), 1.0) + silence(0.3)
              + glide(note_to_hz("A4"), note_to_hz("A6"), 1.2) + silence(0.3)
              + glide(note_to_hz("A6"), note_to_hz("A4"), 1.2) + silence(0.3))
    intents = run(interpreter, frames)
    assert not any(intent.goal_whistle for intent in intents)


# --- reporting and housekeeping --------------------------------------------

def test_the_intent_names_the_note_for_the_monitor(interpreter):
    intents = run(interpreter, hold(note_to_hz("C6"), 0.3))
    assert intents[-1].note == "C6"
    assert intents[-1].frequency == pytest.approx(note_to_hz("C6"))


def test_an_unvoiced_intent_reports_no_note(interpreter):
    assert run(interpreter, silence(0.2))[-1].note is None


def test_reset_clears_everything(interpreter):
    run(interpreter, glide(note_to_hz("A4"), note_to_hz("A5"), 0.6))
    interpreter.reset()
    assert interpreter.update(99.0, reading(None)).drive is Drive.STOP
    assert interpreter.update(99.0, reading(None)).steering is False


def test_rejected_frames_are_treated_as_silence(interpreter):
    """A frame the gates rejected still carries a frequency; it must be ignored."""
    noisy = reading(880.0, voiced=False, reject_reason="not a lone peak")
    for i in range(30):
        intent = interpreter.update(i * HOP, noisy)
    assert intent.drive is Drive.STOP
    assert intent.frequency is None
