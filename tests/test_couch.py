"""The couch motor: a single motor a partner turns by pivot[<angle>]."""

from __future__ import annotations

import pytest
from legoeducation import MOTOR_MOVE_DIRECTION_CLOCKWISE as CLOCKWISE
from legoeducation import MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE as COUNTERCLOCKWISE

from config import HardwareConfig, MqttConfig
from tests.test_session import VOCAB, make, started
from whistle.console import parse_console_line
from whistle.couch import CouchMotor, parse_command
from whistle.match import Phase, Role


class FakeMotor:
    def __init__(self):
        self.turns = []
        self.stops = 0
        self.fail = False

    def motor_run_for_degrees(self, degrees, *, direction, speed, blocking):
        if self.fail:
            raise OSError("bluetooth hiccup")
        self.turns.append((degrees, direction, speed, blocking))

    def stop(self):
        self.stops += 1


@pytest.fixture
def couch():
    return CouchMotor(FakeMotor(), HardwareConfig(), log=lambda _: None)


# --- reading the message ------------------------------------------------------

@pytest.mark.parametrize("payload, angle", [
    ("pivot[90]", 90), ("pivot[-45]", -45), ("pivot[0]", 0), ("pivot[+30]", 30),
    ("pivot[12.5]", 12.5), ("pivot[-7.25]", -7.25), ("pivot[1e2]", 100),
    ("PIVOT[90]", 90), ("  Pivot[90]\n", 90),
    ("pivot [90]", 90), ("pivot[ 90 ]", 90), ("pivot[ -45 ]", -45),
])
def test_the_angle_is_read_from_the_brackets(payload, angle):
    assert parse_command(payload, MqttConfig()) == pytest.approx(angle)


@pytest.mark.parametrize("payload", [
    "", "pivot", "pivot 90", "pivot(90)", "start", "goal", "tagged",
    "pivot[90", "pivot90]", "xpivot[90]", "pivot[90]x", "couchleft",
])
def test_anything_not_bracketed_is_not_a_couch_message(payload):
    """Only the bracketed form is claimed; the rest goes to the match rules."""
    assert parse_command(payload, MqttConfig()) is None


@pytest.mark.parametrize("payload", ["pivot[]", "pivot[abc]", "pivot[ ]", "pivot[nan]",
                                     "pivot[inf]", "pivot[9 0]", "pivot[--5]"])
def test_brackets_with_no_usable_number_are_reported(payload):
    with pytest.raises(ValueError, match="pivot"):
        parse_command(payload, MqttConfig())


def test_the_word_is_configurable():
    config = MqttConfig(pivot_message="turn")
    assert parse_command("turn[15]", config) == 15
    assert parse_command("pivot[15]", config) is None


# --- turning the motor --------------------------------------------------------

def test_a_positive_angle_turns_clockwise(couch):
    couch.pivot(90)
    assert couch.motor.turns == [(90, CLOCKWISE, HardwareConfig().couch_speed, False)]


def test_a_negative_angle_turns_the_other_way_by_its_size(couch):
    """The library takes an unsigned angle and a direction, so -45 is 45 counter-
    clockwise, not a negative angle passed through."""
    couch.pivot(-45)
    assert couch.motor.turns == [(45, COUNTERCLOCKWISE, HardwareConfig().couch_speed, False)]


def test_the_direction_can_be_inverted_for_how_it_is_mounted():
    motor = FakeMotor()
    CouchMotor(motor, HardwareConfig(couch_invert=True), log=lambda _: None).pivot(90)
    assert motor.turns[0][1] == COUNTERCLOCKWISE


def test_the_turn_does_not_wait_for_the_motor(couch):
    """Waiting would stall the loop that listens for whistles, and the car would
    keep driving on its last command meanwhile."""
    couch.pivot(720)
    assert couch.motor.turns[0][3] is False


def test_fractional_angles_are_rounded_to_whole_degrees(couch):
    couch.pivot(89.6)
    assert couch.motor.turns[0][0] == 90


def test_zero_does_nothing(couch):
    assert "no turn" in couch.pivot(0)
    assert couch.motor.turns == []


def test_a_fraction_that_rounds_to_zero_does_nothing(couch):
    couch.pivot(0.4)
    assert couch.motor.turns == []


def test_a_huge_angle_is_refused(couch):
    """A mistyped pivot[9000000] must not spin the motor for minutes."""
    with pytest.raises(ValueError, match="limit"):
        couch.pivot(9_000_000)
    assert couch.motor.turns == []


def test_the_limit_itself_is_allowed(couch):
    limit = HardwareConfig().couch_max_degrees
    couch.pivot(limit)
    couch.pivot(-limit)
    assert len(couch.motor.turns) == 2


def test_the_state_reports_the_last_turn(couch):
    assert couch.state == "idle"
    couch.pivot(90)
    assert "+90" in couch.state
    couch.pivot(-45)
    assert "-45" in couch.state


def test_shutdown_stops_a_motor_that_is_mid_turn(couch):
    couch.pivot(3000)
    couch.shutdown()
    assert couch.motor.stops == 1 and couch.state == "idle"


def test_a_bluetooth_failure_is_swallowed_and_the_state_is_kept(couch):
    couch.pivot(90)
    couch.motor.fail = True
    assert "failed" in couch.pivot(45)
    assert "+90" in couch.state, "the motor was never told about the second turn"


def test_a_failing_stop_at_shutdown_does_not_raise():
    class BadStop(FakeMotor):
        def stop(self):
            raise OSError("gone")

    CouchMotor(BadStop(), HardwareConfig(), log=lambda _: None).shutdown()


# --- through the session ------------------------------------------------------

def make_with_couch(role=Role.BALL):
    session, driver, player, comms = make(role=role)
    session.couch = CouchMotor(FakeMotor(), HardwareConfig(), log=lambda _: None)
    return session, driver, player, comms


def test_a_pivot_message_turns_the_couch():
    session, *_ = make_with_couch()
    session.on_message("pivot[-30]")
    assert session.couch.motor.turns[0][:2] == (30, COUNTERCLOCKWISE)


@pytest.mark.parametrize("role", list(Role))
def test_it_works_in_either_role(role):
    session, *_ = make_with_couch(role)
    session.on_message("pivot[90]")
    assert len(session.couch.motor.turns) == 1


def test_it_works_before_the_match_starts_and_after_it_ends():
    session, *_ = make_with_couch()
    assert session.match.phase is Phase.WAITING
    session.on_message("pivot[10]")
    session.on_message("start")
    session.handle("sim_goal")
    assert session.match.phase is Phase.OVER
    session.on_message("pivot[20]")
    assert [t[0] for t in session.couch.motor.turns] == [10, 20]


def test_a_pivot_message_never_reaches_the_match_rules():
    """It would otherwise be logged as an unknown message, and could never start
    or end a match."""
    session, _, player, comms = make_with_couch()
    session.on_message("pivot[45]")
    assert session.match.phase is Phase.WAITING
    assert "unknown" not in session.notice and player.played == [] and comms.sent == []


def test_a_bare_pivot_is_not_claimed_and_goes_to_the_match():
    session, *_ = make_with_couch()
    session.on_message("pivot")
    assert session.couch.motor.turns == []
    assert "unknown" in session.notice


def test_a_malformed_pivot_is_reported_and_turns_nothing():
    session, *_ = make_with_couch()
    session.on_message("pivot[abc]")
    assert session.couch.motor.turns == []
    assert "could not read an angle" in session.notice


def test_a_huge_angle_is_reported_and_turns_nothing():
    session, *_ = make_with_couch()
    session.on_message("pivot[9000000]")
    assert session.couch.motor.turns == [] and "limit" in session.notice


def test_matches_and_role_changes_do_not_touch_the_couch():
    session, *_ = make_with_couch()
    session.on_message("pivot[90]")
    session.on_message("start")
    session.handle("sim_tag")
    session.handle("reset")
    session.handle("role", "goalie")
    session.handle("topic", "ME193/Test")
    assert len(session.couch.motor.turns) == 1 and session.couch.motor.stops == 0


def test_with_no_motor_connected_it_says_so_rather_than_failing():
    session, *_ = make()
    session.on_message("pivot[90]")
    assert "no single motor" in session.notice


def test_the_state_is_reported_for_the_monitor():
    session, *_ = make()
    assert session.couch_state() == "not connected"
    session, *_ = make_with_couch()
    session.on_message("pivot[90]")
    assert "+90" in session.couch_state()


def test_game_messages_cannot_be_reworded_to_the_couch_word():
    session, *_ = started()
    session.handle("message", ("goal", "pivot"))
    assert session.match.config.goal_message == VOCAB.goal_message


# --- the console --------------------------------------------------------------

@pytest.mark.parametrize("line, angle", [
    ("pivot 90", "90"), ("pivot -45", "-45"), ("pivot +10", "+10"), ("pivot 12.5", "12.5"),
    ("pivot[90]", "90"), ("pivot [ -45 ]", "-45"), ("PIVOT 30", "30"),
])
def test_the_console_reads_an_angle(line, angle):
    assert parse_console_line(line) == ("pivot", angle)


@pytest.mark.parametrize("line", ["pivot", "pivot abc", "pivot[]", "pivot 9 0"])
def test_the_console_explains_a_missing_angle(line):
    with pytest.raises(ValueError, match="angle"):
        parse_console_line(line)


def test_the_console_publishes_the_wire_form_the_partner_would_send():
    session, _, _, comms = make()
    session.handle("pivot", "-45")
    assert comms.sent == [("ME193/Rogers", "pivot[-45]")]


def test_the_console_uses_the_configured_word():
    session, _, _, comms = make()
    session.match.config = MqttConfig(pivot_message="turn")
    session.handle("pivot", "15")
    assert comms.sent == [("ME193/Rogers", "turn[15]")]


def test_a_console_pivot_round_trips_through_the_parser():
    """What the console sends is exactly what the couch reads back."""
    session, _, _, comms = make_with_couch()
    session.handle("pivot", "-45")
    _, message = comms.sent[0]
    session.on_message(message)
    assert session.couch.motor.turns[0][:2] == (45, COUNTERCLOCKWISE)


def test_a_non_numeric_console_angle_is_refused():
    session, _, _, comms = make()
    session.handle("pivot", "abc")
    assert comms.sent == [] and "not an angle" in session.notice
