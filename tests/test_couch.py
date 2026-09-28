"""The couch motor: a single motor a partner spins by MQTT message."""

from __future__ import annotations

import pytest

from config import HardwareConfig, MqttConfig
from tests.test_session import VOCAB, make, started
from whistle.console import parse_console_line
from whistle.couch import LEFT, RIGHT, STOP, CouchMotor, parse_command
from whistle.match import Phase, Role


class FakeMotor:
    def __init__(self):
        self.calls = []
        self.fail = False

    def run(self, speed):
        if self.fail:
            raise OSError("bluetooth hiccup")
        self.calls.append(("run", speed))

    def stop(self):
        if self.fail:
            raise OSError("bluetooth hiccup")
        self.calls.append(("stop",))


@pytest.fixture
def couch():
    return CouchMotor(FakeMotor(), HardwareConfig(), log=lambda _: None)


# --- recognising the commands -------------------------------------------------

@pytest.mark.parametrize("payload, action", [
    ("couchleft", LEFT), ("couchright", RIGHT), ("couchstop", STOP),
    ("CouchLeft", LEFT), ("  couchstop\n", STOP),
])
def test_the_three_commands_are_recognised(payload, action):
    assert parse_command(payload, MqttConfig()) == action


@pytest.mark.parametrize("payload", ["", "couch", "left", "couch left", "start", "goal", "couchleft2"])
def test_anything_else_is_not_a_couch_command(payload):
    assert parse_command(payload, MqttConfig()) is None


def test_the_wording_is_configurable():
    config = MqttConfig(couch_left_message="spinl")
    assert parse_command("spinl", config) == LEFT
    assert parse_command("couchleft", config) is None


# --- driving the motor --------------------------------------------------------

def test_left_and_right_spin_opposite_ways(couch):
    couch.execute(LEFT)
    couch.execute(RIGHT)
    (_, left), (_, right) = couch.motor.calls
    assert left < 0 < right


def test_the_speeds_come_from_config():
    motor = FakeMotor()
    CouchMotor(motor, HardwareConfig(couch_left_speed=-80, couch_right_speed=30),
               log=lambda _: None).execute(LEFT)
    assert motor.calls == [("run", -80)]


def test_stop_stops(couch):
    couch.execute(LEFT)
    couch.execute(STOP)
    assert couch.motor.calls[-1] == ("stop",)


def test_the_state_follows_the_commands(couch):
    assert couch.state == "stopped"
    couch.execute(LEFT)
    assert couch.state == "spinning left"
    couch.execute(RIGHT)
    assert couch.state == "spinning right"
    couch.execute(STOP)
    assert couch.state == "stopped"


def test_shutdown_stops_a_spinning_couch(couch):
    """Otherwise quitting the program leaves it turning."""
    couch.execute(RIGHT)
    couch.shutdown()
    assert couch.motor.calls[-1] == ("stop",)
    assert couch.state == "stopped"


def test_a_bluetooth_failure_is_swallowed_and_the_state_is_kept(couch):
    couch.execute(LEFT)
    couch.motor.fail = True
    description = couch.execute(STOP)
    assert "failed" in description
    assert couch.state == "spinning left", "the motor was never told to stop"


def test_an_unknown_action_is_a_programming_error(couch):
    with pytest.raises(ValueError):
        couch.execute("sideways")


# --- through the session ------------------------------------------------------

def make_with_couch(role=Role.BALL):
    session, driver, player, comms = make(role=role)
    session.couch = CouchMotor(FakeMotor(), HardwareConfig(), log=lambda _: None)
    return session, driver, player, comms


def test_a_couch_message_spins_the_couch():
    session, *_ = make_with_couch()
    session.on_message("couchleft")
    assert session.couch.motor.calls == [("run", HardwareConfig().couch_left_speed)]


@pytest.mark.parametrize("role", list(Role))
def test_it_works_in_either_role(role):
    session, *_ = make_with_couch(role)
    session.on_message("couchright")
    assert session.couch.state == "spinning right"


def test_it_works_before_the_match_starts_and_after_it_ends():
    session, *_ = make_with_couch()
    assert session.match.phase is Phase.WAITING
    session.on_message("couchleft")
    session.on_message("start")
    session.handle("sim_goal")
    assert session.match.phase is Phase.OVER
    session.on_message("couchright")
    assert session.couch.state == "spinning right"


def test_a_couch_message_never_reaches_the_match_rules():
    """It would otherwise be logged as an unknown message, and could never start
    or end a match."""
    session, _, player, comms = make_with_couch()
    session.on_message("couchstop")
    assert session.match.phase is Phase.WAITING
    assert "unknown" not in session.notice and player.played == [] and comms.sent == []


def test_the_couch_carries_on_through_a_match():
    session, *_ = make_with_couch()
    session.on_message("couchleft")
    session.on_message("start")
    session.handle("sim_tag")
    session.handle("reset")
    assert session.couch.state == "spinning left", "matches do not touch the couch"


def test_new_matches_do_not_stop_the_couch():
    session, *_ = make_with_couch()
    session.on_message("couchright")
    session.handle("role", "goalie")
    session.handle("topic", "ME193/Test")
    assert session.couch.motor.calls == [("run", HardwareConfig().couch_right_speed)]


def test_with_no_motor_connected_it_says_so_rather_than_failing():
    session, *_ = make()
    session.on_message("couchleft")
    assert "no single motor" in session.notice


def test_the_state_is_reported_for_the_monitor():
    session, *_ = make()
    assert session.couch_state() == "not connected"
    session, *_ = make_with_couch()
    session.on_message("couchleft")
    assert session.couch_state() == "spinning left"


def test_game_messages_cannot_be_reworded_to_a_couch_command():
    session, *_ = started()
    for wording in ("couchleft", "COUCHSTOP"):
        session.handle("message", ("goal", wording))
        assert session.match.config.goal_message == VOCAB.goal_message


# --- the console --------------------------------------------------------------

@pytest.mark.parametrize("word", ["couchleft", "couchright", "couchstop"])
def test_the_console_can_send_couch_commands_as_the_partner(word):
    assert parse_console_line(word) == ("opponent", word)


@pytest.mark.parametrize("word, message", [
    ("couchleft", "couchleft"), ("couchright", "couchright"), ("couchstop", "couchstop"),
])
def test_the_console_publishes_the_configured_wording(word, message):
    session, _, _, comms = make()
    session.handle("opponent", word)
    assert comms.sent == [("ME193/Rogers", message)]
