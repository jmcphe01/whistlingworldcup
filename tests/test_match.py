"""Match rules, including the echo trap on a shared MQTT topic."""

from __future__ import annotations

import pytest

from config import MqttConfig
from whistle.match import Match, MatchRunner, Outcome, Phase, Role, normalise
from whistle.songs import Song


@pytest.fixture
def ball():
    return Match(Role.BALL)


@pytest.fixture
def goalie():
    return Match(Role.GOALIE)


def started(match: Match) -> Match:
    match.on_message("start")
    return match


# --- start ------------------------------------------------------------------

@pytest.mark.parametrize("role", list(Role))
def test_both_roles_wait_for_start(role):
    assert Match(role).phase is Phase.WAITING


@pytest.mark.parametrize("payload", ["start", "START", "  Start\n"])
def test_start_is_matched_loosely(ball, payload):
    """Whitespace and case from whoever types it must not decide the match."""
    outcome = ball.on_message(payload)
    assert outcome.changed and ball.phase is Phase.RUNNING


def test_a_second_start_is_ignored(ball):
    started(ball)
    assert ball.on_message("start").changed is False


def test_unknown_messages_are_ignored(ball):
    outcome = started(ball).on_message("hello world")
    assert outcome.changed is False
    assert ball.phase is Phase.RUNNING


def test_nothing_happens_before_start(ball):
    """Driving into the goal during the warm-up should not win the match."""
    assert ball.on_goal_whistle().changed is False
    assert ball.on_proximity().changed is False
    assert ball.phase is Phase.WAITING


# --- the ball loses ---------------------------------------------------------

def test_the_tagged_ball_stops_publishes_and_dies(ball):
    outcome = started(ball).on_proximity()
    assert outcome.changed
    assert outcome.stop_robot
    assert outcome.song is Song.DEATH
    assert outcome.publish == (MqttConfig().ball_tagged_message,)
    assert ball.phase is Phase.OVER


def test_the_goalie_celebrates_the_tag(goalie):
    outcome = started(goalie).on_message(MqttConfig().ball_tagged_message)
    assert outcome.song is Song.VICTORY
    assert goalie.phase is Phase.OVER


# --- the ball scores --------------------------------------------------------

def test_the_scoring_ball_publishes_and_celebrates(ball):
    outcome = started(ball).on_goal_whistle()
    assert outcome.song is Song.VICTORY
    assert outcome.publish == (MqttConfig().ball_scored_message,)
    assert outcome.stop_robot
    assert ball.phase is Phase.OVER


def test_the_goalie_mourns_the_goal(goalie):
    outcome = started(goalie).on_message(MqttConfig().ball_scored_message)
    assert outcome.song is Song.DEATH
    assert goalie.phase is Phase.OVER


def test_the_two_endings_give_opposite_songs(ball, goalie):
    scored = started(Match(Role.BALL)).on_goal_whistle().song
    mourned = started(Match(Role.GOALIE)).on_message(MqttConfig().ball_scored_message).song
    assert scored is Song.VICTORY and mourned is Song.DEATH

    tagged = started(Match(Role.BALL)).on_proximity().song
    celebrated = started(Match(Role.GOALIE)).on_message(MqttConfig().ball_tagged_message).song
    assert tagged is Song.DEATH and celebrated is Song.VICTORY


# --- the echo trap ----------------------------------------------------------

def test_the_ball_ignores_the_echo_of_its_own_tagged_message(ball):
    """Both cars subscribe to the topic they publish on, so a public broker
    hands every message straight back. Without the role guard the ball would
    react to itself."""
    started(ball)
    ball.on_proximity()
    echo = ball.on_message(MqttConfig().ball_tagged_message)
    assert echo.changed is False
    assert echo.song is None


def test_the_ball_ignores_the_echo_of_its_own_scored_message(ball):
    started(ball)
    ball.on_goal_whistle()
    assert ball.on_message(MqttConfig().ball_scored_message).song is None


def test_the_goalie_does_not_react_to_outcomes_twice(goalie):
    started(goalie)
    goalie.on_message(MqttConfig().ball_scored_message)
    assert goalie.on_message(MqttConfig().ball_tagged_message).changed is False


# --- role guards ------------------------------------------------------------

def test_the_goalie_has_no_light_sensor_rule(goalie):
    assert started(goalie).on_proximity().changed is False


def test_the_goalie_cannot_claim_the_goal(goalie):
    assert started(goalie).on_goal_whistle().changed is False


def test_a_finished_match_stays_finished(ball):
    started(ball).on_goal_whistle()
    assert ball.on_proximity().changed is False
    assert ball.phase is Phase.OVER


# --- custom vocabulary ------------------------------------------------------

def test_the_agreed_message_wording_is_configurable():
    """The wording is agreed with the opponent, so it has to be a config edit."""
    config = MqttConfig(start_message="GO", ball_tagged_message="BALL_DEAD")
    ball = Match(Role.BALL, config)
    assert ball.on_message("go").changed
    assert ball.on_proximity().publish == ("BALL_DEAD",)


def test_normalise_strips_and_lowercases():
    assert normalise("  Start\n") == "start"


# --- the runner -------------------------------------------------------------

class FakePublisher:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def publish(self, topic, message):
        self.sent.append((topic, message))


class FakePlayer:
    def __init__(self):
        self.played: list[Song] = []

    def play(self, song):
        self.played.append(song)


class FakeDriver:
    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1


@pytest.fixture
def runner(ball):
    return MatchRunner(ball, FakePublisher(), FakePlayer(), FakeDriver())


def test_the_runner_carries_out_every_effect(runner):
    started(runner.match)
    runner.handle(runner.match.on_proximity())

    assert runner.driver.stops == 1
    assert runner.publisher.sent == [(MqttConfig().topic, MqttConfig().ball_tagged_message)]
    assert runner.player.played == [Song.DEATH]


def test_the_car_is_stopped_before_anything_slow_happens(runner):
    """Publishing and playing both block; the car must already be still."""
    order = []
    runner.driver.stop = lambda: order.append("stop")
    runner.publisher.publish = lambda topic, message: order.append("publish")
    runner.player.play = lambda song: order.append("play")

    started(runner.match)
    runner.handle(runner.match.on_proximity())
    assert order == ["stop", "publish", "play"]


def test_the_runner_does_nothing_for_an_unchanged_outcome(runner):
    runner.handle(runner.match.on_message("chatter"))
    assert runner.driver.stops == 0
    assert runner.publisher.sent == []
    assert runner.player.played == []


def test_the_runner_logs_what_it_saw(runner):
    runner.handle(runner.match.on_message("start"))
    assert runner.log == ["start received"]


def test_the_runner_tolerates_missing_collaborators():
    """Useful for a dry run with no robot and no broker attached."""
    match = started(Match(Role.BALL))
    MatchRunner(match).handle(match.on_goal_whistle())
    assert match.phase is Phase.OVER


def test_outcome_defaults_are_inert():
    outcome = Outcome(Phase.WAITING)
    assert (outcome.changed, outcome.publish, outcome.song, outcome.stop_robot) == \
        (False, (), None, False)
