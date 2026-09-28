"""Role, topic, rematch, opponent messages and the light sensor, with fakes.

Nothing here touches a robot, a broker or a sound card: every collaborator the
session is given is replaced by a recorder.
"""

from __future__ import annotations

import queue

import pytest

from config import Config, MqttConfig
from whistle.comms import LoopbackComms, validate_topic
from whistle.commands import Drive
from whistle.interpreter import Intent, Interpreter
from whistle.match import Match, MatchRunner, Phase, Role
from whistle.session import Session
from whistle.songs import Song

VOCAB = MqttConfig()


class FakeDriver:
    def __init__(self):
        self.stops = 0
        self.applied: list[Drive] = []
        self.last_tank = None

    def stop(self):
        self.stops += 1

    def apply(self, now, drive):
        self.applied.append(drive)


class FakePlayer:
    def __init__(self):
        self.played: list[Song] = []

    def play(self, song):
        self.played.append(song)


class FakeComms:
    """Records what is published and validates topics like the real one."""

    def __init__(self, topic="ME193/Rogers"):
        self.topic = topic
        self.sent: list[tuple[str, str]] = []

    def publish(self, topic, message):
        self.sent.append((topic, message))

    def set_topic(self, topic):
        self.topic = validate_topic(topic)
        return self.topic


class FakeSensor:
    def __init__(self):
        self.tripped = False
        self.armed = False
        self.acknowledged = 0

    def arm(self):
        self.armed = True

    def disarm(self):
        self.armed = False

    def acknowledge(self):
        self.tripped = False
        self.acknowledged += 1

    def describe(self):
        return "light sensor: fake"


def make(role=Role.BALL, sensor=True):
    match = Match(role, VOCAB)
    driver, player, comms = FakeDriver(), FakePlayer(), FakeComms()
    fake_sensor = FakeSensor() if sensor else None
    session = Session(match, MatchRunner(match, comms, player, driver), driver,
                      Interpreter(Config()), comms, player, fake_sensor, log=lambda _: None)
    session.sensor_fake = fake_sensor
    return session, driver, player, comms


def intent(drive=Drive.FORWARD, goal=False):
    return Intent(drive=drive, frequency=1000.0, goal_whistle=goal)


def started(**kwargs):
    made = make(**kwargs)
    made[0].on_message("start")
    return made


# --- role -------------------------------------------------------------------

def test_the_role_can_be_changed_and_starts_a_fresh_match():
    session, driver, *_ = started()
    session.handle("role", "goalie")
    assert session.match.role is Role.GOALIE
    assert session.match.phase is Phase.WAITING
    assert driver.stops >= 1, "the car is halted when the match is torn up"


def test_only_the_ball_arms_the_light_sensor():
    session, *_ = make()
    session.handle("role", "ball")
    assert session.sensor_fake.armed
    session.handle("role", "goalie")
    assert not session.sensor_fake.armed


def test_an_invalid_role_is_reported_and_changes_nothing():
    session, *_ = started()
    session.handle("role", "referee")
    assert session.match.role is Role.BALL
    assert session.match.running
    assert "ball or goalie" in session.notice


def test_the_ball_with_no_sensor_is_warned_loudly():
    session, *_ = make(sensor=False)
    session.handle("reset")
    assert "WARNING" in session.notice


# --- topic ------------------------------------------------------------------

def test_the_topic_can_be_changed_and_starts_a_fresh_match():
    session, _, _, comms = started()
    session.handle("topic", "ME193/Test")
    assert comms.topic == session.match.config.topic == "ME193/Test"
    assert session.match.phase is Phase.WAITING


def test_outcomes_follow_the_new_topic():
    """The match published to the old topic would never reach the opponent."""
    session, _, _, comms = make()
    session.handle("topic", "ME193/Test")
    session.on_message("start")
    session.handle("sim_goal")
    assert comms.sent == [("ME193/Test", VOCAB.ball_scored_message)]


@pytest.mark.parametrize("bad", ["", "   ", "ME193/#", "ME193/+/x", "/leading"])
def test_a_bad_topic_is_refused_and_nothing_changes(bad):
    session, _, _, comms = started()
    session.handle("topic", bad)
    assert comms.topic == session.match.config.topic == "ME193/Rogers"
    assert session.match.running, "a rejected topic must not end the match"


def test_topics_are_trimmed():
    assert validate_topic("  ME193/Test \n") == "ME193/Test"


# --- rematch ----------------------------------------------------------------

def test_a_finished_match_stays_finished_until_reset():
    session, *_ = started()
    session.handle("sim_goal")
    session.on_message("start")
    assert session.match.phase is Phase.OVER


def test_reset_allows_a_second_match():
    session, _, player, _ = started()
    session.handle("sim_goal")
    session.handle("reset")
    session.on_message("start")
    session.handle("sim_tag")
    assert player.played == [Song.VICTORY, Song.DEATH]


# --- the two songs, at the right moments -------------------------------------

def test_the_ball_scoring_publishes_and_sings_the_winning_song():
    session, driver, player, comms = started()
    session.handle("sim_goal")
    assert comms.sent == [("ME193/Rogers", VOCAB.ball_scored_message)]
    assert player.played == [Song.VICTORY]


def test_the_ball_being_tagged_publishes_and_plays_the_death_song():
    session, driver, player, comms = started()
    stops = driver.stops
    session.handle("sim_tag")
    assert comms.sent == [("ME193/Rogers", VOCAB.ball_tagged_message)]
    assert player.played == [Song.DEATH]
    assert driver.stops > stops, "shut down"


def test_the_goalie_sings_success_when_the_ball_is_tagged():
    session, _, player, _ = started(role=Role.GOALIE)
    session.on_message(VOCAB.ball_tagged_message)
    assert player.played == [Song.VICTORY]


def test_the_goalie_plays_the_death_song_when_the_ball_scores():
    session, _, player, _ = started(role=Role.GOALIE)
    session.on_message(VOCAB.ball_scored_message)
    assert player.played == [Song.DEATH]


def test_the_ball_does_not_react_to_its_own_echoed_message():
    session, _, player, _ = started()
    session.handle("sim_tag")
    session.on_message(VOCAB.ball_tagged_message)
    assert player.played == [Song.DEATH], "sang once, not twice"


# --- typed opponent messages ------------------------------------------------

@pytest.mark.parametrize("kind, message", [
    ("start", VOCAB.start_message),
    ("tagged", VOCAB.ball_tagged_message),
    ("scored", VOCAB.ball_scored_message),
])
def test_opponent_commands_publish_the_agreed_wording(kind, message):
    session, _, _, comms = make()
    session.handle("opponent", kind)
    assert comms.sent == [("ME193/Rogers", message)]


def test_arbitrary_text_can_be_published():
    session, _, _, comms = make()
    session.handle("publish", "Hello There")
    assert comms.sent == [("ME193/Rogers", "Hello There")]


def test_the_loopback_broker_hands_your_publish_back():
    inbox = queue.Queue()
    loop = LoopbackComms("ME193/Rogers", inbox, log=lambda _: None)
    loop.publish("ME193/Rogers", "start")
    loop.publish("ME193/Other", "ignored")
    assert inbox.get_nowait() == "start" and inbox.empty()


# --- the per-frame step -----------------------------------------------------

def test_the_car_is_held_still_until_start():
    session, driver, *_ = make()
    session.step(0.0, intent(Drive.FORWARD_FAST))
    assert driver.applied == [Drive.STOP]


def test_the_intent_drives_the_car_once_the_match_is_running():
    session, driver, *_ = started()
    session.step(0.0, intent(Drive.FORWARD))
    assert driver.applied == [Drive.FORWARD]


def test_a_goal_whistle_claims_the_goal():
    session, _, player, comms = started()
    session.step(0.0, intent(goal=True))
    assert player.played == [Song.VICTORY]
    assert session.match.phase is Phase.OVER


def test_a_goal_whistle_before_start_does_nothing():
    session, _, player, _ = make()
    session.step(0.0, intent(goal=True))
    assert player.played == [] and session.match.phase is Phase.WAITING


def test_a_finished_match_holds_the_car_still():
    session, driver, *_ = started()
    session.handle("sim_goal")
    session.step(0.0, intent(Drive.FORWARD))
    assert driver.applied[-1] is Drive.STOP


# --- the light sensor -------------------------------------------------------

def test_the_goalie_reaching_the_sensor_ends_the_ball_s_match():
    session, driver, player, comms = started()
    session.sensor_fake.tripped = True
    session.step(0.0, intent())
    assert player.played == [Song.DEATH]
    assert comms.sent == [("ME193/Rogers", VOCAB.ball_tagged_message)]


def test_a_trip_during_warm_up_is_discarded_not_latched():
    """A latched trip would end the match the instant it began."""
    session, _, player, _ = make()
    session.sensor_fake.tripped = True
    session.step(0.0, intent())
    session.on_message("start")
    session.step(0.1, intent())
    assert session.sensor_fake.acknowledged == 1
    assert player.played == [] and session.match.running


def test_being_tagged_takes_priority_over_a_simultaneous_goal_whistle():
    session, _, player, comms = started()
    session.sensor_fake.tripped = True
    session.step(0.0, intent(goal=True))
    assert player.played == [Song.DEATH]


# --- housekeeping -----------------------------------------------------------

def test_unknown_requests_are_reported_not_fatal():
    session, *_ = make()
    session.handle("dance")
    assert "unknown" in session.notice


def test_quit_stops_the_loop():
    session, *_ = make()
    session.handle("quit")
    assert session.quit


def test_songs_can_be_previewed():
    session, _, player, _ = make()
    session.handle("song", "win")
    session.handle("song", "lose")
    assert player.played == [Song.VICTORY, Song.DEATH]


def test_status_does_not_change_anything():
    session, *_ = started()
    session.handle("status")
    assert session.match.running
