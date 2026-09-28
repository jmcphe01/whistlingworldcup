"""Match rules as a pure state machine, plus the MQTT plumbing around it.

`Match` decides what should happen; it never publishes, plays or drives. Each
handler returns an `Outcome` listing the effects, and `MatchRunner` carries them
out. That split is what makes the rules testable without a broker: the tests hand
`Match` a message and inspect the Outcome.

The rules, for both roles on the one shared topic:

    "start"       -> the match begins, both cars may drive
    goalie close  -> the ball shuts down, publishes tagged, plays the death song
    "tagged" -> the goalie plays the song of success
    goal whistle  -> the ball publishes scored, plays the song of success
    "goal" -> the goalie plays the death song

One trap worth naming: a public broker echoes your own publish back to you,
because both cars subscribe to the topic they publish on. Every handler is
therefore role-guarded -- the ball ignores the outcome messages it sent itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from config import MqttConfig
from whistle.songs import Song


class Role(Enum):
    BALL = "ball"
    GOALIE = "goalie"


class Phase(Enum):
    WAITING = "waiting for start"
    RUNNING = "running"
    OVER = "over"


@dataclass(frozen=True)
class Outcome:
    """What the runner should do about an event. `changed` is False for a no-op."""

    phase: Phase
    changed: bool = False
    publish: tuple[str, ...] = ()
    song: Song | None = None
    stop_robot: bool = False
    note: str = ""


def normalise(message: str) -> str:
    return message.strip().lower()


class Match:
    def __init__(self, role: Role, config: MqttConfig | None = None):
        self.role = role
        self.config = config or MqttConfig()
        self.phase = Phase.WAITING

    @property
    def is_ball(self) -> bool:
        return self.role is Role.BALL

    def reset(self, role: Role | None = None) -> None:
        """Back to waiting for `start`, optionally as the other role.

        A finished match stays finished until this is called: a second "start"
        arriving after the whistle must not quietly begin another match.
        """
        if role is not None:
            self.role = role
        self.phase = Phase.WAITING

    def set_messages(self, *, goal: str | None = None, tagged: str | None = None) -> None:
        """Change the agreed wording. A running match carries on: it is the same
        channel, only the words the two teams use for the outcomes are different."""
        changes = {}
        if goal is not None:
            changes["goal_message"] = goal
        if tagged is not None:
            changes["tagged_message"] = tagged
        self.config = replace(self.config, **changes)

    def set_topic(self, topic: str) -> None:
        """Change the shared topic. Outcomes are published to whatever this holds."""
        self.config = replace(self.config, topic=topic)

    @property
    def running(self) -> bool:
        return self.phase is Phase.RUNNING

    def _unchanged(self, note: str = "") -> Outcome:
        return Outcome(self.phase, changed=False, note=note)

    def _finish(self, song: Song, publish: tuple[str, ...], note: str) -> Outcome:
        self.phase = Phase.OVER
        return Outcome(self.phase, changed=True, publish=publish, song=song,
                       stop_robot=True, note=note)

    # --- events -------------------------------------------------------------

    def on_message(self, payload: str) -> Outcome:
        """Handle one MQTT payload from the shared topic."""
        text = normalise(payload)

        if text == normalise(self.config.start_message):
            if self.phase is not Phase.WAITING:
                return self._unchanged("start ignored: match already under way")
            self.phase = Phase.RUNNING
            return Outcome(self.phase, changed=True, note="start received")

        if text == normalise(self.config.tagged_message):
            if self.is_ball:
                return self._unchanged("own tagged message echoed back")
            if self.phase is Phase.OVER:
                return self._unchanged("match already over")
            return self._finish(Song.VICTORY, (), "ball was tagged: goalie wins")

        if text == normalise(self.config.goal_message):
            if self.is_ball:
                return self._unchanged("own scored message echoed back")
            if self.phase is Phase.OVER:
                return self._unchanged("match already over")
            return self._finish(Song.DEATH, (), "ball scored: goalie loses")

        return self._unchanged(f"ignored unknown message: {payload.strip()!r}")

    def on_proximity(self) -> Outcome:
        """The ball's forward light sensor saw the goalie arrive."""
        if not self.is_ball:
            return self._unchanged("proximity ignored: not the ball")
        if not self.running:
            return self._unchanged("proximity ignored: match not running")
        return self._finish(Song.DEATH, (self.config.tagged_message,),
                            "tagged by the goalie")

    def on_goal_whistle(self) -> Outcome:
        """The ball whistled the goal command."""
        if not self.is_ball:
            return self._unchanged("goal whistle ignored: not the ball")
        if not self.running:
            return self._unchanged("goal whistle ignored: match not running")
        return self._finish(Song.VICTORY, (self.config.goal_message,),
                            "scored")


@dataclass
class MatchRunner:
    """Carries out Outcomes: publishes, plays songs, stops the car.

    Every collaborator is injected, so tests substitute fakes for all three.
    """

    match: Match
    publisher: object | None = None      # .publish(topic, message)
    player: object | None = None         # .play(Song)
    driver: object | None = None         # .stop()
    log: list[str] = field(default_factory=list)

    def handle(self, outcome: Outcome) -> Outcome:
        if outcome.note:
            self.log.append(outcome.note)
        if not outcome.changed:
            return outcome

        # Stop first: whatever else happens, the car should not still be moving.
        if outcome.stop_robot and self.driver is not None:
            self.driver.stop()
        for message in outcome.publish:
            if self.publisher is not None:
                self.publisher.publish(self.match.config.topic, message)
        if outcome.song is not None and self.player is not None:
            self.player.play(outcome.song)
        return outcome
