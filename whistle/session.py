"""Everything that happens to a match while the program runs.

The audio loop produces an intent every frame; this decides what the match does
with it, and it is also where the outside world's requests land: a role or topic
picked in the monitor, a message from the broker, a line typed in the console, a
trip from the light sensor. All of them arrive as `(kind, value)` and are handled
here, on the main thread, so match state never has two writers.

Every collaborator is injected, which is what lets this be exercised without a
robot, a broker or a sound card.
"""

from __future__ import annotations

from whistle.commands import Drive
from whistle.console import HELP
from whistle.match import Match, MatchRunner, Outcome, Phase, Role
from whistle.songs import Song


class Session:
    def __init__(self, match: Match, runner: MatchRunner, driver, interpreter, comms,
                 player=None, sensor=None, log=print):
        self.match = match
        self.runner = runner
        self.driver = driver
        self.interpreter = interpreter
        self.comms = comms
        self.player = player
        self.sensor = sensor
        self._log = log

        self.notice = ""            # the latest event, shown in the monitor
        self.quit = False
        self._announced = match.phase

    # --- reporting -----------------------------------------------------------

    def say(self, text: str) -> None:
        self.notice = text
        self._log(f"  {text}")

    def _run(self, outcome: Outcome) -> Outcome:
        self.runner.handle(outcome)
        if outcome.note:
            self.say(outcome.note)
        return outcome

    def _announce_phase(self) -> None:
        if self.match.phase is not self._announced:
            self._announced = self.match.phase
            self._log(f"  phase: {self.match.phase.value}")

    # --- the per-frame step --------------------------------------------------

    def step(self, now: float, intent) -> None:
        """One analysis frame: sensor, then goal, then throttle."""
        self._poll_sensor()

        if self.match.running and intent.goal_whistle:
            self._run(self.match.on_goal_whistle())
        elif self.match.running:
            self.driver.apply(now, intent.drive)
        else:
            self.driver.apply(now, Drive.STOP)

        self._announce_phase()

    def _poll_sensor(self) -> None:
        if self.sensor is None or not self.sensor.tripped:
            return
        self.sensor.acknowledge()
        if self.match.running and self.match.is_ball:
            self._run(self.match.on_proximity())
        else:
            self.say("light sensor tripped, ignored: no match is running")

    def on_message(self, payload: str) -> None:
        """A payload from the shared topic."""
        self.say(f"received {payload!r}")
        self._run(self.match.on_message(payload))
        self._announce_phase()

    # --- match lifecycle -----------------------------------------------------

    def new_match(self, role: Role | None = None) -> str:
        """Tear up the match and start again. Returns a warning, or "".

        The warning is returned rather than said, because callers follow it with
        their own notice and the second would overwrite the first -- and the
        monitor only shows the latest.
        """
        self.match.reset(role)
        self.driver.stop()
        self.interpreter.reset()
        self._announced = self.match.phase
        return self.arm_sensor()

    def arm_sensor(self) -> str:
        """Arm the light sensor for the ball, disarm it otherwise."""
        if self.sensor is None:
            if self.match.is_ball:
                return ("WARNING: no light sensor is connected, so the ball cannot be "
                        "tagged. The rules require it open and facing forward.")
            return ""
        if self.match.is_ball:
            self.sensor.arm()
        else:
            self.sensor.disarm()
        return ""

    def _fresh_match(self, headline: str, role: Role | None = None) -> None:
        warning = self.new_match(role)
        self.say(f"{headline} {warning}".strip())

    # --- requests from the monitor and the console ---------------------------

    def handle(self, kind: str, value=None) -> None:
        method = getattr(self, f"_do_{kind}", None)
        if method is None:
            self.say(f"ignored unknown request {kind!r}")
            return
        try:
            method(value)
        except ValueError as error:
            self.say(str(error))

    def _do_role(self, value) -> None:
        try:
            role = Role(str(value).strip().lower())
        except ValueError:
            raise ValueError(f"role must be ball or goalie, not {value!r}") from None
        self._fresh_match(f"role is now {role.value}; new match, waiting for start.", role)

    def _do_topic(self, value) -> None:
        topic = self.comms.set_topic(value)     # validates, so it can raise
        self.match.set_topic(topic)
        self._fresh_match(f"topic is now {topic}; new match, waiting for start.")

    def _do_reset(self, _value) -> None:
        self._fresh_match("new match, waiting for start.")

    def _publish(self, message: str) -> None:
        topic = self.match.config.topic
        self.comms.publish(topic, message)
        self.say(f"published {message!r} to {topic}")

    def _do_publish(self, text) -> None:
        self._publish(str(text))

    def _do_opponent(self, kind) -> None:
        config = self.match.config
        self._publish({
            "start": config.start_message,
            "tagged": config.ball_tagged_message,
            "scored": config.ball_scored_message,
        }[kind])

    def _do_sim_goal(self, _value) -> None:
        self.say("simulating: the ball whistled the goal command")
        self._run(self.match.on_goal_whistle())
        self._announce_phase()

    def _do_sim_tag(self, _value) -> None:
        self.say("simulating: the goalie reached the light sensor")
        self._run(self.match.on_proximity())
        self._announce_phase()

    def _do_song(self, which) -> None:
        if self.player is None:
            raise ValueError("no sound output is attached")
        self.say(f"playing the {'winning' if which == 'win' else 'losing'} song")
        self.player.play(Song.VICTORY if which == "win" else Song.DEATH)

    def _do_status(self, _value) -> None:
        sensor = self.sensor.describe() if self.sensor else "light sensor: not connected"
        tank = self.driver.last_tank
        for line in (
            f"role    {self.match.role.value}",
            f"phase   {self.match.phase.value}",
            f"topic   {self.match.config.topic}",
            f"drive   {'not yet commanded' if tank is None else tank}",
            sensor,
        ):
            self._log(f"  {line}")

    def _do_help(self, _value) -> None:
        self._log(HELP)

    def _do_quit(self, _value) -> None:
        self.quit = True
