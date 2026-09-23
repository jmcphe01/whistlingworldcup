"""Turns a stream of pitch readings into one drive intent per frame.

This is the whole control scheme in one place, and it is pure: readings and a
timestamp in, an Intent out. No audio, no robot, no clock of its own.

    sustained pitch   ->  throttle (three zones; silence means stop)
    rising sweep      ->  pivot right, latched briefly
    falling sweep     ->  pivot left, latched briefly
    three high chirps ->  goal claimed

A latched turn overrides the throttle. It has to: the sweep is only recognised
once it is finished, and by then you have stopped whistling, so the throttle has
already fallen back to STOP. Without the latch a turn would be cancelled in the
same frame it was requested.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import Config
from whistle.commands import Drive
from whistle.gestures import RISING, ChirpSequenceDetector, SweepDetector
from whistle.notes import describe
from whistle.pitch import PitchReading, PitchTracker
from whistle.throttle import ThrottleMapper


@dataclass(frozen=True)
class Intent:
    """What the interpreter believes you are asking for, this frame."""

    drive: Drive
    frequency: float | None         # smoothed pitch, None when not whistling
    goal_whistle: bool              # the chirp sequence completed this frame
    steering: bool = False          # drive came from a latched sweep
    sweep_cents: float = 0.0        # net travel of the sweep that latched it

    @property
    def note(self) -> str | None:
        return None if self.frequency is None else describe(self.frequency)[0]


class Interpreter:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.tracker = PitchTracker(self.config.gates)
        self.throttle = ThrottleMapper(self.config.throttle)
        self.sweeps = SweepDetector(self.config.gestures)
        self.chirps = ChirpSequenceDetector(self.config.gestures)

        self._steer: Drive | None = None
        self._steer_until = 0.0
        self._steer_cents = 0.0

    def reset(self) -> None:
        self.tracker.reset()
        self.throttle.reset()
        self.sweeps.reset()
        self.chirps.reset()
        self._steer = None
        self._steer_until = 0.0
        self._steer_cents = 0.0

    def update(self, now: float, reading: PitchReading) -> Intent:
        raw = reading.frequency if reading.voiced else None
        smoothed = self.tracker.update(reading)

        # Chirps read the raw pitch; see the note in gestures.py.
        goal = self.chirps.update(now, raw)

        sweep = self.sweeps.update(now, smoothed)
        if sweep is not None:
            self._steer = Drive.TURN_RIGHT if sweep.direction == RISING else Drive.TURN_LEFT
            self._steer_until = now + self.config.gestures.steer_hold_seconds
            self._steer_cents = sweep.cents

        # Keep the throttle mapper current even while a turn is latched, so it
        # resumes from the pitch you are actually whistling when the turn ends.
        throttle = self.throttle.update(smoothed)

        if self._steer is not None and now < self._steer_until:
            return Intent(self._steer, smoothed, goal, steering=True,
                          sweep_cents=self._steer_cents)

        self._steer = None
        self._steer_cents = 0.0
        return Intent(throttle, smoothed, goal)
