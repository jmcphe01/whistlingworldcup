"""Turns a stream of pitch readings into one drive intent per frame.

This is the whole control scheme in one place, and it is pure: readings and a
timestamp in, an Intent out. No audio, no robot, no clock of its own.

    pitch held steady ->  throttle (three zones; silence means stop)
    pitch sliding up  ->  pivot right, for as long as the slide lasts
    pitch sliding down->  pivot left, for as long as the slide lasts
    three high chirps ->  goal claimed

Held and sliding are mutually exclusive, so the car never drives forward on a
slide's way through the forward zone: you turn for exactly as long as you slide,
and you drive only while the note is steady. The short release on a turn exists
only to bridge a dropped frame, not to outlive the gesture.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import Config
from whistle.commands import Drive
from whistle.gestures import ChirpSequenceDetector, Motion, MotionClassifier
from whistle.notes import describe
from whistle.pitch import PitchReading, PitchTracker
from whistle.throttle import ThrottleMapper


@dataclass(frozen=True)
class Intent:
    """What the interpreter believes you are asking for, this frame."""

    drive: Drive
    frequency: float | None         # smoothed pitch, None when not whistling
    goal_whistle: bool              # the chirp sequence completed this frame
    steering: bool = False          # drive came from a slide rather than a zone
    slide_rate: float = 0.0         # signed cents per second of that slide
    motion: Motion = Motion.SILENT  # what the pitch is doing, for the monitor

    @property
    def note(self) -> str | None:
        return None if self.frequency is None else describe(self.frequency)[0]


class Interpreter:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.tracker = PitchTracker(self.config.gates)
        self.throttle = ThrottleMapper(self.config.throttle)
        self.motion = MotionClassifier(self.config.gestures)
        self.chirps = ChirpSequenceDetector(self.config.gestures)

        self._steer: Drive | None = None
        self._steer_until = 0.0
        self._steer_rate = 0.0

    def reset(self) -> None:
        self.tracker.reset()
        self.throttle.reset()
        self.motion.reset()
        self.chirps.reset()
        self._steer = None
        self._steer_until = 0.0
        self._steer_rate = 0.0

    def update(self, now: float, reading: PitchReading) -> Intent:
        raw = reading.frequency if reading.voiced else None
        smoothed = self.tracker.update(reading)

        # Chirps read the raw pitch; see the note in gestures.py.
        goal = self.chirps.update(now, raw)

        motion = self.motion.update(now, smoothed)
        if motion.motion.is_slide:
            self._steer = (Drive.TURN_RIGHT if motion.motion is Motion.RISING
                           else Drive.TURN_LEFT)
            self._steer_until = now + self.config.gestures.steer_release_seconds
            self._steer_rate = motion.rate_cents

        # Throttle only applies to a steady note. Anything else clears the
        # mapper, so the next held note is judged fresh rather than against a
        # band left over from whatever the pitch was doing on the way there.
        held = motion.motion is Motion.HELD
        throttle = self.throttle.update(smoothed if held else None)

        if self._steer is not None and now < self._steer_until:
            return Intent(self._steer, smoothed, goal, steering=True,
                          slide_rate=self._steer_rate, motion=motion.motion)

        self._steer = None
        self._steer_rate = 0.0
        return Intent(throttle, smoothed, goal, motion=motion.motion)
