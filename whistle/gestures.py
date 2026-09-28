"""Gestures: how the pitch is *moving* steers, and a warble scores.

Both recognisers are fed one frame at a time with an explicit timestamp, and hold
no reference to a clock. Tests therefore drive them through whole gestures
instantly, and nothing here needs a microphone.

Both use the *smoothed* pitch, since smoothing only helps when the question is
which way a note is travelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from config import GestureConfig
from whistle.notes import cents_above_a4

class Motion(Enum):
    """What the pitch is doing, which is what decides the command."""

    SILENT = "silent"           # not whistling
    UNSETTLED = "unsettled"     # moving, but neither clearly held nor a slide
    HELD = "held"               # steady -> throttle
    RISING = "rising"           # sliding up -> turn right
    FALLING = "falling"         # sliding down -> turn left

    @property
    def is_slide(self) -> bool:
        return self in (Motion.RISING, Motion.FALLING)


@dataclass(frozen=True)
class MotionReading:
    motion: Motion
    rate_cents: float = 0.0     # signed cents per second
    spread_cents: float = 0.0   # highest minus lowest over the window
    duration: float = 0.0


class MotionClassifier:
    """Decides, each frame, whether the pitch is being held or slid.

    This is the whole control split. A held note is a throttle command and a
    slide is a steering command, so the two can never be issued at once and a
    slide cannot drive the car forward on its way past the forward zone.
    """

    def __init__(self, config: GestureConfig | None = None):
        self.config = config or GestureConfig()
        self._track: list[tuple[float, float]] = []   # (time, cents above A4)
        self._last_voiced: float | None = None
        self.last = MotionReading(Motion.SILENT)

    def reset(self) -> None:
        self._track.clear()
        self._last_voiced = None
        self.last = MotionReading(Motion.SILENT)

    @property
    def rate_cents(self) -> float:
        """Current slide rate, for the live monitor."""
        return self.last.rate_cents

    def update(self, now: float, frequency: float | None) -> MotionReading:
        if frequency is None:
            if self._last_voiced is not None and \
                    now - self._last_voiced > self.config.motion_gap_timeout:
                self.reset()
            self.last = MotionReading(Motion.SILENT)
            return self.last

        # A long enough silence means a new gesture, not a continuation.
        if self._last_voiced is not None and \
                now - self._last_voiced > self.config.motion_gap_timeout:
            self._track.clear()

        self._track.append((now, cents_above_a4(frequency)))
        self._last_voiced = now
        window = self._window()

        if len(window) < 3:
            self.last = MotionReading(Motion.UNSETTLED)
            return self.last

        duration = window[-1][0] - window[0][0]
        if duration <= 0:
            self.last = MotionReading(Motion.UNSETTLED)
            return self.last

        pitches = [cents for _, cents in window]
        # Least-squares slope, not the difference between the first and last
        # samples. Endpoint difference is at the mercy of where the window
        # happens to land: on a wobbling note whose ends fall on opposite swings
        # it reports a slide that is not there. A slope averages the wobble out
        # while still tracking a genuine glide exactly.
        rate = self._slope(window)
        net = pitches[-1] - pitches[0]
        spread = max(pitches) - min(pitches)

        self.last = MotionReading(
            motion=self._classify(window, rate, spread, duration),
            rate_cents=rate,
            spread_cents=spread,
            duration=duration,
        )
        return self.last

    def _classify(self, window, rate: float, spread: float,
                  duration: float) -> Motion:
        config = self.config

        if (duration >= config.slide_min_duration
                and abs(rate) >= config.slide_min_rate_cents
                and self._monotonic_fraction(window, rate) >= config.slide_monotonic_fraction):
            return Motion.RISING if rate > 0 else Motion.FALLING

        # The rate test is what stops the opening moments of a slide -- which have
        # barely moved, so their spread is still small -- reading as a held note.
        if abs(rate) < config.hold_max_rate_cents and spread <= config.hold_tolerance_cents:
            return Motion.HELD

        return Motion.UNSETTLED

    def _window(self) -> list[tuple[float, float]]:
        """The most recent `motion_window_seconds` of the current whistle."""
        if not self._track:
            return []
        cutoff = self._track[-1][0] - self.config.motion_window_seconds
        self._track = [point for point in self._track if point[0] >= cutoff]
        return self._track

    @staticmethod
    def _slope(window) -> float:
        """Least-squares rate of change, in cents per second."""
        times = np.array([t for t, _ in window])
        pitches = np.array([cents for _, cents in window])
        spread = times - times.mean()
        denominator = float((spread ** 2).sum())
        if denominator == 0.0:
            return 0.0
        return float((spread * (pitches - pitches.mean())).sum() / denominator)

    @staticmethod
    def _monotonic_fraction(window, direction: float) -> float:
        """Share of consecutive steps moving the same way as the overall trend."""
        steps = [b[1] - a[1] for a, b in zip(window, window[1:])]
        if not steps:
            return 0.0
        return sum(1 for step in steps if step * direction > 0) / len(steps)


_UP = 1
_DOWN = -1
_DIRECTIONS = {"up": _UP, "right": _UP, "down": _DOWN, "left": _DOWN}
MIN_PATTERN_LEGS = 3


def parse_pattern(names) -> tuple[int, ...]:
    """Turn ("left", "right", "left") into directions, rejecting unsafe patterns.

    "left" is a falling slide and "right" a rising one, because that is what
    each steers. Two things are refused rather than quietly accepted, since a
    bad pattern here would either score by accident or never score at all:
    fewer than three legs (a lone slide is already a steering command), and two
    identical directions in a row (the detector alternates by construction, so
    such a pattern could never match).
    """
    pattern = []
    for name in names:
        try:
            pattern.append(_DIRECTIONS[str(name).strip().lower()])
        except KeyError:
            raise ValueError(
                f"unknown goal direction {name!r}; use left/right or down/up") from None

    if len(pattern) < MIN_PATTERN_LEGS:
        raise ValueError(
            f"the goal pattern needs at least {MIN_PATTERN_LEGS} legs, got {len(pattern)}")
    if any(a == b for a, b in zip(pattern, pattern[1:])):
        raise ValueError("the goal pattern must alternate direction "
                         "(a warble goes down, then up, then down)")
    return tuple(pattern)


def describe_pattern(pattern) -> str:
    """(-1, 1, -1) -> "left > right > left", for the monitor."""
    return " > ".join("right" if direction == _UP else "left" for direction in pattern)


class GoalWhistleDetector:
    """Recognises the scoring command: a warble, e.g. left-right-left.

    A warble is a series of legs -- down, up, down -- so this finds the turning
    points of the pitch rather than classifying a window of it. Each leg has to
    travel `goal_leg_cents`, and the pitch has to retrace that far before a
    reversal counts, which is what makes vibrato (tens of cents) invisible to it
    while a deliberate 4-semitone swing is unmistakable.

    A leg counts the moment it has travelled far enough, not when it ends. The
    final leg of the command has no reversal after it, so waiting for one would
    mean the goal is never claimed.

    A leg must also be a *glide*. Turning points alone are not enough: a tone that
    merely jumps between pitches has plenty of them, and an earlier version
    scored for a beeping tone that never slid at all. So a step between frames
    bigger than `goal_max_step_cents` abandons the attempt, and a leg must take at
    least `goal_min_leg_seconds` and be seen over `goal_min_leg_frames` frames.

    Three more things separate the command from ordinary steering, which uses the
    same slides. It must be one unbroken whistle (a silence longer than
    `goal_gap_timeout` abandons the attempt), every leg must be quick, and the
    whole pattern must fit inside `goal_window`. If it fires while you steer,
    lengthen `goal_pattern` to five legs.
    """

    def __init__(self, config: GestureConfig | None = None):
        self.config = config or GestureConfig()
        self.pattern = parse_pattern(self.config.goal_pattern)
        self.last_fire = ""      # what the last completed warble looked like
        self.reset()

    def reset(self) -> None:
        self._last_voiced: float | None = None
        self._last_point: tuple[float, float] | None = None
        self._recent: list[tuple[float, float]] = []      # (time, cents) before any leg
        self._direction = 0
        self._extreme: tuple[float, float] | None = None
        self._since_extreme = 0                           # frames since the extreme moved
        self._legs: list[tuple[int, float, float, float]] = []   # direction, time, cents, seconds

    @property
    def legs_matched(self) -> int:
        """How far into the pattern the whistle has got, for the live monitor.

        The longest run of recent legs that is the start of the pattern.
        """
        seen = tuple(leg[0] for leg in self._legs)
        for length in range(min(len(seen), len(self.pattern) - 1), 0, -1):
            if seen[-length:] == self.pattern[:length]:
                return length
        return 0

    def update(self, now: float, frequency: float | None) -> bool:
        """Feed one smoothed frame. True on the frame the pattern completes."""
        config = self.config
        if self._last_voiced is not None and now - self._last_voiced > config.goal_gap_timeout:
            self.reset()
        if frequency is None:
            return False

        point = (now, cents_above_a4(frequency))
        if self._last_point is not None \
                and abs(point[1] - self._last_point[1]) > config.goal_max_step_cents:
            self.reset()         # a whistle glides; this jumped
        self._last_voiced = now
        self._last_point = point
        leg = config.goal_leg_cents

        if self._direction == 0:
            # Before the first leg: look back over the last leg's-worth of time for
            # the lowest and highest points, and start whichever leg the pitch
            # commits to first. The latest of equal points is the pivot, so a long
            # steady note before the glide is not counted as part of the glide.
            self._recent.append(point)
            cutoff = now - config.goal_max_leg_seconds
            self._recent = [p for p in self._recent if p[0] >= cutoff]
            low = min(reversed(self._recent), key=lambda p: p[1])
            high = max(reversed(self._recent), key=lambda p: p[1])
            if point[1] - low[1] >= leg:
                return self._begin_leg(_UP, low, point,
                                       sum(1 for p in self._recent if p[0] >= low[0]))
            if high[1] - point[1] >= leg:
                return self._begin_leg(_DOWN, high, point,
                                       sum(1 for p in self._recent if p[0] >= high[0]))
            return False

        self._since_extreme += 1
        if self._direction == _UP:
            if point[1] >= self._extreme[1]:
                self._extreme, self._since_extreme = point, 0
            elif self._extreme[1] - point[1] >= leg:
                return self._begin_leg(_DOWN, self._extreme, point, self._since_extreme)
        else:
            if point[1] <= self._extreme[1]:
                self._extreme, self._since_extreme = point, 0
            elif point[1] - self._extreme[1] >= leg:
                return self._begin_leg(_UP, self._extreme, point, self._since_extreme)
        return False

    def _begin_leg(self, direction: int, pivot: tuple[float, float],
                   point: tuple[float, float], frames: int) -> bool:
        config = self.config
        now = point[0]
        seconds = now - pivot[0]

        if seconds < config.goal_min_leg_seconds or frames < config.goal_min_leg_frames:
            # Too abrupt to be a slide. Not a leg: start over from this point.
            self.reset()
            self._last_voiced, self._last_point, self._recent = now, point, [point]
            return False

        if seconds > config.goal_max_leg_seconds:
            self._legs.clear()      # too slow to be part of a warble
        self._direction = direction
        self._extreme = point
        self._since_extreme = 0
        self._legs.append((direction, now, abs(point[1] - pivot[1]), seconds))

        cutoff = now - config.goal_window
        self._legs = [entry for entry in self._legs if entry[1] >= cutoff]

        recent = tuple(entry[0] for entry in self._legs[-len(self.pattern):])
        if recent == self.pattern:
            self.last_fire = ", ".join(
                f"{'right' if d == _UP else 'left'} {cents:.0f} cents in {secs:.2f}s"
                for d, _, cents, secs in self._legs[-len(self.pattern):])
            self.reset()
            return True
        return False
