"""Gestures: how the pitch is *moving* steers, and three short high chirps score.

Both recognisers are fed one frame at a time with an explicit timestamp, and hold
no reference to a clock. Tests therefore drive them through whole gestures
instantly, and nothing here needs a microphone.

Motion uses the *smoothed* pitch, since smoothing only helps when the question
is which way a note is travelling. Chirps use the raw per-frame pitch: they are
short enough that the tracker's persistence requirement would swallow them, and
requiring tens of milliseconds of continuous voicing is its own persistence check.
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


class ChirpSequenceDetector:
    """Recognises the goal command: N short, high chirps in quick succession.

    A single extreme pitch would be a simpler command, but a false positive here
    ends the match, and a stray noise can produce one pitch. It cannot produce
    three deliberately spaced high chirps. Keeping the chirps short also makes
    them unmistakable against the sustained tones that drive the car and the
    sweeps that steer it -- an over-long burst breaks the chain rather than
    counting toward it.
    """

    def __init__(self, config: GestureConfig | None = None):
        self.config = config or GestureConfig()
        self._burst_start: float | None = None
        self._last_on: float | None = None
        self._completed: list[float] = []   # end time of each accepted chirp
        # Set once a burst has run too long. Holds until the pitch drops, so the
        # tail of one sustained note cannot be harvested as a chirp.
        self._suppressed = False

    def reset(self) -> None:
        self._burst_start = None
        self._last_on = None
        self._completed.clear()
        self._suppressed = False

    @property
    def chirps_so_far(self) -> int:
        return len(self._completed)

    def update(self, now: float, frequency: float | None) -> bool:
        """Feed one raw frame. True on the frame the full sequence completes."""
        high = frequency is not None and frequency >= self.config.goal_chirp_min_hz

        if high:
            if self._suppressed:
                return False
            if self._burst_start is None:
                self._burst_start = now
            self._last_on = now
            # A burst that outstays its welcome is a sustained note, not a chirp.
            # Suppress the rest of it: without this, the note's tail would start a
            # fresh burst and could be counted as a chirp in its own right.
            if now - self._burst_start > self.config.goal_chirp_max_duration:
                self._suppressed = True
                self._burst_start = None
                self._last_on = None
                self._completed.clear()
            return False

        self._suppressed = False
        if self._burst_start is None:
            self._expire(now)
            return False

        duration = now - self._burst_start
        start = self._burst_start
        self._burst_start = None
        self._last_on = None

        if not (self.config.goal_chirp_min_duration
                <= duration
                <= self.config.goal_chirp_max_duration):
            self._completed.clear()
            return False

        # Too long a gap since the previous chirp starts a new attempt.
        if self._completed and start - self._completed[-1] > self.config.goal_chirp_max_gap:
            self._completed.clear()

        self._completed.append(now)
        self._expire(now)

        if len(self._completed) >= self.config.goal_chirp_count:
            self.reset()
            return True
        return False

    def _expire(self, now: float) -> None:
        cutoff = now - self.config.goal_chirp_window
        self._completed = [end for end in self._completed if end >= cutoff]
