"""Gestures: pitch sweeps steer, and three short high chirps claim the goal.

Both recognisers are fed one frame at a time with an explicit timestamp, and hold
no reference to a clock. Tests therefore drive them through whole gestures
instantly, and nothing here needs a microphone.

Sweeps use the *smoothed* pitch, since they are long by definition and smoothing
only helps. Chirps use the raw per-frame pitch: they are short enough that the
tracker's persistence requirement would swallow them, and requiring tens of
milliseconds of continuous voicing is its own persistence check.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import GestureConfig
from whistle.notes import cents_above_a4

RISING = 1
FALLING = -1


@dataclass(frozen=True)
class Sweep:
    """A completed "long whistle that changes pitch significantly"."""

    direction: int      # RISING or FALLING
    cents: float        # signed net travel
    duration: float     # seconds


class SweepDetector:
    """Recognises a sustained rising or falling whistle.

    Three conditions: the whistle must be continuous for `sweep_min_duration`,
    travel at least `sweep_min_cents` net, and be mostly monotonic. The last
    condition is what separates a deliberate sweep from a wobbly held note that
    happens to drift.

    It fires as soon as those minimums are met rather than waiting for the
    whistle to end, which is what keeps steering responsive -- the turn starts
    part way into the sweep instead of after it. The tracking resets on each
    fire, so keeping the sweep going earns another turn every further
    `sweep_min_cents` of travel. A longer sweep therefore turns further, which is
    the behaviour you want from a steering gesture.
    """

    def __init__(self, config: GestureConfig | None = None):
        self.config = config or GestureConfig()
        self._track: list[tuple[float, float]] = []   # (time, cents above A4)
        self._last_voiced: float | None = None

    def reset(self) -> None:
        self._track.clear()
        self._last_voiced = None

    @property
    def travel_cents(self) -> float:
        """Net travel so far, for the live monitor."""
        window = self._window()
        return 0.0 if len(window) < 2 else window[-1][1] - window[0][1]

    def update(self, now: float, frequency: float | None) -> Sweep | None:
        """Feed one frame. Returns a Sweep on the frame the gesture completes."""
        if frequency is None:
            if self._last_voiced is not None and \
                    now - self._last_voiced > self.config.sweep_gap_timeout:
                self.reset()
            return None

        # A long enough silence means this is a new gesture, not a continuation.
        if self._last_voiced is not None and \
                now - self._last_voiced > self.config.sweep_gap_timeout:
            self._track.clear()

        self._track.append((now, cents_above_a4(frequency)))
        self._last_voiced = now

        window = self._window()
        if len(window) < 3:
            return None

        duration = window[-1][0] - window[0][0]
        if duration < self.config.sweep_min_duration:
            return None

        net = window[-1][1] - window[0][1]
        if abs(net) < self.config.sweep_min_cents:
            return None

        if self._monotonic_fraction(window, net) < self.config.sweep_monotonic_fraction:
            return None

        # Reset rather than stop: a continuing sweep can earn another turn once
        # it has travelled another `sweep_min_cents`.
        self.reset()
        return Sweep(
            direction=RISING if net > 0 else FALLING,
            cents=net,
            duration=duration,
        )

    def _window(self) -> list[tuple[float, float]]:
        """The most recent `sweep_max_duration` seconds of the current whistle."""
        if not self._track:
            return []
        cutoff = self._track[-1][0] - self.config.sweep_max_duration
        self._track = [point for point in self._track if point[0] >= cutoff]
        return self._track

    @staticmethod
    def _monotonic_fraction(window: list[tuple[float, float]], net: float) -> float:
        """Share of consecutive steps that move the same way as the net travel."""
        steps = [b[1] - a[1] for a, b in zip(window, window[1:])]
        if not steps:
            return 0.0
        agreeing = sum(1 for step in steps if step * net > 0)
        return agreeing / len(steps)


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
