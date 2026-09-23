"""Sustained pitch -> throttle.

Whistle and hold to drive; stop whistling and the car stops. Three zones, set in
config as note names:

    C4 .. F#5   reverse
    F#5 .. A6   forward
    A6 and up   forward fast   (D7, the target note, sits well inside)

Band edges are compared in cents rather than Hz so the zones are perceptually
even, and the active band is widened by `hysteresis_cents` on both sides. Without
that widening a pitch resting on a boundary would flip the car between two
commands several times a second.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import ThrottleConfig, throttle_bounds
from whistle.commands import Drive


@dataclass(frozen=True)
class Band:
    drive: Drive
    low_hz: float       # -inf for the bottom band
    high_hz: float      # +inf for the top band

    def contains(self, hz: float) -> bool:
        return self.low_hz <= hz < self.high_hz


def build_bands(config: ThrottleConfig) -> list[Band]:
    """The three throttle zones, ascending."""
    backward_top, forward_top = throttle_bounds(config)
    if not backward_top < forward_top:
        raise ValueError(
            f"throttle bands out of order: backward_top ({config.backward_top}) must be "
            f"below forward_top ({config.forward_top})"
        )
    return [
        Band(Drive.BACKWARD, float("-inf"), backward_top),
        Band(Drive.FORWARD, backward_top, forward_top),
        Band(Drive.FORWARD_FAST, forward_top, float("inf")),
    ]


def _widen(edge: float, cents: float) -> float:
    """Move a band edge by a number of cents (positive = upward in pitch)."""
    if edge in (float("-inf"), float("inf")):
        return edge
    return edge * 2.0 ** (cents / 1200.0)


class ThrottleMapper:
    """Maps a smoothed pitch to a throttle command, with hysteresis."""

    def __init__(self, config: ThrottleConfig | None = None):
        self.config = config or ThrottleConfig()
        self.bands = build_bands(self.config)
        self._current: Band | None = None

    @property
    def current_drive(self) -> Drive:
        return Drive.STOP if self._current is None else self._current.drive

    def reset(self) -> None:
        self._current = None

    def update(self, frequency: float | None) -> Drive:
        """Feed the smoothed pitch (None when not whistling); get a command."""
        if frequency is None:
            self._current = None
            return Drive.STOP

        if self._current is not None and self._still_inside(self._current, frequency):
            return self._current.drive

        for band in self.bands:
            if band.contains(frequency):
                self._current = band
                return band.drive

        # Unreachable: the bands span the whole axis.
        self._current = None
        return Drive.STOP

    def _still_inside(self, band: Band, frequency: float) -> bool:
        """Is the pitch inside the active band, widened by the hysteresis margin?"""
        margin = self.config.hysteresis_cents
        return _widen(band.low_hz, -margin) <= frequency < _widen(band.high_hz, margin)
