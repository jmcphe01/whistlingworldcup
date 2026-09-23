"""Watching the forward-facing colour sensor for an approaching goalie.

Reflection rises as something gets close. The trigger is a sustained rise over a
baseline measured at match start, never an absolute value -- venue lighting will
not match wherever this was tested, and an absolute threshold would either fire
on the first shadow or never fire at all.

Requiring several consecutive reads costs about a quarter of a second and buys
immunity to the single-sample spikes that a passing shadow or a camera flash
produces. Losing the match to a flashbulb would be a poor way to lose it.
"""

from __future__ import annotations

import statistics

from config import SensorConfig


class ProximityWatch:
    """Pure logic: reflection values in, one trigger out."""

    def __init__(self, config: SensorConfig | None = None):
        self.config = config or SensorConfig()
        self.baseline: float | None = None
        self._samples: list[float] = []
        self._consecutive = 0
        self.triggered = False

    def add_baseline_sample(self, reflection: float) -> None:
        self._samples.append(reflection)

    def finish_baseline(self) -> float:
        """Median of the collected samples. Median, so one bad read cannot skew it."""
        if not self._samples:
            raise ValueError("no baseline samples collected")
        self.baseline = statistics.median(self._samples)
        return self.baseline

    def set_baseline(self, value: float) -> None:
        self.baseline = value

    def reset(self) -> None:
        self._consecutive = 0
        self.triggered = False

    @property
    def threshold(self) -> float | None:
        if self.baseline is None:
            return None
        return self.baseline + self.config.trigger_delta

    def update(self, reflection: float) -> bool:
        """Feed one reading. True on the read that confirms the trigger.

        Returns True once only; later reads return False so the caller cannot
        end the match twice.
        """
        if self.baseline is None:
            raise RuntimeError("call finish_baseline() or set_baseline() first")
        if self.triggered:
            return False

        if reflection >= self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0

        if self._consecutive >= self.config.persistence_reads:
            self.triggered = True
            return True
        return False
