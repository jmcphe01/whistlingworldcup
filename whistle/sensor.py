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
import threading
import time

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


class SensorMonitor:
    """Polls the colour sensor on a background thread and reports a trip.

    Reading the sensor is a Bluetooth round trip, far too slow to do inside a
    loop that runs 86 times a second, so it lives on its own thread. The thread
    only *reports*: it sets `tripped`, and the match logic on the main thread
    decides what that means.

    Arming measures a fresh baseline first, then watches. The thread keeps
    running after a trip so that a trip during the warm-up (before `start`) can
    be acknowledged and discarded instead of sitting there latched, ready to end
    the match the instant it begins.
    """

    def __init__(self, read, config: SensorConfig | None = None, log=print,
                 clock=time.monotonic, sleep=time.sleep):
        self._read = read
        self.config = config or SensorConfig()
        self._log = log
        self._clock = clock
        self._sleep = sleep

        self.watch = ProximityWatch(self.config)
        self.last_reflection: float | None = None
        self._tripped = threading.Event()
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    @property
    def armed(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def tripped(self) -> bool:
        return self._tripped.is_set()

    def arm(self) -> None:
        """Start watching. Does nothing if already armed."""
        if self.armed:
            return
        self._tripped.clear()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self._stop,),
                                        daemon=True, name="proximity")
        self._thread.start()

    def disarm(self) -> None:
        if self._stop is not None:
            self._stop.set()

    def acknowledge(self) -> None:
        """Consume a trip so the next one can be reported."""
        self._tripped.clear()
        self.watch.reset()

    def readout(self) -> str:
        """A short live reading for the monitor: what it sees against what trips it."""
        if not self.armed:
            return "off"
        if self.watch.baseline is None:
            return "taking baseline"
        last = "no reading" if self.last_reflection is None else f"{self.last_reflection:.0f}"
        return f"{last}  (trips at {self.watch.threshold:.0f})"

    def describe(self) -> str:
        if not self.armed:
            return "light sensor: not armed"
        if self.watch.baseline is None:
            return "light sensor: taking baseline"
        last = "n/a" if self.last_reflection is None else f"{self.last_reflection:.1f}"
        return (f"light sensor: reflection {last}, baseline {self.watch.baseline:.1f}, "
                f"trips at {self.watch.threshold:.1f}")

    def _read_safely(self) -> float | None:
        try:
            return float(self._read())
        except Exception as error:      # a Bluetooth hiccup must not end the match
            self._log(f"  [sensor] read failed: {error}")
            return None

    def _run(self, stop: threading.Event) -> None:
        watch = ProximityWatch(self.config)
        self.watch = watch

        self._log(f"Baselining the light sensor for {self.config.baseline_seconds:.1f}s "
                  "-- keep the area in front of it clear...")
        deadline = self._clock() + self.config.baseline_seconds
        while self._clock() < deadline and not stop.is_set():
            value = self._read_safely()
            if value is not None:
                self.last_reflection = value
                watch.add_baseline_sample(value)
            self._sleep(self.config.poll_interval)

        if stop.is_set():
            return
        try:
            baseline = watch.finish_baseline()
        except ValueError:
            self._log("  [sensor] could not read the sensor; tag detection is OFF")
            return
        self._log(f"  baseline {baseline:.1f}, triggers at {watch.threshold:.1f}")

        while not stop.is_set():
            value = self._read_safely()
            if value is not None:
                self.last_reflection = value
                if watch.update(value):
                    self._tripped.set()
            self._sleep(self.config.poll_interval)
