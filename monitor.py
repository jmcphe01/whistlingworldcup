"""Live view of what the microphone is hearing and what it is being read as.

Runs in its own process, fed snapshots through a queue that drops frames when
full. That isolation is the point: matplotlib redraws take tens of milliseconds
and must never sit between a whistle and the motors. If the monitor stalls, the
car keeps driving.

Four panels:

  spectrum     the search band, with the detected peak and the zone edges marked
  pitch track  the last few seconds against the throttle zones as coloured bands
  gates        level against the measured noise floor, and each shape gate
  readout      note, cents, command, match phase, gesture progress

The gate panel is the one worth watching in a noisy room: when a whistle does not
take, it tells you which test rejected it rather than leaving you guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRACK_SECONDS = 6.0
MONITOR_BINS = 400

ZONE_COLOURS = {
    "backward": "#6a8fd8",
    "forward": "#5aa469",
    "forward fast": "#d8a34a",
}


@dataclass(frozen=True)
class Snapshot:
    """One frame of everything the monitor draws. Must stay picklable."""

    t: float
    spectrum_db: np.ndarray
    frequency: float | None
    note: str | None
    cents_off: float | None
    level_db: float
    noise_floor_db: float
    peak_to_median_db: float
    peak_to_second_db: float
    subharmonic_db: float
    reject_reason: str | None
    drive: str
    steering: bool
    phase: str
    role: str
    sweep_cents: float = 0.0
    chirps: int = 0
    gate_thresholds: tuple[float, float, float] = (14.0, 8.0, 6.0)


def pool_max(values: np.ndarray, bins: int) -> np.ndarray:
    """Downsample by taking the maximum of each group.

    Max, not mean: the whole point of the spectrum panel is the height of a
    narrow peak, and averaging is exactly the operation that hides it.
    """
    values = np.asarray(values, dtype=np.float64)
    if bins <= 0:
        raise ValueError("bins must be positive")
    if values.size <= bins:
        return values

    edges = np.linspace(0, values.size, bins + 1).astype(int)
    return np.array([values[a:b].max() for a, b in zip(edges, edges[1:]) if b > a])


def pool_centres(axis: np.ndarray, bins: int) -> np.ndarray:
    """The matching frequency axis for `pool_max`."""
    axis = np.asarray(axis, dtype=np.float64)
    if axis.size <= bins:
        return axis

    edges = np.linspace(0, axis.size, bins + 1).astype(int)
    return np.array([axis[a:b].mean() for a, b in zip(edges, edges[1:]) if b > a])


@dataclass
class TrackHistory:
    """Rolling pitch history, kept trimmed to the visible window."""

    seconds: float = TRACK_SECONDS
    times: list[float] = field(default_factory=list)
    pitches: list[float] = field(default_factory=list)

    def add(self, t: float, frequency: float | None) -> None:
        # NaN rather than a skipped point, so the line breaks at silences
        # instead of drawing a false glide across them.
        self.times.append(t)
        self.pitches.append(float("nan") if frequency is None else frequency)
        cutoff = t - self.seconds
        keep = next((i for i, when in enumerate(self.times) if when >= cutoff), 0)
        if keep:
            del self.times[:keep]
            del self.pitches[:keep]


def run_monitor(snapshots, band_frequencies, boundaries, title="Whistling World Cup"):
    """Process entry point. Reads snapshots until it receives None."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    axis = pool_centres(np.asarray(band_frequencies), MONITOR_BINS)
    backward_top, forward_top = boundaries
    history = TrackHistory()

    fig, axes = plt.subplot_mosaic(
        [["spectrum", "track"], ["gates", "readout"]],
        figsize=(12, 7), constrained_layout=True,
    )
    fig.canvas.manager.set_window_title(title)

    # --- spectrum ---
    spectrum_ax = axes["spectrum"]
    (spectrum_line,) = spectrum_ax.plot(axis, np.full(axis.size, -120.0), lw=0.9)
    peak_marker = spectrum_ax.axvline(axis[0], color="crimson", lw=1.4, alpha=0.0)
    spectrum_ax.set_xscale("log")
    spectrum_ax.set_xlim(axis[0], axis[-1])
    spectrum_ax.set_ylim(-120, 10)
    spectrum_ax.set_title("spectrum (search band)")
    spectrum_ax.set_xlabel("Hz")
    spectrum_ax.set_ylabel("dB")
    for edge in (backward_top, forward_top):
        spectrum_ax.axvline(edge, color="0.6", ls="--", lw=0.8)

    # --- pitch track ---
    track_ax = axes["track"]
    (track_line,) = track_ax.plot([], [], lw=1.8, color="crimson")
    track_ax.set_yscale("log")
    track_ax.set_ylim(axis[0], axis[-1])
    track_ax.set_xlim(-TRACK_SECONDS, 0)
    track_ax.set_title("pitch, last %.0f s" % TRACK_SECONDS)
    track_ax.set_xlabel("seconds ago")
    track_ax.set_ylabel("Hz")
    track_ax.axhspan(axis[0], backward_top, color=ZONE_COLOURS["backward"], alpha=0.22)
    track_ax.axhspan(backward_top, forward_top, color=ZONE_COLOURS["forward"], alpha=0.22)
    track_ax.axhspan(forward_top, axis[-1], color=ZONE_COLOURS["forward fast"], alpha=0.22)
    for label, position in (("backward", axis[0] * 1.15),
                            ("forward", backward_top * 1.15),
                            ("fast", forward_top * 1.15)):
        track_ax.text(-TRACK_SECONDS * 0.98, position, label, fontsize=8, color="0.25")

    # --- gates ---
    gate_ax = axes["gates"]
    gate_names = ["level over floor", "peak / median", "peak / 2nd peak", "sub-harmonic"]
    bars = gate_ax.barh(gate_names, [0, 0, 0, 0], color="0.7")
    thresholds = gate_ax.scatter([0, 0, 0, 0], range(4), marker="|", s=400,
                                 color="crimson", zorder=3)
    gate_ax.set_xlim(0, 70)
    gate_ax.set_xlabel("dB")
    gate_ax.set_title("gates (bar past the mark = pass)")
    gate_ax.invert_yaxis()

    # --- readout ---
    readout_ax = axes["readout"]
    readout_ax.axis("off")
    readout = readout_ax.text(0.02, 0.96, "", va="top", ha="left", fontsize=13,
                              family="monospace", transform=readout_ax.transAxes)

    state = {"latest": None, "running": True}

    def drain():
        """Take the freshest snapshot available and discard the backlog."""
        latest = state["latest"]
        while True:
            try:
                item = snapshots.get_nowait()
            except Exception:
                break
            if item is None:
                state["running"] = False
                break
            latest = item
        state["latest"] = latest
        return latest

    def draw(_frame):
        snapshot = drain()
        if not state["running"]:
            plt.close(fig)
            return []
        if snapshot is None:
            return []

        spectrum = pool_max(snapshot.spectrum_db, MONITOR_BINS)
        spectrum_line.set_ydata(spectrum)
        spectrum_ax.set_ylim(min(-120.0, spectrum.min() - 5), max(10.0, spectrum.max() + 5))

        if snapshot.frequency is not None:
            peak_marker.set_xdata([snapshot.frequency, snapshot.frequency])
            peak_marker.set_alpha(0.9)
        else:
            peak_marker.set_alpha(0.0)

        history.add(snapshot.t, snapshot.frequency)
        now = history.times[-1]
        track_line.set_data([t - now for t in history.times], history.pitches)

        over_floor = snapshot.level_db - snapshot.noise_floor_db
        values = [over_floor, snapshot.peak_to_median_db, snapshot.peak_to_second_db,
                  min(snapshot.subharmonic_db, 70.0)]
        marks = [10.0, *snapshot.gate_thresholds]   # noise margin, then the shape gates
        for bar, value, mark in zip(bars, values, marks):
            bar.set_width(max(0.0, value))
            bar.set_color("#5aa469" if value >= mark else "#c0504d")
        thresholds.set_offsets(np.column_stack([marks, range(4)]))

        colour = ZONE_COLOURS.get(snapshot.drive, "0.2")
        note = snapshot.note or "--"
        cents = "" if snapshot.cents_off is None else f" {snapshot.cents_off:+.0f}c"
        pitch = "--" if snapshot.frequency is None else f"{snapshot.frequency:7.1f} Hz"
        verdict = "listening" if snapshot.reject_reason is None else snapshot.reject_reason
        readout.set_text(
            f"role      {snapshot.role}\n"
            f"phase     {snapshot.phase}\n\n"
            f"pitch     {pitch}\n"
            f"note      {note}{cents}\n"
            f"gate      {verdict}\n\n"
            f"command   {snapshot.drive.upper()}"
            f"{'  (steering)' if snapshot.steering else ''}\n"
            f"sweep     {snapshot.sweep_cents:+.0f} cents\n"
            f"chirps    {snapshot.chirps}/3\n"
            f"level     {snapshot.level_db:6.1f} dB   floor {snapshot.noise_floor_db:6.1f} dB"
        )
        readout.set_color(colour)
        return []

    animation = FuncAnimation(fig, draw, interval=40, blit=False, cache_frame_data=False)
    fig._whistle_animation = animation   # keep a reference alive
    plt.show()
