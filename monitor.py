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
    device: str = ""
    # noise margin, peak/median, peak/2nd peak, sub-harmonic -- in bar order
    gate_thresholds: tuple[float, float, float, float] = (10.0, 14.0, 8.0, 6.0)
    sensitivity: float = 5.0


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


from whistle.pitch import SENSITIVITY_MAX, SENSITIVITY_MIN

MAX_LABEL_CHARS = 26


def shorten(name: str, limit: int = MAX_LABEL_CHARS) -> str:
    """Trim a device name to fit a button, keeping the distinguishing start."""
    name = name.strip()
    return name if len(name) <= limit else name[:limit - 1].rstrip() + "\u2026"


def device_labels(devices) -> list[str]:
    """One label per device. The index prefix guarantees they stay unique even
    when two interfaces report the same name."""
    return [f"[{device.index}] {shorten(device.name)}" for device in devices]


def label_to_index(label: str, devices) -> int:
    """Map a label from the list back to its PortAudio device index."""
    for device, candidate in zip(devices, device_labels(devices)):
        if candidate == label:
            return device.index
    raise KeyError(f"no device matching {label!r}")


class DeviceSelector:
    """A collapsed button that expands into the list of input devices.

    matplotlib has no combobox, so this is a Button whose click toggles a
    RadioButtons panel drawn over the readout text. The panel is hidden *and*
    deactivated when collapsed: hiding an Axes does not stop its widget
    receiving clicks, so without deactivating it the invisible list would keep
    swallowing presses in that corner of the window.
    """

    def __init__(self, host_ax, devices, current_index, on_select):
        from matplotlib.widgets import Button, RadioButtons

        self.devices = list(devices)
        self.on_select = on_select
        self.labels = device_labels(self.devices)
        self.open = False

        try:
            active = [d.index for d in self.devices].index(current_index)
        except ValueError:
            active = 0

        self.button_ax = host_ax.inset_axes([0.02, 0.02, 0.66, 0.095])
        self.button = Button(self.button_ax, "", hovercolor="0.88")
        self.button.label.set_fontsize(9)
        self.button.on_clicked(self._toggle)

        height = min(0.74, 0.085 * len(self.labels) + 0.05)
        self.list_ax = host_ax.inset_axes([0.02, 0.21, 0.66, height])
        self.list_ax.set_zorder(20)
        self.list_ax.set_facecolor("white")
        self.list_ax.patch.set_alpha(1.0)

        self.radio = RadioButtons(self.list_ax, self.labels, active=active)
        for text in self.radio.labels:
            text.set_fontsize(8)
        self.radio.on_clicked(self._choose)

        self.current_index = self.devices[active].index if self.devices else current_index
        self._set_open(False)
        self._refresh_button()

    def _refresh_button(self) -> None:
        current = next((d for d in self.devices if d.index == self.current_index), None)
        name = shorten(current.name, 22) if current else "select input"
        self.button.label.set_text(f"mic: {name}   {'^' if self.open else 'v'}")

    def _set_open(self, is_open: bool) -> None:
        self.open = is_open
        self.list_ax.set_visible(is_open)
        # Widget.active is the enable flag consulted by ignore(); a hidden Axes
        # would otherwise still hand clicks to the radio buttons.
        self.radio.active = is_open
        self._refresh_button()

    def _toggle(self, _event) -> None:
        self._set_open(not self.open)
        self.list_ax.figure.canvas.draw_idle()

    def _choose(self, label: str) -> None:
        index = label_to_index(label, self.devices)
        self.current_index = index
        self._set_open(False)
        self.list_ax.figure.canvas.draw_idle()
        self.on_select(index)


def _shutdown(state, fig, plt) -> None:
    """Stop the animation, then close. In that order, and off the step."""
    animation = state.get("animation")
    if animation is not None and animation.event_source is not None:
        animation.event_source.stop()
    plt.close(fig)


def run_monitor(snapshots, band_frequencies, boundaries, devices=(), current_device=None,
                commands=None, sensitivity=5.0, title="Whistling World Cup"):
    """Process entry point. Reads snapshots until it receives None."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Slider

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
    readout = readout_ax.text(0.02, 0.97, "", va="top", ha="left", fontsize=11,
                              family="monospace", transform=readout_ax.transAxes)

    def request(kind, value):
        """Post a request to the audio loop. It owns the detector, not us."""
        if commands is None:
            return
        try:
            commands.put_nowait((kind, value))
        except Exception:
            pass   # the loop is busy; a dropped slider tick costs nothing

    selector = None
    if devices:
        selector = DeviceSelector(readout_ax, devices, current_device,
                                  lambda index: request("device", index))

    # Sensitivity scales all four gate thresholds at once. The marks on the gate
    # panel are drawn from the values that come back in each snapshot, so
    # dragging this visibly moves them -- the slider explains itself.
    slider_ax = readout_ax.inset_axes([0.17, 0.135, 0.48, 0.05])
    sensitivity_slider = Slider(slider_ax, "sens ", SENSITIVITY_MIN, SENSITIVITY_MAX,
                                valinit=sensitivity, valstep=0.5, valfmt="%.1f")
    sensitivity_slider.label.set_fontsize(9)
    sensitivity_slider.valtext.set_fontsize(9)
    # valstep quantises the drag, so this fires a handful of times per sweep
    # rather than once per pixel.
    sensitivity_slider.on_changed(lambda value: request("sensitivity", value))

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
            if not state.get("closing"):
                state["closing"] = True
                # Closing here would pull the timer out from under the animation
                # mid-step, and matplotlib then raises on its own event source.
                # Hand the close to a one-shot timer so it runs outside the step.
                closer = fig.canvas.new_timer(interval=1)
                closer.add_callback(_shutdown, state, fig, plt)
                closer.start()
                state["closer"] = closer     # a dropped timer never fires
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
        marks = list(snapshot.gate_thresholds)
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
            f"level     {snapshot.level_db:6.1f} dB\n"
            f"floor     {snapshot.noise_floor_db:6.1f} dB\n\n"
            f"mic       {shorten(snapshot.device, 22) or '--'}\n"
            f"sens      {snapshot.sensitivity:.1f}"
        )
        readout.set_color(colour)
        return []

    animation = FuncAnimation(fig, draw, interval=40, blit=False, cache_frame_data=False)
    fig._whistle_animation = animation   # keep a reference alive
    state["animation"] = animation
    plt.show()
