"""Live view of what the microphone is hearing and what it is being read as.

Runs in its own process, fed snapshots through a queue that drops frames when
full. That isolation is the point: matplotlib redraws take tens of milliseconds
and must never sit between a whistle and the motors. If the monitor stalls, the
car keeps driving.

Four panels and a control strip:

  spectrum     the search band, with the detected peak and the zone edges marked
  spectrogram  the last few seconds of the spectrum, with the throttle zone edges
               marked and the detected pitch drawn over it
  gates        level against the measured noise floor, and each shape gate
  readout      note, cents, command, match phase, goal-whistle progress
  controls     role, topic, microphone, sensitivity and a new-match button

The gate panel is the one worth watching in a noisy room: when a whistle does not
take, it tells you which test rejected it rather than leaving you guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRACK_SECONDS = 6.0
MONITOR_BINS = 400

# The spectrogram is redrawn about 25 times a second, so it is kept coarse: 128
# frequency rows by 120 time columns is ~15,000 cells, which redraws in a few
# milliseconds. Rows are max-pooled, so a narrow whistle stays as bright as it is.
SPECTROGRAM_BINS = 128
SPECTROGRAM_COLUMNS = 120

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
    slide_rate: float = 0.0
    motion: str = ""
    goal_progress: int = 0      # legs of the goal warble matched so far
    goal_pattern: str = ""      # e.g. "left > right > left"
    topic: str = ""
    goal_message: str = ""
    tagged_message: str = ""
    notice: str = ""            # the latest event, e.g. "received 'start'"
    light: str = ""             # the light sensor: what it sees, and what trips it
    single_motor: str = ""             # the single motor: idle, or the last turn it was sent
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


class SpectrogramHistory:
    """The last few seconds of spectrum columns, resampled onto a fixed time grid.

    Snapshots arrive at whatever rate the audio loop manages, so columns are kept
    with their timestamps and laid onto evenly spaced slots afterwards. Drawing a
    column per snapshot instead would stretch or squash time whenever the rate
    wobbled, and a whistle's slope on the display is exactly what you read.
    """

    def __init__(self, seconds: float = TRACK_SECONDS, columns: int = SPECTROGRAM_COLUMNS):
        self.seconds = seconds
        self.columns = columns
        self._times: list[float] = []
        self._data: list[np.ndarray] = []

    def add(self, t: float, column: np.ndarray) -> bool:
        """Record a column. Repeats of the same instant are ignored, because the
        display redraws more often than snapshots arrive. True if it was new."""
        if self._times and t <= self._times[-1]:
            return False
        self._times.append(t)
        self._data.append(np.asarray(column, dtype=np.float64))

        cutoff = t - self.seconds
        keep = next((i for i, when in enumerate(self._times) if when >= cutoff), 0)
        if keep > 1:                 # keep one column older than the window, so the
            del self._times[:keep - 1]   # leftmost slot has something to show
            del self._data[:keep - 1]
        return True

    def frame(self, now: float, rows: int) -> np.ndarray:
        """A (rows, columns) array, oldest on the left. Slots with no data yet are
        NaN, so the caller decides what "nothing" looks like."""
        out = np.full((rows, self.columns), np.nan)
        if not self._times:
            return out

        slots = np.linspace(now - self.seconds, now, self.columns)
        times = np.asarray(self._times)
        for slot, index in enumerate(np.searchsorted(times, slots, side="right") - 1):
            if index >= 0:
                out[:, slot] = self._data[index]
        return out


class SpectrogramScale:
    """Colour limits that follow the signal without flickering.

    A fixed range would be wrong for every room and every microphone, and limits
    recomputed from each frame jump about. So: the background is the lower
    quartile and the top is a high percentile (never less than `span_db` above the
    background, so a silent room does not stretch noise up to full brightness), and
    both are smoothed.
    """

    def __init__(self, smoothing: float = 0.15, span_db: float = 30.0, max_range_db: float = 80.0):
        self.smoothing = smoothing
        self.span_db = span_db
        self.max_range_db = max_range_db
        self.vmin: float | None = None
        self.vmax: float | None = None

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        valid = frame[np.isfinite(frame)]
        if valid.size == 0:
            return (0.0, 1.0) if self.vmin is None else (self.vmin, self.vmax)

        low = float(np.percentile(valid, 25))
        high = max(float(np.percentile(valid, 99.8)), low + self.span_db)
        low = max(low, high - self.max_range_db)

        if self.vmin is None:
            self.vmin, self.vmax = low, high
        else:
            a = self.smoothing
            self.vmin += a * (low - self.vmin)
            self.vmax += a * (high - self.vmax)
        return self.vmin, self.vmax


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


class Dropdown:
    """A collapsed button that expands into a list of choices.

    matplotlib has no combobox, so this is a Button whose click toggles a
    RadioButtons list. The list is hidden *and* deactivated when collapsed:
    hiding an Axes does not stop its widget receiving clicks, so without
    deactivating it the invisible list would keep swallowing presses.

    `make_axes` decides where the two axes live. A list that opens outside its
    parent axes has to be a figure-level axes: matplotlib only finds child axes
    inside their parent's bounds, so a child list would look right and take no
    clicks.
    """

    def __init__(self, items, current, on_select, *, prefix, make_axes, button_bounds,
                 list_bounds, short=None):
        from matplotlib.widgets import Button, RadioButtons

        self.items = list(items)            # (label, value)
        self.on_select = on_select
        self.prefix = prefix
        self._short = dict(short or {})     # value -> the text the button shows
        self.open = False

        values = [value for _, value in self.items]
        active = values.index(current) if current in values else 0

        self.button_ax = make_axes(button_bounds)
        self.button = Button(self.button_ax, "", hovercolor="0.88")
        self.button.label.set_fontsize(9)
        self.button.on_clicked(self._toggle)

        self.list_ax = make_axes(list_bounds)
        self.list_ax.set_zorder(20)
        self.list_ax.set_facecolor("white")
        self.list_ax.patch.set_alpha(1.0)

        self.radio = RadioButtons(self.list_ax, [label for label, _ in self.items],
                                  active=active)
        for text in self.radio.labels:
            text.set_fontsize(8)
        self.radio.on_clicked(self._choose)

        self.current = values[active] if values else current
        self._set_open(False)

    def _button_text(self) -> str:
        label = next((label for label, value in self.items if value == self.current), "")
        name = self._short.get(self.current, label) or "select"
        return f"{self.prefix}: {name}   {'^' if self.open else 'v'}"

    def _set_open(self, is_open: bool) -> None:
        self.open = is_open
        self.list_ax.set_visible(is_open)
        # Widget.active is the enable flag consulted by ignore(); a hidden Axes
        # would otherwise still hand clicks to the radio buttons.
        self.radio.active = is_open
        self.button.label.set_text(self._button_text())

    def _toggle(self, _event) -> None:
        self._set_open(not self.open)
        self.list_ax.figure.canvas.draw_idle()

    def _choose(self, label: str) -> None:
        self.current = next(value for text, value in self.items if text == label)
        self._set_open(False)
        self.list_ax.figure.canvas.draw_idle()
        self.on_select(self.current)

    def set_current(self, value) -> None:
        """Follow a change made elsewhere, without firing `on_select`.

        Used when the console changes the role: the dropdown should show what is
        actually true, and echoing it back as a request would loop.
        """
        values = [v for _, v in self.items]
        if value == self.current or value not in values:
            return
        self.current = value
        self.radio.eventson = False
        try:
            self.radio.set_active(values.index(value))
        finally:
            self.radio.eventson = True
        self.button.label.set_text(self._button_text())


class DeviceSelector(Dropdown):
    """The microphone dropdown: a Dropdown over PortAudio input devices."""

    def __init__(self, host_ax, devices, current_index, on_select, *, make_axes=None,
                 button_bounds=(0.02, 0.02, 0.66, 0.095), list_bounds=None):
        self.devices = list(devices)
        items = list(zip(device_labels(self.devices), [d.index for d in self.devices]))
        if list_bounds is None:
            height = min(0.74, 0.085 * len(items) + 0.05)
            list_bounds = (0.02, 0.21, 0.66, height)
        super().__init__(
            items, current_index, on_select, prefix="mic",
            make_axes=make_axes or host_ax.inset_axes,
            button_bounds=button_bounds, list_bounds=list_bounds,
            short={d.index: shorten(d.name, 22) for d in self.devices},
        )

    @property
    def current_index(self) -> int:
        return self.current


def _shutdown(state, fig, plt) -> None:
    """Stop the animation, then close. In that order, and off the step."""
    animation = state.get("animation")
    if animation is not None and animation.event_source is not None:
        animation.event_source.stop()
    plt.close(fig)


def run_monitor(snapshots, band_frequencies, boundaries, devices=(), current_device=None,
                commands=None, sensitivity=5.0, role="ball", topic="",
                goal_message="goal", tagged_message="tagged",
                title="Whistling World Cup"):
    """Process entry point. Reads snapshots until it receives None."""
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button, Slider, TextBox

    axis = pool_centres(np.asarray(band_frequencies), MONITOR_BINS)
    backward_top, forward_top = boundaries
    history = TrackHistory()

    # A fixed grid rather than constrained_layout: the control strip needs a
    # known position to hang its widgets off, and dropdown lists have to be
    # figure-level axes (see Dropdown).
    fig = plt.figure(figsize=(12, 9.0))
    fig.canvas.manager.set_window_title(title)
    grid = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1.75], left=0.125, right=0.985,
                            top=0.955, bottom=0.03, hspace=0.45, wspace=0.22)
    spectrum_ax = fig.add_subplot(grid[0, 0])
    track_ax = fig.add_subplot(grid[0, 1])
    gate_ax = fig.add_subplot(grid[1, 0])
    readout_ax = fig.add_subplot(grid[1, 1])
    strip = grid[2, :].get_position(fig)
    strip_height_in = strip.height * fig.get_size_inches()[1]

    def at(x, y, w, h):
        """A rectangle in the control strip, from fractions of the strip."""
        return [strip.x0 + x * strip.width, strip.y0 + y * strip.height,
                w * strip.width, h * strip.height]

    def list_above(x, w, count):
        """Where a dropdown's list goes: directly above the strip."""
        height = (count * 0.27 + 0.15) / strip_height_in
        return at(x, 1.0, w, height)

    # --- spectrum ---
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

    # --- spectrogram, with the zone edges and the detected pitch over it ---
    spectrogram_axis = pool_centres(np.asarray(band_frequencies), SPECTROGRAM_BINS)
    spectrogram = SpectrogramHistory()
    spectrogram_scale = SpectrogramScale()
    slot_times = np.linspace(-TRACK_SECONDS, 0, SPECTROGRAM_COLUMNS)
    mesh = track_ax.pcolormesh(
        slot_times, spectrogram_axis,
        np.zeros((spectrogram_axis.size, SPECTROGRAM_COLUMNS)),
        shading="nearest", cmap="magma", vmin=0.0, vmax=1.0, zorder=1)
    # Outlined in black: a whistle is a bright ridge in the spectrogram, and a thin
    # coloured line on top of it disappears without something to set it apart.
    (track_line,) = track_ax.plot(
        [], [], lw=1.4, color="#4fd1ff", zorder=3,
        path_effects=[patheffects.withStroke(linewidth=3.2, foreground="black")])
    track_ax.set_yscale("log")
    track_ax.set_ylim(axis[0], axis[-1])
    track_ax.set_xlim(-TRACK_SECONDS, 0)
    track_ax.set_title("spectrogram, last %.0f s" % TRACK_SECONDS)
    track_ax.set_xlabel("seconds ago")
    track_ax.set_ylabel("Hz")
    # The zone edges stay as they were; only the coloured fills are gone.
    for edge in (backward_top, forward_top):
        track_ax.axhline(edge, color="white", ls="--", lw=1.0, alpha=0.85, zorder=2)
    for label, position in (("backward", axis[0] * 1.15),
                            ("forward", backward_top * 1.15),
                            ("fast", forward_top * 1.15)):
        track_ax.text(-TRACK_SECONDS * 0.98, position, label, fontsize=8, color="white",
                      zorder=4)

    # --- gates ---
    gate_names = ["level over floor", "peak / median", "peak / 2nd peak", "sub-harmonic"]
    bars = gate_ax.barh(gate_names, [0, 0, 0, 0], color="0.7")
    thresholds = gate_ax.scatter([0, 0, 0, 0], range(4), marker="|", s=400,
                                 color="crimson", zorder=3)
    gate_ax.set_xlim(0, 70)
    gate_ax.set_xlabel("dB")
    gate_ax.set_title("gates (bar past the mark = pass)")
    gate_ax.invert_yaxis()

    # --- readout ---
    readout_ax.axis("off")
    readout = readout_ax.text(0.02, 0.97, "", va="top", ha="left", fontsize=10,
                              family="monospace", transform=readout_ax.transAxes)

    state = {"latest": None, "running": True, "topic": topic,
             "goal_message": goal_message, "tagged_message": tagged_message}

    def request(kind, value=None):
        """Post a request to the audio loop. It owns the match, not us."""
        if commands is None:
            return
        try:
            commands.put_nowait((kind, value))
        except Exception:
            pass   # the loop is busy; a dropped slider tick costs nothing

    # --- control strip: role, topic, microphone, new match ---
    roles = [("ball", "ball"), ("goalie", "goalie")]
    role_dropdown = Dropdown(
        roles, role, lambda value: request("role", value), prefix="role",
        make_axes=fig.add_axes, button_bounds=at(0.0, 0.70, 0.14, 0.26),
        list_bounds=list_above(0.0, 0.14, len(roles)))

    topic_box = TextBox(fig.add_axes(at(0.215, 0.70, 0.27, 0.26)), "topic ", initial=topic)
    topic_box.label.set_fontsize(9)
    topic_box.text_disp.set_fontsize(9)

    def submit_topic(text):
        text = text.strip()
        if text and text != state["topic"]:
            # Optimistic: Return and then clicking away both submit, and the main
            # loop has not confirmed the first by the time the second arrives. The
            # next snapshot puts the true value back, so a rejection self-corrects.
            state["topic"] = text
            request("topic", text)

    topic_box.on_submit(submit_topic)     # Return, or clicking away

    def message_box(label, key, x, w, initial):
        """A text box for one outcome message, applied on Return like the topic."""
        box = TextBox(fig.add_axes(at(x, 0.38, w, 0.26)), label, initial=initial)
        box.label.set_fontsize(9)
        box.text_disp.set_fontsize(9)

        def submit(text):
            text = text.strip()
            if text and text != state[key]:
                state[key] = text      # optimistic; see submit_topic
                request("message", (key.replace("_message", ""), text))

        box.on_submit(submit)
        return box

    goal_box = message_box("goal msg ", "goal_message", 0.215, 0.27, goal_message)
    tagged_box = message_box("tagged msg ", "tagged_message", 0.665, 0.32, tagged_message)

    mic_dropdown = None
    if devices:
        mic_dropdown = DeviceSelector(
            None, devices, current_device, lambda index: request("device", index),
            make_axes=fig.add_axes, button_bounds=at(0.53, 0.70, 0.28, 0.26),
            list_bounds=list_above(0.53, 0.28, len(devices)))

    new_match = Button(fig.add_axes(at(0.84, 0.70, 0.15, 0.26)), "new match",
                       hovercolor="0.88")
    new_match.label.set_fontsize(9)
    new_match.on_clicked(lambda _event: request("reset"))

    # Sensitivity scales all four gate thresholds at once. The marks on the gate
    # panel are drawn from the values that come back in each snapshot, so
    # dragging this visibly moves them -- the slider explains itself.
    sensitivity_slider = Slider(fig.add_axes(at(0.06, 0.06, 0.34, 0.22)), "sens ",
                                SENSITIVITY_MIN, SENSITIVITY_MAX,
                                valinit=sensitivity, valstep=0.5, valfmt="%.1f")
    sensitivity_slider.label.set_fontsize(9)
    sensitivity_slider.valtext.set_fontsize(9)
    # valstep quantises the drag, so this fires a handful of times per sweep
    # rather than once per pixel.
    sensitivity_slider.on_changed(lambda value: request("sensitivity", value))

    notice = fig.text(strip.x0 + 0.44 * strip.width, strip.y0 + 0.17 * strip.height, "",
                      fontsize=9, va="center", ha="left", color="0.25")

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

        spectrogram.add(snapshot.t, pool_max(snapshot.spectrum_db, SPECTROGRAM_BINS))
        grid = spectrogram.frame(snapshot.t, spectrogram_axis.size)
        vmin, vmax = spectrogram_scale.update(grid)
        mesh.set_array(np.where(np.isfinite(grid), grid, vmin))    # no data yet = darkest
        mesh.set_clim(vmin, vmax)

        over_floor = snapshot.level_db - snapshot.noise_floor_db
        values = [over_floor, snapshot.peak_to_median_db, snapshot.peak_to_second_db,
                  min(snapshot.subharmonic_db, 70.0)]
        marks = list(snapshot.gate_thresholds)
        for bar, value, mark in zip(bars, values, marks):
            bar.set_width(max(0.0, value))
            bar.set_color("#5aa469" if value >= mark else "#c0504d")
        thresholds.set_offsets(np.column_stack([marks, range(4)]))

        # Follow changes made from the console, so the controls show what is
        # actually true. Neither fires a request back.
        role_dropdown.set_current(snapshot.role)
        state["topic"] = snapshot.topic
        if snapshot.topic and not topic_box.capturekeystrokes \
                and topic_box.text != snapshot.topic:
            topic_box.set_val(snapshot.topic)      # dedupes against state["topic"]

        for key, box, live in (("goal_message", goal_box, snapshot.goal_message),
                               ("tagged_message", tagged_box, snapshot.tagged_message)):
            state[key] = live or state[key]
            if live and not box.capturekeystrokes and box.text != live:
                box.set_val(live)      # dedupes against state[key]

        colour = ZONE_COLOURS.get(snapshot.drive, "0.2")
        note = snapshot.note or "--"
        cents = "" if snapshot.cents_off is None else f" {snapshot.cents_off:+.0f}c"
        pitch = "--" if snapshot.frequency is None else f"{snapshot.frequency:7.1f} Hz"
        verdict = "listening" if snapshot.reject_reason is None else snapshot.reject_reason
        legs = snapshot.goal_pattern.count(">") + 1 if snapshot.goal_pattern else 0
        readout.set_text(
            f"role      {snapshot.role}\n"
            f"phase     {snapshot.phase}\n"
            f"topic     {snapshot.topic or '--'}\n\n"
            f"pitch     {pitch}\n"
            f"note      {note}{cents}\n"
            f"gate      {verdict}\n\n"
            f"command   {snapshot.drive.upper()}"
            f"{'  (steering)' if snapshot.steering else ''}\n"
            f"motion    {snapshot.motion or '--'}\n"
            f"slide     {snapshot.slide_rate:+.0f} cents/s\n"
            f"goal      {snapshot.goal_pattern or '--'}  {snapshot.goal_progress}/{legs}\n"
            f"level     {snapshot.level_db:6.1f} dB\n"
            f"floor     {snapshot.noise_floor_db:6.1f} dB\n"
            f"light     {snapshot.light or '--'}\n"
            f"single    {snapshot.single_motor or '--'}"
        )
        readout.set_color(colour)
        notice.set_text(snapshot.notice[:110])
        return []

    animation = FuncAnimation(fig, draw, interval=40, blit=False, cache_frame_data=False)
    fig._whistle_animation = animation   # keep a reference alive
    state["animation"] = animation
    plt.show()
