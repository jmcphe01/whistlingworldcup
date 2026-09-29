"""Live view of the whistle detector, so you can *see* your whistle on a
spectrogram instead of just guessing whether it worked.

Run this on its own -- no MQTT broker or robot needed, just a mic. It runs
the exact same PitchWhistleDetector used by whistle_controller.py, and
shows two things stacked:

  Top:    a spectrogram of the raw mic input, with the six pitch zones
          marked as dotted lines. A whistle shows up as a thin, bright,
          steady line; workshop/room noise shows up as broad, messy blobs
          spread across many frequencies instead.

  Bottom: the estimated pitch over time (a dot per audio block, colored by
          which zone it fell in), so you can watch your whistle actually
          land in the zone you meant to hit.

The title bar shows the live decision (STOP / pivot[90] / pivot[45] /
pivot[-45] / pivot[-90] / FORWARD), the estimated pitch, and the
dominance ratio + duration-gate progress -- the same numbers
whistle_controller.py uses to decide, so you can tune
FREQ_MIN/FREQ_MAX/DOMINANCE_THRESHOLD/MIN_DURATION below against your
actual whistle and room before running the real thing.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

from mic_select import choose_input_device
from whistle_detector import PitchWhistleDetector

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
HISTORY_SECONDS = 6
DISPLAY_MAX_HZ = 6000

# Keep these in sync with whistle_controller.py while you tune.
FREQ_MIN = 800.0           # Hz -- bottom of the whistle range -> STOP
FREQ_MAX = 1600.0          # Hz -- top of the whistle range -> FORWARD
DOMINANCE_THRESHOLD = 0.5  # filtered/raw RMS ratio required to count as "whistling"
MIN_DURATION = 0.25        # seconds the ratio must hold continuously to count

ZONE_COLORS = {
    "STOP": "gray",
    "pivot[90]": "tab:blue",
    "pivot[45]": "tab:cyan",
    "pivot[-45]": "tab:orange",
    "pivot[-90]": "tab:red",
    "FORWARD": "tab:green",
}

detector = PitchWhistleDetector(
    sample_rate=SAMPLE_RATE,
    freq_min=FREQ_MIN,
    freq_max=FREQ_MAX,
    dominance_threshold=DOMINANCE_THRESHOLD,
    min_duration=MIN_DURATION,
)

freqs = np.fft.rfftfreq(BLOCK_SIZE, 1 / SAMPLE_RATE)
n_cols = int(HISTORY_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
spectrogram = np.full((len(freqs), n_cols), -60.0)
pitch_history = np.full(n_cols, np.nan)  # NaN where no whistle -- leaves a gap in the plot

state = {"zone": "STOP"}


def on_audio(chunk):
    windowed = chunk * np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(windowed))
    spectrogram[:, :-1] = spectrogram[:, 1:]
    spectrogram[:, -1] = 20 * np.log10(spectrum + 1e-6)

    state["zone"] = detector.process(chunk)

    pitch_history[:-1] = pitch_history[1:]
    pitch_history[-1] = detector.pitch_hz if detector.pitch_hz is not None else np.nan


def build_plot():
    fig, (ax_spec, ax_pitch) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    img = ax_spec.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        extent=[-HISTORY_SECONDS, 0, freqs[0], freqs[-1]],
        cmap="inferno",
        vmin=-60,
        vmax=20,
    )
    ax_spec.set_ylim(0, DISPLAY_MAX_HZ)
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.axhspan(FREQ_MIN, FREQ_MAX, color="cyan", alpha=0.12)
    zone_bounds = (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6)
    for frac in zone_bounds:
        line_hz = FREQ_MIN + frac * (FREQ_MAX - FREQ_MIN)
        ax_spec.axhline(line_hz, color="cyan", linewidth=1, linestyle=":")
    title = ax_spec.set_title("listening...")

    time_axis = np.linspace(-HISTORY_SECONDS, 0, n_cols)
    (pitch_line,) = ax_pitch.plot(
        time_axis, pitch_history, color="black", marker=".", markersize=2, linestyle="none"
    )
    ax_pitch.set_ylim(FREQ_MIN, FREQ_MAX)
    ax_pitch.set_ylabel("Detected pitch (Hz)")
    ax_pitch.set_xlabel("Seconds ago")
    for frac in zone_bounds:
        ax_pitch.axhline(FREQ_MIN + frac * (FREQ_MAX - FREQ_MIN), color="cyan", linewidth=1, linestyle=":")

    def update(frame):
        img.set_data(spectrogram)
        pitch_line.set_ydata(pitch_history)

        zone = state["zone"]
        color = ZONE_COLORS[zone]
        pitch_line.set_color(color)
        pitch_str = f"{detector.pitch_hz:.0f} Hz" if detector.pitch_hz is not None else "--"
        title.set_text(
            f"DECISION: {zone}   pitch={pitch_str}   "
            f"dominance={detector.dominance:.2f}  held {detector.seconds_above:.2f}s/{MIN_DURATION}s"
        )
        title.set_color(color)
        return img, pitch_line, title

    ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    return fig, ani


if __name__ == "__main__":
    device_index = choose_input_device()
    pa = pyaudio.PyAudio()
    if device_index is None:
        device_index = pa.get_default_input_device_info()["index"]
    print(f"Using input device: {pa.get_device_info_by_index(device_index)['name']}")

    def audio_callback(in_data, frame_count, time_info, status):
        chunk = np.frombuffer(in_data, dtype=np.float32)
        on_audio(chunk)
        return (None, pyaudio.paContinue)

    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=BLOCK_SIZE,
        stream_callback=audio_callback,
    )
    stream.start_stream()

    fig, ani = build_plot()
    print("Showing live spectrogram. Blow your whistle and watch it land in a "
          "zone. Close the window to stop.")
    try:
        plt.show()
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
