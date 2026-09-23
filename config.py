"""Every tunable value in one place.

Anything you might want to change trackside lives here. `Config.load()` also
reads `config.local.json` if it exists, so machine-specific values (input
device, measured noise floor) stay out of git.

Pitch boundaries are written as note names and converted to Hz at import, so
the zones land on musical intervals rather than arbitrary Hz numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from whistle.notes import note_to_hz

LOCAL_CONFIG_PATH = Path(__file__).with_name("config.local.json")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 44100
    frame_size: int = 2048          # analysis window: 46 ms -> ~1 Hz after interpolation
    hop_size: int = 512             # 11.6 ms between updates -> ~86 readings/second
    channels: int = 1

    # None = system default. May be an int index or a case-insensitive substring
    # of the device name, so swapping in the directional mic is a config change.
    input_device: int | str | None = None


@dataclass(frozen=True)
class GateConfig:
    """Thresholds that decide whether a frame contains a whistle at all."""

    # Search band. The floor sits just under C4 (261.6 Hz), the lowest note Jake
    # can whistle; the ceiling is above any usable whistle harmonic-free peak.
    min_hz: float = 240.0
    max_hz: float = 4200.0

    # Three shape tests, cheapest first. All key on spectral *shape*, which is
    # why they survive a loud room where a plain level gate would not.

    # 1. A whistle is a narrow spike, so its peak towers over the in-band
    #    median. Broadband noise (clapping, scraping, HVAC) does not.
    peak_to_median_db: float = 14.0

    # 2. A whistle is a *lone* spike. Voiced speech is a harmonic comb, so its
    #    peaks come in families of comparable height. Measured separation:
    #    whistles 52-70 dB, a whistle buried in babble 10-21 dB, babble alone
    #    under 4 dB. This is the gate that does the real work.
    peak_to_second_db: float = 8.0

    # 3. If the peak is itself a harmonic of something lower, there is energy at
    #    f/2 or f/3 -- the signature of a voice, not a whistle.
    min_subharmonic_db: float = 6.0

    # In-band level must clear the measured noise floor by this much.
    noise_margin_db: float = 10.0

    # Fallback floor used before calibration runs, in dBFS.
    noise_floor_db: float = -60.0

    calibration_seconds: float = 2.0

    # Consecutive agreeing frames required before a pitch is trusted, and the
    # width of the median filter that removes octave-jump outliers.
    persistence_frames: int = 3
    median_window: int = 3
    agreement_cents: float = 120.0


@dataclass(frozen=True)
class ThrottleConfig:
    """Sustained-pitch zones. Whistle and hold to drive; stop whistling to stop.

    Bands are contiguous and ascending. `hysteresis_cents` widens whichever band
    is currently active, so a pitch resting on a boundary cannot chatter.
    """

    backward_top: str = "F#5"       # C4..F#5  -> reverse
    forward_top: str = "A6"         # F#5..A6  -> forward
    # Above A6 -> forward fast. D7 (2349 Hz) sits comfortably inside this zone.

    hysteresis_cents: float = 60.0

    speed_forward: int = 45
    speed_fast: int = 90
    speed_backward: int = 40
    speed_turn: int = 55


@dataclass(frozen=True)
class GestureConfig:
    """Pitch sweeps steer; three short high chirps claim the goal."""

    # A "long whistle that changes pitch significantly".
    sweep_min_cents: float = 500.0          # ~5 semitones of net travel
    sweep_min_duration: float = 0.35        # seconds of continuous whistling
    sweep_max_duration: float = 2.00        # only the most recent 2 s are considered
    sweep_monotonic_fraction: float = 0.65  # share of steps that move the same way
    sweep_gap_timeout: float = 0.12         # silence this long ends a sweep

    # A completed sweep latches a pivot for this long. The turn has to outlive
    # the whistle that requested it, or it would be cancelled the instant you
    # stop whistling.
    steer_hold_seconds: float = 0.70

    # Goal command: short, high, repeated. Deliberately unlike both the
    # sustained drive tones and the sweeps, because a false positive here ends
    # the match.
    goal_chirp_count: int = 3
    goal_chirp_min_hz: float = 1760.0       # A6 and up
    goal_chirp_min_duration: float = 0.04
    goal_chirp_max_duration: float = 0.25   # stays clear of sweep_min_duration
    goal_chirp_max_gap: float = 0.45
    goal_chirp_window: float = 2.50


@dataclass(frozen=True)
class SensorConfig:
    """Colour sensor watching forward for an approaching goalie."""

    # Reflection rises as something gets close. Trigger on a sustained jump over
    # the baseline measured at match start, never on an absolute value -- venue
    # lighting will not match wherever this was tested.
    baseline_seconds: float = 1.5
    trigger_delta: float = 12.0     # reflection units (0-100) above baseline
    persistence_reads: int = 5
    poll_interval: float = 0.05


@dataclass(frozen=True)
class MqttConfig:
    broker: str = "test.mosquitto.org"
    port: int = 1883
    topic: str = "ME193/Rogers"

    # The vocabulary. Agree these with your opponent before the match; they are
    # compared case-insensitively after stripping whitespace.
    start_message: str = "start"
    ball_scored_message: str = "ball scored"
    ball_tagged_message: str = "ball tagged"


@dataclass(frozen=True)
class HardwareConfig:
    card_color: str = "red"
    card_serial: int = 1129
    use_color_sensor: bool = True


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    throttle: ThrottleConfig = field(default_factory=ThrottleConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Build a Config, applying overrides from config.local.json if present.

        The file holds one object per section, e.g.
        `{"audio": {"input_device": "Scarlett"}, "gates": {"noise_floor_db": -48}}`
        """
        path = LOCAL_CONFIG_PATH if path is None else path
        config = cls()
        if not path.exists():
            return config

        overrides = json.loads(path.read_text())
        sections = {}
        for name, values in overrides.items():
            if not hasattr(config, name):
                raise ValueError(f"config.local.json: unknown section {name!r}")
            sections[name] = replace(getattr(config, name), **values)
        return replace(config, **sections)


def throttle_bounds(throttle: ThrottleConfig) -> tuple[float, float]:
    """The two Hz boundaries between reverse, forward and forward-fast."""
    return note_to_hz(throttle.backward_top), note_to_hz(throttle.forward_top)
