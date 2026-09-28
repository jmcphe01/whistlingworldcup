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

    backward_top: str = "A5"        # C4..A5  -> reverse
    forward_top: str = "A6"         # A5..A6  -> forward
    # Above A6 -> forward fast. D7 (2349 Hz) sits comfortably inside this zone.

    hysteresis_cents: float = 60.0

    speed_forward: int = 45
    speed_fast: int = 90
    speed_backward: int = 40
    # Reverse forward and backward for the double motor. Only the linear motion:
    # pivots are left as they are, so slides still turn the way they always did.
    # Set False if the car is ever remounted the other way round.
    invert_drive: bool = True

    speed_turn: int = 28        # deliberately gentle: turns are easy to overshoot


@dataclass(frozen=True)
class GestureConfig:
    """Holding a pitch drives; sliding it steers.

    The two are told apart by how fast the pitch is moving, not by where it sits,
    so a slide never doubles as a throttle command on its way through the zones.
    There is a deliberate dead band between `hold_max_rate_cents` and
    `slide_min_rate_cents` where neither applies and the car simply stops --
    better an ambiguous whistle does nothing than the wrong thing.
    """

    motion_window_seconds: float = 0.30     # how much recent pitch to judge from
    motion_gap_timeout: float = 0.12        # silence this long starts a new gesture

    # Holding. The rate test matters as much as the spread one: the first
    # moments of a slide have barely moved yet, and without it they would read
    # as a held note and lurch the car forward before the turn took over.
    hold_tolerance_cents: float = 100.0
    hold_max_rate_cents: float = 250.0      # cents per second

    # Sliding.
    slide_min_rate_cents: float = 400.0     # cents per second, about 4 semitones
    slide_min_duration: float = 0.15        # enough to tell a slide from a wobble
    slide_monotonic_fraction: float = 0.60  # share of steps going the same way

    # The turn outlives the slide by this much, purely to bridge the odd dropped
    # frame. Steering is otherwise live: you turn for exactly as long as you slide.
    steer_release_seconds: float = 0.20

    # Goal command: a warble, a series of ups and downs. "left" is the falling
    # slide and "right" the rising one, matching how each steers. The default is
    # three humps, each one up then down, so six legs: a whistle that is easy to
    # tell from a single steering slide, and tolerant of the pitch line dropping out
    # at the peaks, which a real whistle does. At least three legs are required
    # regardless: a single slide is already a steering command.
    goal_pattern: tuple[str, ...] = ("right", "left", "right", "left", "right", "left")
    goal_leg_cents: float = 350.0           # each leg must travel this far (~3.5 semitones)
    goal_max_leg_seconds: float = 1.2       # slower than this is a drift, not a warble
    goal_window: float = 5.0                # the whole pattern must fit in this
    goal_gap_timeout: float = 0.7           # a skip this long is forgiven; longer abandons it

    # The warble only counts in the forward and fast zones: pitches below the
    # reverse ceiling (less this margin) are ignored by the goal detector, so voices
    # and low hum cannot contribute legs. None turns the restriction off.
    goal_floor_margin_cents: float | None = 150.0


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

    # The vocabulary. Compared case-insensitively after stripping whitespace, and
    # editable while the program runs from the monitor. Only the last two are ever
    # sent by this program: `start` comes from whoever starts the match.
    start_message: str = "start"
    goal_message: str = "goal"          # the ball scored; the goalie hears this
    tagged_message: str = "tagged"      # the goalie reached the ball; the goalie hears this

    # The single motor command, sent by a partner as pivot[<angle>], e.g. pivot[90] or
    # pivot[-45]. It shares the topic with the match messages and works in either
    # role, at any time.
    pivot_message: str = "pivot"


@dataclass(frozen=True)
class HardwareConfig:
    card_color: str = "red"
    card_serial: int = 1129
    use_color_sensor: bool = True
    # How often a held note is re-beeped. The hub's beep has no duration, so a
    # long note is a run of beeps; tune this by ear with `song win` / `song lose`.
    beep_sustain_seconds: float = 0.3
    # Octaves to transpose the songs up on the beeper. The beeper has no volume
    # control, and a small speaker is louder higher up; capped so no note passes
    # the hardware's 2700 Hz limit. 0 plays them as written.
    beep_octave_shift: int = 1

    # The single motor. A positive angle turns clockwise and a negative one the other
    # way; set single_motor_invert if that is the wrong way round for how it is mounted.
    single_motor_enabled: bool = True
    single_motor_speed: int = 50
    single_motor_invert: bool = False
    single_motor_max_degrees: int = 3600       # ten turns; refuses a mistyped huge angle


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
