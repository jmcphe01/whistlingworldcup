"""Audio samples in, motor commands out.

The other suites test one layer each. This one runs the real chain --
FrameAssembler -> PitchDetector -> Interpreter -> RobotDriver -> fake robot --
over synthesized audio, which is the only place the layers are checked against
each other. It catches the integration mistakes unit tests cannot: a window
length mismatch, a sample rate assumed in one module and configured in another,
gates tuned against a tone the interpreter then rejects.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import Config
from tests.conftest import babble
from tests.test_driver import FakeRobot
from whistle.commands import Drive
from whistle.driver import RobotDriver
from whistle.interpreter import Interpreter
from whistle.notes import note_to_hz
from whistle.pitch import PitchDetector
from whistle.stream import FrameAssembler

SAMPLE_RATE = 44100


def tone(frequency, seconds, amplitude=0.3, start_phase=0.0):
    """A whistle-like pure tone, phase-continuous when chained."""
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * frequency * t + start_phase)


def sweep(start_hz, end_hz, seconds, amplitude=0.3):
    """A linear glide, built from its instantaneous phase."""
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    frequency = start_hz + (end_hz - start_hz) * t / max(t[-1], 1e-9)
    return amplitude * np.sin(2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE)


def quiet(seconds, level=0.0005):
    rng = np.random.default_rng(3)
    return level * rng.standard_normal(int(seconds * SAMPLE_RATE))


class Rig:
    """The whole control chain, driven by raw audio."""

    def __init__(self, config: Config | None = None, noise_floor_db: float = -70.0):
        self.config = config or Config()
        self.detector = PitchDetector(self.config.audio.sample_rate,
                                      self.config.audio.frame_size, self.config.gates)
        self.detector.set_noise_floor(noise_floor_db)
        self.assembler = FrameAssembler(self.config.audio.frame_size,
                                        self.config.audio.hop_size)
        self.interpreter = Interpreter(self.config)
        self.robot = FakeRobot()
        self.driver = RobotDriver(self.robot, self.config.throttle, resend_interval=1e9)
        self.commands: list[Drive] = []
        self.goals = 0
        self._samples = 0

    def feed(self, audio: np.ndarray) -> "Rig":
        hop = self.config.audio.hop_size
        blocks = len(audio) // hop
        for index in range(blocks):
            block = audio[index * hop:(index + 1) * hop]
            frame = self.assembler.push(block)
            self._samples += hop
            if frame is None:
                continue
            now = self._samples / self.config.audio.sample_rate
            intent = self.interpreter.update(now, self.detector.analyse(frame))
            if intent.goal_whistle:
                self.goals += 1
            self.driver.apply(now, intent.drive)
            self.commands.append(intent.drive)
        return self

    @property
    def tank_calls(self):
        return self.robot.tank_calls

    def saw(self, drive: Drive) -> bool:
        return drive in self.commands

    @property
    def final(self) -> Drive:
        return self.commands[-1]


# --- throttle ---------------------------------------------------------------

@pytest.mark.parametrize(
    "note, expected",
    [("D7", Drive.FORWARD_FAST), ("C6", Drive.FORWARD), ("C4", Drive.BACKWARD)],
)
def test_whistling_a_note_drives_the_motors(note, expected):
    rig = Rig().feed(tone(note_to_hz(note), 0.5))
    assert rig.final is expected
    assert rig.tank_calls[-1] != (0, 0)


def test_the_motors_stop_when_the_whistle_stops():
    rig = Rig().feed(np.concatenate([tone(note_to_hz("C6"), 0.5), quiet(0.4)]))
    assert rig.final is Drive.STOP
    assert rig.tank_calls[-1] == (0, 0)


def test_going_fast_really_is_faster():
    forward = Rig().feed(tone(note_to_hz("C6"), 0.5)).tank_calls[-1]
    fast = Rig().feed(tone(note_to_hz("D7"), 0.5)).tank_calls[-1]
    assert fast[0] > forward[0]


def test_reverse_runs_the_tracks_backwards():
    assert Rig().feed(tone(note_to_hz("C4"), 0.5)).tank_calls[-1][0] < 0


def test_a_run_through_every_throttle_zone():
    rig = Rig().feed(np.concatenate([
        tone(note_to_hz("C6"), 0.4), quiet(0.2),
        tone(note_to_hz("D7"), 0.4), quiet(0.2),
        tone(note_to_hz("C4"), 0.4), quiet(0.2),
    ]))
    assert rig.saw(Drive.FORWARD)
    assert rig.saw(Drive.FORWARD_FAST)
    assert rig.saw(Drive.BACKWARD)
    assert rig.final is Drive.STOP


# --- steering ---------------------------------------------------------------

def test_a_real_rising_glide_steers_right():
    rig = Rig().feed(np.concatenate([
        sweep(note_to_hz("A4"), note_to_hz("A5"), 0.8), quiet(0.2)]))
    assert rig.saw(Drive.TURN_RIGHT)
    assert (55, -55) in rig.tank_calls


def test_a_real_falling_glide_steers_left():
    rig = Rig().feed(np.concatenate([
        sweep(note_to_hz("A5"), note_to_hz("A4"), 0.8), quiet(0.2)]))
    assert rig.saw(Drive.TURN_LEFT)


def test_a_steady_note_never_steers():
    rig = Rig().feed(tone(note_to_hz("C6"), 2.0))
    assert not any(command.is_turn for command in rig.commands)


def test_the_two_glides_steer_opposite_ways():
    up = Rig().feed(sweep(note_to_hz("A4"), note_to_hz("A6"), 1.0))
    down = Rig().feed(sweep(note_to_hz("A6"), note_to_hz("A4"), 1.0))
    assert up.saw(Drive.TURN_RIGHT) and not up.saw(Drive.TURN_LEFT)
    assert down.saw(Drive.TURN_LEFT) and not down.saw(Drive.TURN_RIGHT)


# --- the goal whistle -------------------------------------------------------

def test_three_real_chirps_claim_the_goal():
    chirps = [x for _ in range(3)
              for x in (tone(note_to_hz("C7"), 0.14), quiet(0.16))]
    assert Rig().feed(np.concatenate(chirps)).goals == 1


def test_a_whole_driving_session_never_claims_the_goal():
    """The strongest safety check in the suite: every drive and steer command,
    back to back, must not score."""
    rig = Rig().feed(np.concatenate([
        tone(note_to_hz("C6"), 0.8), quiet(0.3),
        tone(note_to_hz("D7"), 0.8), quiet(0.3),
        tone(note_to_hz("C4"), 0.8), quiet(0.3),
        sweep(note_to_hz("A4"), note_to_hz("A6"), 1.0), quiet(0.3),
        sweep(note_to_hz("A6"), note_to_hz("A4"), 1.0), quiet(0.3),
    ]))
    assert rig.goals == 0


# --- noise ------------------------------------------------------------------

def test_a_noisy_room_alone_does_not_move_the_car():
    """The one that matters on the day: babble at full tilt, no whistle."""
    room = np.concatenate([babble(amplitude=0.5, seed=s, n=4096) for s in range(30)])
    rig = Rig(noise_floor_db=-70.0).feed(room)
    assert rig.final is Drive.STOP
    assert not any(command is not Drive.STOP for command in rig.commands)


def test_a_whistle_over_a_noisy_room_still_drives():
    room = np.concatenate([babble(amplitude=0.22, seed=s, n=4096) for s in range(12)])
    whistle = tone(note_to_hz("C6"), len(room) / SAMPLE_RATE, amplitude=0.28)
    rig = Rig(noise_floor_db=-70.0).feed(room + whistle)
    assert rig.final is Drive.FORWARD


def test_silence_sends_no_needless_commands():
    """BLE budget: a car sitting still should not be chattering at the motors."""
    rig = Rig().feed(quiet(1.0))
    assert len(rig.tank_calls) <= 1
