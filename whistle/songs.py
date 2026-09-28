"""The death song and the song of success.

Melodies are rendered to a numpy buffer and played through pygame, so there are
no audio files to lose before a match. Rendering is pure, which is what lets the
tests check the melody without opening an output device.

A short attack and release on every note matters more than it sounds: stepping
straight to full amplitude produces a click on each note, which on a laptop
speaker is louder than the note.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from legoeducation import SOUND_PATTERN_BEEP_SINGLE

from whistle.notes import note_to_hz

DEFAULT_SAMPLE_RATE = 44100
_ENVELOPE_SECONDS = 0.012


@dataclass(frozen=True)
class Note:
    """One note, or a rest when `name` is None."""

    name: str | None
    seconds: float

    @property
    def frequency(self) -> float | None:
        return None if self.name is None else note_to_hz(self.name)


class Song(Enum):
    DEATH = "death"
    VICTORY = "victory"


# The winning song is the hook of the Rick Astley chorus, transcribed by ear in C
# major (as scale degrees 1 2 4 2 6 6 5, then 1 2 4 2 5 5 4), so that every note
# sits inside the beeper's 0-2700 Hz range. Only its first two lines are here: they
# are the part everyone recognises, and the rhythm is approximate. The losing song
# is the classic "womp womp womp womp": four notes, each a half step below the last,
# with each one held and the last held longest.
_EIGHTH = 0.22

MELODIES: dict[Song, tuple[Note, ...]] = {
    Song.VICTORY: (
        Note("C5", _EIGHTH), Note("D5", _EIGHTH), Note("F5", _EIGHTH), Note("D5", _EIGHTH),
        Note("A5", 2 * _EIGHTH), Note("A5", _EIGHTH), Note("G5", 3 * _EIGHTH),
        Note(None, _EIGHTH),
        Note("C5", _EIGHTH), Note("D5", _EIGHTH), Note("F5", _EIGHTH), Note("D5", _EIGHTH),
        Note("G5", 2 * _EIGHTH), Note("G5", _EIGHTH), Note("F5", 3 * _EIGHTH),
    ),
    Song.DEATH: (
        Note("Bb5", 0.6), Note(None, 0.08),
        Note("A5", 0.6), Note(None, 0.08),
        Note("Ab5", 0.6), Note(None, 0.08),
        Note("G5", 1.8),
    ),
}


def _envelope(length: int, sample_rate: int) -> np.ndarray:
    """Short linear attack and release, to stop each note clicking."""
    ramp = min(int(_ENVELOPE_SECONDS * sample_rate), length // 2)
    shape = np.ones(length)
    if ramp > 0:
        shape[:ramp] = np.linspace(0.0, 1.0, ramp)
        shape[-ramp:] = np.linspace(1.0, 0.0, ramp)
    return shape


def render_note(note: Note, sample_rate: int = DEFAULT_SAMPLE_RATE,
                amplitude: float = 0.35) -> np.ndarray:
    """One note as a mono float buffer in [-1, 1]. A rest renders as silence."""
    length = max(1, int(round(note.seconds * sample_rate)))
    if note.frequency is None:
        return np.zeros(length)

    t = np.arange(length) / sample_rate
    # A touch of second and third harmonic: a bare sine is thin over a laptop
    # speaker in a loud room.
    wave = (np.sin(2 * np.pi * note.frequency * t)
            + 0.30 * np.sin(4 * np.pi * note.frequency * t)
            + 0.12 * np.sin(6 * np.pi * note.frequency * t))
    wave /= 1.42     # keep the sum inside [-1, 1] before scaling
    return amplitude * wave * _envelope(length, sample_rate)


def render(melody, sample_rate: int = DEFAULT_SAMPLE_RATE,
           amplitude: float = 0.35) -> np.ndarray:
    """Render a sequence of Notes to one buffer."""
    notes = tuple(melody)
    if not notes:
        return np.zeros(0)
    return np.concatenate([render_note(n, sample_rate, amplitude) for n in notes])


def render_song(song: Song, sample_rate: int = DEFAULT_SAMPLE_RATE,
                amplitude: float = 0.35) -> np.ndarray:
    return render(MELODIES[song], sample_rate, amplitude)


def duration(melody) -> float:
    return sum(note.seconds for note in melody)


class BeepPlayer:
    """Plays the songs on the robot's own beeper.

    `beep()` takes a frequency and a pattern but no duration, so a note's length
    has to come from timing: start the beep without waiting, sleep for the note,
    then `stop_beep()`. A note longer than one beep is kept sounding by starting
    another every `sustain_seconds`; if the hub's beep is longer than that the
    restart is inaudible, and if it is shorter the note stutters. That value is
    the thing to tune by ear, and `song win` / `song lose` in the console play a
    song on demand for exactly that.

    A Bluetooth hiccup must never crash the end of a match, so a failed beep is
    counted and skipped rather than raised.
    """

    MAX_HZ = 2700       # the hardware's limit

    def __init__(self, robot, sustain_seconds: float = 0.3, octave_shift: int = 1,
                 sleep=time.sleep, log=print):
        self.robot = robot
        self.sustain_seconds = sustain_seconds
        # A small speaker is weak at low pitches and loudest higher up, and the
        # beeper has no volume control, so pitch is the one lever for loudness.
        self.octave_shift = octave_shift
        self._sleep = sleep
        self._log = log
        self.failures = 0

    def play(self, song: Song, wait: bool = True) -> float:
        melody = MELODIES[song]
        factor = 2.0 ** self.shift_for(melody)
        for note in melody:
            self._play_note(note, factor)
        return duration(melody)

    def shift_for(self, melody) -> int:
        """The octave shift actually used: as configured, but never so high that
        the top note passes the hardware's limit. Clamping single notes instead
        would flatten the melody; lowering the whole song keeps its shape."""
        top = max((n.frequency for n in melody if n.frequency is not None), default=1.0)
        room = math.floor(math.log2(self.MAX_HZ / top))
        return min(self.octave_shift, room)

    def _play_note(self, note: Note, factor: float = 1.0) -> None:
        if note.frequency is None:
            self._sleep(note.seconds)
            return

        hz = min(self.MAX_HZ, int(round(note.frequency * factor)))
        remaining = note.seconds
        while remaining > 1e-9:
            self._call("beep", pattern=SOUND_PATTERN_BEEP_SINGLE, frequency=hz,
                       blocking=False)
            step = min(self.sustain_seconds, remaining)
            self._sleep(step)
            remaining -= step
        self._call("stop_beep", blocking=False)

    def _call(self, method: str, **kwargs) -> None:
        try:
            getattr(self.robot, method)(**kwargs)
        except Exception as error:
            self.failures += 1
            if self.failures == 1:
                self._log(f"  [beeper] {method} failed: {error}")

    def close(self) -> None:
        pass


class PlayerPair:
    """Plays on the robot and the laptop together, for a louder result."""

    def __init__(self, beeper, speaker):
        self.beeper = beeper
        self.speaker = speaker

    def play(self, song: Song, wait: bool = True) -> float:
        self.speaker.play(song, wait=False)      # starts and returns at once
        return self.beeper.play(song)            # blocks for the length of the song

    def close(self) -> None:
        self.beeper.close()
        self.speaker.close()


class SongPlayer:
    """Plays the songs through pygame, initialising it only when first needed.

    The microphone is still open while a song plays, so the detector would
    happily track the melody as if it were a whistle. Callers mute the detector
    for the duration -- by the time either song plays, the match is over anyway.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, amplitude: float = 0.35):
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self._mixer = None

    def _ensure_mixer(self):
        if self._mixer is None:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame

            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)
            self._mixer = pygame.mixer
        return self._mixer

    def play(self, song: Song, wait: bool = True) -> float:
        """Play a song. Returns its length in seconds."""
        import pygame

        mixer = self._ensure_mixer()
        buffer = render_song(song, self.sample_rate, self.amplitude)
        samples = np.int16(np.clip(buffer, -1.0, 1.0) * 32767)
        sound = pygame.sndarray.make_sound(samples)
        sound.play()
        length = duration(MELODIES[song])
        if wait:
            pygame.time.wait(int(length * 1000) + 120)
        return length

    def close(self) -> None:
        if self._mixer is not None:
            self._mixer.quit()
            self._mixer = None
