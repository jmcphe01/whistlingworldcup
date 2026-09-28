"""The death song and the song of success.

Melodies are rendered to a numpy buffer and played through pygame, so there are
no audio files to lose before a match. Rendering is pure, which is what lets the
tests check the melody without opening an output device.

A short attack and release on every note matters more than it sounds: stepping
straight to full amplitude produces a click on each note, which on a laptop
speaker is louder than the note.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import numpy as np

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


# A slow descending minor line, and a rising fanfare. Both deliberately short:
# nobody wants to wait four seconds to find out who won.
MELODIES: dict[Song, tuple[Note, ...]] = {
    Song.DEATH: (
        Note("A4", 0.22), Note("G#4", 0.22), Note("G4", 0.22), Note("F#4", 0.22),
        Note(None, 0.06), Note("F4", 0.30), Note("D4", 0.70),
    ),
    Song.VICTORY: (
        Note("C5", 0.13), Note("E5", 0.13), Note("G5", 0.13), Note("C6", 0.34),
        Note(None, 0.05), Note("G5", 0.13), Note("C6", 0.55),
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
