"""Note names, frequencies and cents.

Pitch is perceived logarithmically, so every comparison in this project happens
in cents (1/100 of a semitone, 1200 to the octave) rather than in Hz. A 20 Hz
error is nothing at C7 and a semitone at C4; cents make the two comparable.
"""

from __future__ import annotations

import math

A4_HZ = 440.0
A4_MIDI = 69

_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_SEMITONE_OF = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}


def note_to_midi(name: str) -> int:
    """'A4' -> 69, 'F#5' -> 78, 'Bb3' -> 58. Case-insensitive on the letter."""
    text = name.strip()
    if not text:
        raise ValueError("empty note name")

    letter = text[0].upper()
    if letter not in _SEMITONE_OF:
        raise ValueError(f"bad note name {name!r}: {letter!r} is not A-G")

    index = 1
    accidental = 0
    while index < len(text) and text[index] in "#b♯♭":
        accidental += 1 if text[index] in "#♯" else -1
        index += 1

    octave_text = text[index:]
    try:
        octave = int(octave_text)
    except ValueError:
        raise ValueError(f"bad note name {name!r}: {octave_text!r} is not an octave")

    # MIDI octaves start at C, and C4 is MIDI 60.
    return 12 * (octave + 1) + _SEMITONE_OF[letter] + accidental


def midi_to_hz(midi: float) -> float:
    return A4_HZ * 2.0 ** ((midi - A4_MIDI) / 12.0)


def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        raise ValueError("frequency must be positive")
    return A4_MIDI + 12.0 * math.log2(hz / A4_HZ)


def note_to_hz(name: str) -> float:
    """'A4' -> 440.0, 'D7' -> 2349.32..."""
    return midi_to_hz(note_to_midi(name))


def cents_between(hz: float, reference_hz: float) -> float:
    """Signed distance in cents. Positive means `hz` is the higher pitch."""
    if hz <= 0 or reference_hz <= 0:
        raise ValueError("frequencies must be positive")
    return 1200.0 * math.log2(hz / reference_hz)


def cents_above_a4(hz: float) -> float:
    """Absolute log-pitch scale used by the sweep detector."""
    return cents_between(hz, A4_HZ)


def describe(hz: float) -> tuple[str, float]:
    """Nearest note name and the signed cents deviation from it.

    `describe(445.0)` -> ('A4', 19.6) reads as "A4, 20 cents sharp".
    """
    midi = hz_to_midi(hz)
    nearest = int(round(midi))
    name = f"{_SHARP_NAMES[nearest % 12]}{nearest // 12 - 1}"
    return name, (midi - nearest) * 100.0
