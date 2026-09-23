#!/usr/bin/env python3
"""Measure this room and this whistle, then write config.local.json.

Run this in the competition room before the match. Two measurements:

  noise floor   the in-band level of the room with nobody whistling, which the
                level gate then works from. An absolute threshold tuned in a
                quiet lab either fires on everything or nothing in a loud room.

  whistle range your actual lowest and highest comfortable whistle. The search
                band floor is the single most valuable noise-rejection setting
                available, because the lower it goes the more speech energy it
                lets in -- so it is worth setting from a measurement rather than
                a guess.

Nothing here touches the robot or the network.

    python calibrate.py --list-devices
    python calibrate.py
    python calibrate.py --device Scarlett
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace

from config import Config, LOCAL_CONFIG_PATH, throttle_bounds
from whistle.notes import describe
from whistle.pitch import PitchDetector, measure_noise_floor
from whistle.stream import (
    AudioStream,
    SilentInputError,
    check_audio_present,
    frames_for_seconds,
    list_input_devices,
)

BAND_MARGIN_CENTS = 200.0   # headroom below your lowest whistle, ~2 semitones


def with_device(config: Config, device: str) -> Config:
    """Apply a --device argument, accepting either an index or a name."""
    spec: int | str = int(device) if device.strip().isdigit() else device
    return replace(config, audio=replace(config.audio, input_device=spec))


def print_devices() -> None:
    import pyaudio

    audio = pyaudio.PyAudio()
    try:
        devices = list_input_devices(audio)
        default = int(audio.get_default_input_device_info()["index"])
    finally:
        audio.terminate()

    print("Input devices:")
    for device in devices:
        marker = "  <- default" if device.index == default else ""
        print(f"  {device}{marker}")
    print("\nPut the name (or index) in config.local.json as audio.input_device,")
    print('e.g. {"audio": {"input_device": "Scarlett"}}')


def wait_for_return(prompt: str) -> None:
    try:
        input(prompt)
    except EOFError:
        print()


def measure_room(stream: AudioStream, detector: PitchDetector, config: Config) -> float:
    seconds = config.gates.calibration_seconds
    wait_for_return(f"Silence please. Press Return, then stay quiet for {seconds:.0f}s... ")
    frames = stream.read_frames(frames_for_seconds(seconds, config.audio))
    check_audio_present(frames, stream.device)
    floor = measure_noise_floor(frames, detector)
    print(f"  noise floor: {floor:6.1f} dBFS  "
          f"(gate opens at {floor + config.gates.noise_margin_db:6.1f} dBFS)")
    return floor


def measure_whistle(stream: AudioStream, detector: PitchDetector, config: Config,
                    label: str, seconds: float = 3.0) -> float | None:
    wait_for_return(f"Press Return, then whistle your {label} note for {seconds:.0f}s... ")
    frames = stream.read_frames(frames_for_seconds(seconds, config.audio))

    pitches = [r.frequency for r in (detector.analyse(f) for f in frames)
               if r.voiced and r.frequency is not None]
    if len(pitches) < 10:
        print(f"  heard almost nothing ({len(pitches)} good frames). "
              "Whistle louder, or closer to the mic.")
        return None

    # Median over the sustained part: the attack and release of a whistle slide.
    pitch = statistics.median(pitches)
    name, cents = describe(pitch)
    print(f"  {label}: {pitch:7.1f} Hz  ({name} {cents:+.0f}c)  "
          f"from {len(pitches)}/{len(frames)} frames")
    return pitch


def recommend(lowest: float | None, highest: float | None, config: Config) -> dict:
    """Turn the measurements into config overrides worth writing."""
    overrides: dict[str, dict] = {}
    if lowest is None or highest is None:
        return overrides

    floor = lowest * 2.0 ** (-BAND_MARGIN_CENTS / 1200.0)
    ceiling = highest * 2.0 ** (BAND_MARGIN_CENTS / 1200.0)
    overrides["gates"] = {"min_hz": round(floor, 1), "max_hz": round(ceiling, 1)}

    print(f"\n  whistle range: {lowest:.0f}-{highest:.0f} Hz "
          f"({describe(lowest)[0]} to {describe(highest)[0]})")
    print(f"  search band:   {floor:.0f}-{ceiling:.0f} Hz "
          f"(your range plus {BAND_MARGIN_CENTS:.0f} cents either side)")

    if floor < 500.0:
        print("\n  Note: a search band reaching below ~500 Hz overlaps the range where")
        print("  speech harmonics live, so the shape gates have to work harder there.")
        print("  If you can comfortably whistle higher, raising the floor is the")
        print("  cheapest noise immunity available.")

    backward_top, forward_top = throttle_bounds(config.throttle)
    if not lowest < backward_top < forward_top < highest:
        print("\n  Warning: the configured throttle zones do not all sit inside your")
        print(f"  whistle range. Zone edges are {backward_top:.0f} Hz and "
              f"{forward_top:.0f} Hz; you measured {lowest:.0f}-{highest:.0f} Hz.")
        print("  Adjust throttle.backward_top / forward_top in config.py.")
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-devices", action="store_true",
                        help="show the input devices and exit")
    parser.add_argument("--device", help="input device name or index to calibrate with")
    parser.add_argument("--write", action="store_true",
                        help="write config.local.json without asking")
    args = parser.parse_args(argv)

    if args.list_devices:
        print_devices()
        return 0

    config = Config.load()
    if args.device is not None:
        config = with_device(config, args.device)

    detector = PitchDetector(config.audio.sample_rate, config.audio.frame_size,
                             config.gates)

    try:
        with AudioStream(config.audio) as stream:
            print(f"Listening on {stream.device}\n")
            floor = measure_room(stream, detector, config)
            detector.set_noise_floor(floor)

            print()
            lowest = measure_whistle(stream, detector, config, "lowest")
            highest = measure_whistle(stream, detector, config, "highest")
    except SilentInputError as error:
        print(f"\nNo audio is reaching the program.\n\n{error}")
        return 1

    overrides = recommend(lowest, highest, config)
    overrides.setdefault("gates", {})["noise_floor_db"] = round(floor, 1)
    if config.audio.input_device is not None:
        overrides["audio"] = {"input_device": config.audio.input_device}

    print(f"\nProposed {LOCAL_CONFIG_PATH.name}:")
    print(json.dumps(overrides, indent=2))

    if not args.write:
        try:
            if input("\nWrite it? [y/N] ").strip().lower() not in {"y", "yes"}:
                print("Not written.")
                return 0
        except EOFError:
            print("\nNot written.")
            return 0

    LOCAL_CONFIG_PATH.write_text(json.dumps(overrides, indent=2) + "\n")
    print(f"Wrote {LOCAL_CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
