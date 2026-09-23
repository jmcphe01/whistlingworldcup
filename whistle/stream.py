"""The PyAudio input stream and the sliding window it feeds.

The PortAudio callback runs on a real-time thread, so it does exactly one thing:
hand the block off and return. All analysis happens on the consumer side.

Windows overlap: the window is `frame_size` long because that sets frequency
accuracy, but a new one is produced every `hop_size` samples because that sets
latency. At 44.1 kHz with a 2048/512 split that is 46 ms of audio analysed every
11.6 ms, so pitch updates about 86 times a second on 46 ms of evidence.

When the consumer falls behind, the oldest block is dropped rather than the
newest. Stale audio is worse than missing audio for a car you are steering.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from config import AudioConfig


class SilentInputError(RuntimeError):
    """The input device delivered nothing but digital silence.

    This is worth its own error because of how macOS fails: when an application
    has not been granted microphone access, PortAudio still opens the stream and
    still delivers buffers, and every sample in them is exactly zero. Nothing
    raises. Without this check the whole program runs perfectly on silence -- it
    calibrates a noise floor of -240 dBFS, opens its gates, and then never hears
    a whistle, which looks like a tuning problem rather than a permissions one.
    """


def check_audio_present(frames, device: "DeviceInfo | None" = None) -> None:
    """Raise SilentInputError if every sample across `frames` is exactly zero.

    Exactly zero is the tell. A working microphone in a silent room still
    produces dither and self-noise; a muted or unauthorised one produces
    mathematical silence.
    """
    if any(np.any(np.asarray(frame)) for frame in frames):
        return

    name = "the input device" if device is None else f"'{device.name}'"
    raise SilentInputError(
        f"{name} delivered only digital silence.\n"
        "\n"
        "On macOS this almost always means the application running Python has "
        "not been granted microphone access. PortAudio opens the stream and "
        "hands back zeroed buffers rather than reporting an error.\n"
        "\n"
        "  1. System Settings -> Privacy & Security -> Microphone\n"
        "  2. Enable the app you are running this from (the terminal app, your "
        "editor, or Claude).\n"
        "  3. Quit that app completely and reopen it. macOS does not apply a "
        "new permission to an already-running process, so this step is not "
        "optional.\n"
        "\n"
        "If the app is not in the list, run this from a plain Terminal window "
        "once so macOS raises the prompt. Check the device with "
        "`python main.py --list-devices`; a virtual device such as BlackHole "
        "will also read as silent when nothing is routed into it."
    )


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    channels: int
    sample_rate: int

    def __str__(self) -> str:
        return f"[{self.index}] {self.name}  ({self.channels} ch, {self.sample_rate} Hz)"


def list_input_devices(audio) -> list[DeviceInfo]:
    """Every device that can record, as PortAudio sees it."""
    devices = []
    for index in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(index)
        if info["maxInputChannels"] > 0:
            devices.append(DeviceInfo(
                index=index,
                name=str(info["name"]),
                channels=int(info["maxInputChannels"]),
                sample_rate=int(info["defaultSampleRate"]),
            ))
    return devices


def resolve_input_device(devices: list[DeviceInfo], spec: int | str | None) -> int | None:
    """Turn a config value into a device index.

    `None` means the system default. An int is used as-is. A string matches a
    device name case-insensitively, so the directional microphone can be named
    in config rather than pinned to an index that shifts when USB devices come
    and go.
    """
    if spec is None or isinstance(spec, int):
        return spec

    needle = spec.strip().lower()
    matches = [device for device in devices if needle in device.name.lower()]
    if not matches:
        available = ", ".join(repr(device.name) for device in devices) or "none"
        raise ValueError(f"no input device matching {spec!r}. Available: {available}")
    if len(matches) > 1:
        names = ", ".join(repr(device.name) for device in matches)
        raise ValueError(f"{spec!r} matches more than one input device: {names}")
    return matches[0].index


class FrameAssembler:
    """Turns hop-sized blocks into overlapping analysis windows.

    Kept separate from PyAudio so the overlap logic can be tested without an
    audio device.
    """

    def __init__(self, frame_size: int, hop_size: int):
        if not 0 < hop_size <= frame_size:
            raise ValueError("hop_size must be positive and no larger than frame_size")
        self.frame_size = frame_size
        self.hop_size = hop_size
        self._buffer = np.zeros(frame_size, dtype=np.float32)
        self._filled = 0

    @property
    def primed(self) -> bool:
        """Has a full window of real audio arrived yet?"""
        return self._filled >= self.frame_size

    def push(self, block: np.ndarray) -> np.ndarray | None:
        """Add one block; return the new window, or None while still priming."""
        block = np.asarray(block, dtype=np.float32)
        if block.size != self.hop_size:
            raise ValueError(f"expected {self.hop_size} samples, got {block.size}")

        self._buffer = np.roll(self._buffer, -self.hop_size)
        self._buffer[-self.hop_size:] = block
        self._filled += self.hop_size
        return self._buffer.copy() if self.primed else None


class AudioStream:
    """Microphone -> overlapping analysis windows. Use as a context manager."""

    def __init__(self, config: AudioConfig | None = None, max_queued_blocks: int = 8):
        self.config = config or AudioConfig()
        self._audio = None
        self._stream = None
        self._blocks: queue.Queue = queue.Queue(maxsize=max_queued_blocks)
        self._assembler = FrameAssembler(self.config.frame_size, self.config.hop_size)
        self.dropped_blocks = 0
        self.device: DeviceInfo | None = None

    def open(self) -> "AudioStream":
        import pyaudio

        self._audio = pyaudio.PyAudio()
        devices = list_input_devices(self._audio)
        index = resolve_input_device(devices, self.config.input_device)
        if index is None:
            index = int(self._audio.get_default_input_device_info()["index"])
        self.device = next((d for d in devices if d.index == index), None)

        self._stream = self._audio.open(
            format=pyaudio.paFloat32,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            input_device_index=index,
            frames_per_buffer=self.config.hop_size,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        return self

    def _callback(self, in_data, frame_count, time_info, status):
        """Real-time thread: hand the block off and return. Nothing else."""
        import pyaudio

        try:
            self._blocks.put_nowait(in_data)
        except queue.Full:
            # Drop the oldest, keep the newest: stale audio is worse than none.
            try:
                self._blocks.get_nowait()
                self._blocks.put_nowait(in_data)
            except (queue.Empty, queue.Full):
                pass
            self.dropped_blocks += 1
        return (None, pyaudio.paContinue)

    def frames(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        """Yield analysis windows as they become available."""
        while True:
            try:
                raw = self._blocks.get(timeout=timeout)
            except queue.Empty:
                if self._stream is None:
                    return
                continue
            frame = self._assembler.push(np.frombuffer(raw, dtype=np.float32))
            if frame is not None:
                yield frame

    def read_frames(self, count: int, timeout: float = 5.0) -> list[np.ndarray]:
        """Collect a fixed number of windows. Used by calibration."""
        collected = []
        for frame in self.frames(timeout=timeout):
            collected.append(frame)
            if len(collected) >= count:
                break
        return collected

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._audio is not None:
            self._audio.terminate()
            self._audio = None

    def __enter__(self) -> "AudioStream":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()


def frames_for_seconds(seconds: float, config: AudioConfig) -> int:
    """How many analysis windows make up roughly this much audio."""
    return max(1, int(round(seconds * config.sample_rate / config.hop_size)))
