"""Device selection and window overlap. No microphone is opened."""

from __future__ import annotations

import numpy as np
import pytest

from config import AudioConfig
from whistle.stream import (
    DeviceInfo,
    FrameAssembler,
    SilentInputError,
    check_audio_present,
    frames_for_seconds,
    list_input_devices,
    resolve_input_device,
)

DEVICES = [
    DeviceInfo(0, "BlackHole 16ch", 16, 48000),
    DeviceInfo(1, "MacBook Pro Microphone", 1, 44100),
    DeviceInfo(3, "Scarlett Solo USB", 2, 48000),
]


class FakePyAudio:
    """Stands in for PyAudio's device enumeration."""

    def __init__(self, infos):
        self._infos = infos

    def get_device_count(self):
        return len(self._infos)

    def get_device_info_by_index(self, index):
        return self._infos[index]


def test_only_recording_devices_are_listed():
    audio = FakePyAudio([
        {"name": "Mic", "maxInputChannels": 1, "defaultSampleRate": 44100.0},
        {"name": "Speakers", "maxInputChannels": 0, "defaultSampleRate": 48000.0},
    ])
    assert [device.name for device in list_input_devices(audio)] == ["Mic"]


def test_none_means_the_system_default():
    assert resolve_input_device(DEVICES, None) is None


def test_an_index_is_taken_as_given():
    assert resolve_input_device(DEVICES, 3) == 3


def test_a_device_can_be_chosen_by_name():
    """Names survive USB reshuffling in a way that indices do not, which matters
    for a microphone that only appears on match day."""
    assert resolve_input_device(DEVICES, "Scarlett") == 3


def test_name_matching_ignores_case_and_padding():
    assert resolve_input_device(DEVICES, "  scarlett solo  ") == 3


def test_an_unknown_name_is_an_error_that_lists_the_options():
    with pytest.raises(ValueError, match="Scarlett Solo USB"):
        resolve_input_device(DEVICES, "Focusrite")


def test_an_ambiguous_name_is_an_error_rather_than_a_guess():
    with pytest.raises(ValueError, match="more than one"):
        resolve_input_device(DEVICES, "o")


def test_devices_describe_themselves_for_the_cli():
    assert "Scarlett Solo USB" in str(DEVICES[2])


# --- window overlap ---------------------------------------------------------

def test_no_window_is_produced_until_one_is_full():
    """Otherwise the first readings would be mostly the zeros we started with."""
    assembler = FrameAssembler(frame_size=8, hop_size=4)
    assert assembler.push(np.ones(4)) is None
    assert assembler.primed is False
    assert assembler.push(np.ones(4)) is not None
    assert assembler.primed is True


def test_windows_are_the_configured_length():
    assembler = FrameAssembler(2048, 512)
    frame = None
    for _ in range(4):
        frame = assembler.push(np.zeros(512))
    assert frame.size == 2048


def test_consecutive_windows_overlap_by_frame_minus_hop():
    assembler = FrameAssembler(frame_size=8, hop_size=2)
    for block in range(4):
        assembler.push(np.full(2, float(block)))
    first = assembler.push(np.full(2, 9.0))
    second = assembler.push(np.full(2, 10.0))
    assert np.array_equal(first[2:], second[:-2]), "windows slide by one hop"


def test_the_newest_samples_land_at_the_end():
    assembler = FrameAssembler(frame_size=6, hop_size=2)
    for value in (1.0, 2.0, 3.0):
        frame = assembler.push(np.full(2, value))
    assert np.array_equal(frame, [1, 1, 2, 2, 3, 3])


def test_a_wrong_sized_block_is_an_error():
    assembler = FrameAssembler(8, 4)
    with pytest.raises(ValueError):
        assembler.push(np.zeros(3))


@pytest.mark.parametrize("frame_size, hop_size", [(8, 0), (8, -1), (8, 16)])
def test_impossible_geometry_is_rejected(frame_size, hop_size):
    with pytest.raises(ValueError):
        FrameAssembler(frame_size, hop_size)


def test_a_hop_equal_to_the_frame_means_no_overlap():
    assembler = FrameAssembler(4, 4)
    assert assembler.push(np.ones(4)) is not None


def test_frame_counts_match_real_time():
    config = AudioConfig()
    assert frames_for_seconds(2.0, config) == pytest.approx(2.0 * 44100 / 512, abs=1)
    assert frames_for_seconds(0.0, config) == 1, "always at least one window"


def test_the_real_configuration_updates_about_86_times_a_second():
    """A sanity check on the latency/accuracy split the design depends on."""
    config = AudioConfig()
    assert config.sample_rate / config.hop_size == pytest.approx(86.1, abs=0.5)
    assert config.frame_size / config.sample_rate == pytest.approx(0.046, abs=0.002)


# --- the silent-input guard -------------------------------------------------
#
# macOS fails microphone permission by handing back buffers of exact zeros
# rather than raising. Everything downstream then works perfectly on silence:
# the floor calibrates to -240 dBFS, the gates open, and no whistle ever
# arrives. That looks like a tuning problem, so it has to be caught here.

def test_real_audio_passes_the_guard():
    check_audio_present([np.array([0.0, 0.0, 1e-7])])


def test_all_zero_audio_is_rejected():
    with pytest.raises(SilentInputError):
        check_audio_present([np.zeros(2048), np.zeros(2048)])


def test_the_guard_looks_across_every_frame():
    """Silence at the start of a capture is normal; silence throughout is not."""
    frames = [np.zeros(512), np.zeros(512), np.zeros(512)]
    with pytest.raises(SilentInputError):
        check_audio_present(frames)
    frames[2][17] = -0.0004
    check_audio_present(frames)


def test_the_tiniest_real_sample_is_enough():
    """A working mic in a silent room still produces dither; a muted one does
    not. Exactly zero is the tell, so the threshold must be exact."""
    frame = np.zeros(2048)
    frame[0] = np.finfo(np.float32).tiny
    check_audio_present([frame])


def test_the_error_names_the_device_and_says_what_to_do():
    with pytest.raises(SilentInputError) as caught:
        check_audio_present([np.zeros(16)], DEVICES[1])
    message = str(caught.value)
    assert "MacBook Pro Microphone" in message
    assert "Privacy & Security" in message
    assert "reopen" in message, "the restart step is the one people miss"


def test_an_empty_capture_counts_as_silent():
    with pytest.raises(SilentInputError):
        check_audio_present([])
