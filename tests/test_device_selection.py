"""The microphone dropdown, and the command channel behind it.

The monitor runs in its own process and must never touch the audio device
itself. Picking a microphone only posts a request; the process that owns the
stream decides what to do with it. These tests cover both ends of that channel
without opening a window or a microphone.
"""

from __future__ import annotations

import multiprocessing
import pickle

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import main   # noqa: E402
from monitor import (  # noqa: E402
    DeviceSelector,
    Snapshot,
    device_labels,
    label_to_index,
    shorten,
)
from whistle.stream import DeviceInfo  # noqa: E402

DEVICES = [
    DeviceInfo(0, "BlackHole 16ch", 16, 48000),
    DeviceInfo(1, "MacBook Pro Microphone", 1, 44100),
    DeviceInfo(3, "Scarlett Solo USB Audio Interface", 2, 48000),
]


@pytest.fixture
def selector():
    fig, ax = plt.subplots()
    picked: list[int] = []
    widget = DeviceSelector(ax, DEVICES, current_index=1, on_select=picked.append)
    widget.picked = picked
    yield widget
    plt.close(fig)


# --- labels -----------------------------------------------------------------

def test_every_device_gets_a_label():
    assert len(device_labels(DEVICES)) == len(DEVICES)


def test_labels_carry_the_index_so_duplicates_stay_distinct():
    """Two identical interfaces would otherwise produce the same label and the
    selection would be ambiguous."""
    twins = [DeviceInfo(1, "USB Audio", 2, 48000), DeviceInfo(4, "USB Audio", 2, 48000)]
    labels = device_labels(twins)
    assert labels[0] != labels[1]
    assert label_to_index(labels[1], twins) == 4


def test_a_label_maps_back_to_its_device_index():
    for device, label in zip(DEVICES, device_labels(DEVICES)):
        assert label_to_index(label, DEVICES) == device.index


def test_an_unknown_label_is_an_error():
    with pytest.raises(KeyError):
        label_to_index("[9] Nothing", DEVICES)


def test_long_names_are_trimmed_but_stay_recognisable():
    trimmed = shorten("Scarlett Solo USB Audio Interface", 20)
    assert len(trimmed) <= 20
    assert trimmed.startswith("Scarlett")


def test_short_names_are_left_alone():
    assert shorten("Built-in", 20) == "Built-in"


# --- the widget -------------------------------------------------------------

def test_it_starts_collapsed(selector):
    assert selector.open is False
    assert selector.list_ax.get_visible() is False


def test_the_collapsed_list_cannot_swallow_clicks(selector):
    """Hiding an Axes does not stop its widget receiving events, so the list is
    deactivated as well. Otherwise the invisible panel would eat presses in that
    corner of the window."""
    assert selector.radio.active is False
    selector._toggle(None)
    assert selector.radio.active is True


def test_clicking_the_button_expands_and_collapses(selector):
    selector._toggle(None)
    assert selector.list_ax.get_visible() is True
    selector._toggle(None)
    assert selector.list_ax.get_visible() is False


def test_the_button_names_the_current_microphone(selector):
    assert "MacBook" in selector.button.label.get_text()


def test_choosing_a_device_reports_its_index(selector):
    selector._toggle(None)
    selector.radio.set_active(2)
    assert selector.picked == [3]


def test_choosing_a_device_collapses_the_list_again(selector):
    selector._toggle(None)
    selector.radio.set_active(0)
    assert selector.open is False


def test_the_button_follows_the_selection(selector):
    selector.radio.set_active(2)
    assert "Scarlett" in selector.button.label.get_text()


def test_it_opens_on_the_device_already_in_use(selector):
    assert selector.current_index == 1


def test_an_unknown_current_device_falls_back_to_the_first():
    fig, ax = plt.subplots()
    try:
        widget = DeviceSelector(ax, DEVICES, current_index=99, on_select=lambda _: None)
        assert widget.current_index == DEVICES[0].index
    finally:
        plt.close(fig)


# --- the command channel ----------------------------------------------------

def queue_of(*items):
    import queue as _queue

    q = _queue.Queue()      # no writer thread, so a burst is all there at once
    for item in items:
        q.put(item)
    return q


def test_no_queue_means_no_requests():
    assert main.collect_commands(None) == []
    assert main.collect_commands(queue_of()) == []


def test_a_request_is_picked_up():
    assert main.collect_commands(queue_of(("device", 3))) == [("device", 3)]


def test_only_the_newest_device_is_honoured():
    """Clicking three devices quickly should land on the last one, not replay
    each in turn -- every switch costs a stream reopen and a recalibration."""
    result = main.collect_commands(queue_of(("device", 0), ("device", 1), ("device", 3)))
    assert result == [("device", 3)]


def test_only_the_newest_sensitivity_is_honoured():
    """One drag of the slider posts a stream of values."""
    result = main.collect_commands(
        queue_of(*[("sensitivity", v) for v in (5.0, 6.0, 7.0, 7.5)]))
    assert result == [("sensitivity", 7.5)]


def test_discrete_requests_are_all_kept_in_order():
    """A role change then a reset are two events, not one setting."""
    items = [("role", "goalie"), ("topic", "ME193/Test"), ("reset", None)]
    assert main.collect_commands(queue_of(*items)) == items


def test_requests_from_several_queues_are_combined():
    monitor = queue_of(("role", "goalie"))
    console = queue_of(("opponent", "start"))
    assert main.collect_commands(monitor, console) == [
        ("role", "goalie"), ("opponent", "start")]


def test_malformed_items_are_skipped():
    assert main.collect_commands(queue_of("junk", ("reset", None))) == [("reset", None)]


def test_the_queue_is_drained_so_a_request_fires_once():
    q = queue_of(("device", 3))
    main.collect_commands(q)
    assert main.collect_commands(q) == []


def test_the_widget_posts_straight_onto_a_real_queue():
    """End to end across the boundary: a click in the monitor becomes a request
    the audio loop can read."""
    import time

    commands = multiprocessing.Queue()
    fig, ax = plt.subplots()
    try:
        widget = DeviceSelector(ax, DEVICES, current_index=0,
                                on_select=lambda i: commands.put_nowait(("device", i)))
        widget.radio.set_active(1)
        result = []
        for _ in range(200):     # a multiprocessing queue is fed by a writer thread
            result = main.collect_commands(commands)
            if result:
                break
            time.sleep(0.01)
        assert result == [("device", 1)]
    finally:
        plt.close(fig)


# --- the process boundary ---------------------------------------------------

def test_devices_survive_pickling():
    """They are passed to the monitor process, which on macOS means spawn."""
    assert pickle.loads(pickle.dumps(DEVICES)) == DEVICES


def test_a_snapshot_carries_the_live_device_name():
    """The button shows what was clicked; the readout shows what is actually
    open, so a rejected switch is visible rather than silent."""
    import numpy as np

    snapshot = Snapshot(
        t=0.0, spectrum_db=np.zeros(4), frequency=None, note=None, cents_off=None,
        level_db=-60.0, noise_floor_db=-70.0, peak_to_median_db=1.0,
        peak_to_second_db=1.0, subharmonic_db=1.0, reject_reason="too quiet",
        drive="stop", steering=False, phase="running", role="ball",
        device="MacBook Pro Microphone",
    )
    assert pickle.loads(pickle.dumps(snapshot)).device == "MacBook Pro Microphone"
