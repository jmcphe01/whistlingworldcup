"""Smoke tests: the virtual environment has everything the project needs.

These do not touch hardware. They only prove the imports resolve and that
PortAudio can enumerate at least one input device.
"""

import importlib

import pytest

REQUIRED_MODULES = [
    "pyaudio",
    "numpy",
    "scipy.signal",
    "paho.mqtt.client",
    "legoeducation",
    "pygame",
    "matplotlib",
]


@pytest.mark.parametrize("name", REQUIRED_MODULES)
def test_module_importable(name):
    assert importlib.import_module(name) is not None


def test_lelib_exposes_expected_classes():
    import lelib

    for cls in ("singleMotor", "doubleMotor", "controller", "colorSensor"):
        assert hasattr(lelib, cls), f"lelib is missing {cls}"


def test_portaudio_sees_an_input_device():
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        inputs = [
            pa.get_device_info_by_index(i)
            for i in range(pa.get_device_count())
        ]
        assert any(d["maxInputChannels"] > 0 for d in inputs), "no input devices"
    finally:
        pa.terminate()
