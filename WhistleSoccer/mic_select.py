"""Popup dropdown for picking which microphone to listen on.

Useful when the machine sees several input devices (built-in array,
USB headset, DroidCam, etc.) and you want the one that's actually closest
to you / away from the workshop noise, instead of whatever the OS
defaults to.

Uses PyAudio (not sounddevice) to enumerate devices, since the index it
returns is meant to be passed straight to `pyaudio.PyAudio().open(...,
input_device_index=...)` -- sounddevice and PyAudio can number the same
physical devices differently, so mixing the two would risk picking the
wrong one.

Usage:
    from mic_select import choose_input_device
    device_index = choose_input_device()  # None if cancelled -> system default
"""

import tkinter as tk
from tkinter import ttk

import pyaudio


def choose_input_device():
    """Show a dropdown of available input devices and block until the
    user picks one. Returns the PyAudio device index to pass as
    `input_device_index=` to PyAudio.open(), or None if the user chose
    the system default instead."""
    pa = pyaudio.PyAudio()
    try:
        input_devices = [
            (i, pa.get_device_info_by_index(i))
            for i in range(pa.get_device_count())
            if pa.get_device_info_by_index(i)["maxInputChannels"] > 0
        ]
    finally:
        pa.terminate()

    if not input_devices:
        raise RuntimeError("No input (microphone) devices found.")

    labels = [
        f"{i}: {d['name']}  ({int(d['defaultSampleRate'])} Hz, "
        f"{d['maxInputChannels']} ch)"
        for i, d in input_devices
    ]

    choice = {"index": None}

    root = tk.Tk()
    root.title("Choose a microphone")
    root.resizable(False, False)

    tk.Label(root, text="Microphone:").grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

    selected = tk.StringVar(value=labels[0])
    combo = ttk.Combobox(root, textvariable=selected, values=labels, state="readonly", width=65)
    combo.current(0)
    combo.grid(row=1, column=0, padx=10, pady=5)

    def on_confirm():
        choice["index"] = input_devices[combo.current()][0]
        root.destroy()

    def on_default():
        root.destroy()

    buttons = tk.Frame(root)
    buttons.grid(row=2, column=0, pady=(0, 10))
    tk.Button(buttons, text="Use this mic", command=on_confirm).pack(side="left", padx=5)
    tk.Button(buttons, text="Use system default", command=on_default).pack(side="left", padx=5)

    root.protocol("WM_DELETE_WINDOW", on_default)
    root.eval("tk::PlaceWindow . center")
    root.mainloop()

    return choice["index"]


if __name__ == "__main__":
    picked = choose_input_device()
    pa = pyaudio.PyAudio()
    try:
        if picked is None:
            print("Using system default input device.")
        else:
            print(f"Selected device {picked}: {pa.get_device_info_by_index(picked)['name']}")
    finally:
        pa.terminate()
