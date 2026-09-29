"""Whistle-controlled Double Motor: each whistle you blow toggles the robot
between driving forward and driving backward.

Detection is handled by WhistleDetector (whistle_detector.py): a strict
bandpass filter isolates your whistle's pitch, a dominance ratio checks
that pitch actually dominates the sound (rejecting broadband noise like
hammer strikes or talking), and a duration gate requires that to hold for
a bit before it counts as a real whistle instead of a brief spike.

Tune CENTER_FREQ/BANDWIDTH/DOMINANCE_THRESHOLD/MIN_DURATION below to match
your whistle and room -- use whistle_visualizer.py to see the effect live.
"""

import time

import legoeducation as le
import sounddevice as sd

from mic_select import choose_input_device
from whistle_detector import WhistleDetector

# ── Motor setup (same connection card as the other ME193 examples) ────────
CARD_COLOR = le.LEGO_COLOR_RED
CARD_SERIAL = "1129"
DRIVE_SPEED = 50  # percent

# ── Whistle detection tuning -- keep in sync with whistle_visualizer.py ────
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
CENTER_FREQ = 3000.0        # Hz -- your whistle's pitch
BANDWIDTH = 200.0           # Hz -- total pass width (+/- 100 Hz here)
DOMINANCE_THRESHOLD = 0.6   # filtered/raw RMS ratio required to count as "whistling"
MIN_DURATION = 0.4          # seconds the ratio must hold continuously to trigger

detector = WhistleDetector(
    sample_rate=SAMPLE_RATE,
    center_freq=CENTER_FREQ,
    bandwidth=BANDWIDTH,
    dominance_threshold=DOMINANCE_THRESHOLD,
    min_duration=MIN_DURATION,
)

direction = 1  # 1 = forward, -1 = backward


def drive(speed):
    motor.movement_move_tank(speed, speed, blocking=False)


def on_audio(indata, frames, time_info, status):
    global direction

    if detector.process(indata[:, 0]):
        direction *= -1
        print("Whistle! Now driving", "forward" if direction == 1 else "backward")
        drive(DRIVE_SPEED * direction)
        # Require the tone to build back up from zero before it can fire
        # again, instead of re-triggering every block while you keep blowing.
        detector.reset()


DEVICE = choose_input_device()
if DEVICE is None:
    print("Using system default input device.")
else:
    print(f"Using input device: {sd.query_devices(DEVICE)['name']}")

motor = le.DoubleMotor()
print(f"Scanning for Double Motor (red card, serial {CARD_SERIAL})...")
motor.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)

if not motor.connected:
    raise ConnectionError("Could not connect. Make sure the Double Motor is on and in range.")

print("Connected! Listening for whistles -- press Ctrl+C to stop.")
try:
    drive(DRIVE_SPEED * direction)
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        callback=on_audio,
        device=DEVICE,
    ):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nInterrupted.")
finally:
    print("Stopping...")
    motor.movement_stop()
    motor.disconnect()
    print("Disconnected.")
