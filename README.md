# Whistling World Cup

ME193 Assignment 3. A whistle-controlled LEGO Education robot: a PyAudio input
stream is pitch-tracked in real time and the detected note drives the car, with
MQTT for match signalling and songs for win/loss. See [`README`](README) for the
assignment text.

## Setup

Requires Python 3.11 (3.14 does not have wheels for parts of this stack) and
Homebrew `portaudio`, which PyAudio links against.

```bash
brew install portaudio
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then confirm the environment:

```bash
source .venv/bin/activate && pytest
```

### Microphone permission

macOS gates microphone access per-application. The first capture from a new
terminal or editor raises a system prompt; if it is missed, PyAudio returns
silence rather than an error. Grant access under **System Settings → Privacy &
Security → Microphone** for whichever app runs Python.

## Hardware

`lelib.py` is the course wrapper around `legoeducation` (copied in unmodified —
re-copy it if the class version changes). Devices pair by the color and serial
printed on their Connection Card; the double motor and the color sensor are
separate Bluetooth devices with separate cards.

## Tests

```bash
pytest
```

`tests/test_environment.py` checks only that the environment is intact — it
touches no hardware. Audio, control-mapping and MQTT logic are tested against
synthesized buffers and a fake broker, so the full suite runs with no robot,
no microphone and no network.
