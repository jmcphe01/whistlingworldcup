# Whistling World Cup

ME193 Assignment 3. A whistle-controlled LEGO Education car: a PyAudio stream is
pitch-tracked in real time, the pitch drives the car, and MQTT plus two songs
settle the match. See [`README`](README) for the assignment text.

## The control scheme

| Whistle | Car |
|---|---|
| hold a note steady, C4-A5 | reverse |
| hold a note steady, A5-A6 | forward |
| hold a note steady above A6 (D7 is the target) | forward, fast |
| stop whistling | stop |
| slide the pitch up | pivot right, while you keep sliding |
| slide the pitch down | pivot left |
| three short high chirps (above A6) | claim the goal |

Throttle and steering are told apart by **how the pitch is moving, not where it
sits**. A steady note is a throttle command; a sliding one is a steering command;
they are mutually exclusive. That is why a slide never drives the car forward on
its way up through the forward zone, and why holding a note never steers.

Motion is measured as a least-squares slope over the last 0.3 s rather than the
difference between the first and last samples. Endpoint difference is at the
mercy of where the window lands: on a note with vibrato whose ends fall on
opposite swings it reports a slide that is not there.

Between "clearly steady" and "clearly sliding" there is a deliberate dead band
where the car stops. An ambiguous whistle doing nothing beats it guessing.

Driving is hold-to-go: the car moves only while you are whistling a steady note,
and stops the moment you stop. Steering is live: you turn for exactly as long as
you slide, so a longer slide turns further. Zone edges are set as note names in
[`config.py`](config.py) and compared in cents.

The goal command is three chirps rather than one extreme pitch on purpose: a
false positive there ends the match, and a stray noise can produce one pitch but
not three deliberately spaced high chirps.

## Setup

Requires Python 3.11 (3.14 has no wheels for parts of this stack) and Homebrew
`portaudio`, which PyAudio links against.

```bash
brew install portaudio
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

### Microphone permission

macOS gates microphone access per application. The first capture from a new
terminal or editor raises a system prompt; if it is missed, PyAudio returns
silence rather than an error. Grant access under **System Settings → Privacy &
Security → Microphone** for whichever app runs Python.

## Before a match: calibrate

```bash
python calibrate.py --list-devices
python calibrate.py
```

This measures two things in the room you will actually compete in and writes them
to `config.local.json` (gitignored):

- **the noise floor**, which the level gate then works from. An absolute
  threshold tuned in a quiet lab either fires on everything or on nothing in a
  loud room.
- **your whistle range**, which sets the search band. The band floor is the
  single most valuable noise-rejection setting there is: the lower it reaches,
  the more speech energy it admits.

## Running a match

```bash
python main.py --role ball   --monitor
python main.py --role goalie --monitor
```

Useful flags: `--no-robot` and `--no-mqtt` for a desk test with nothing
connected, `--device "Scarlett"` to pick an input by name, `--broker` / `--topic`
to change brokers trackside, `--list-devices`.

Both roles subscribe to the one topic and wait for `start`. The message wording
lives in `MqttConfig` so agreeing it with your opponent is a one-line edit.

## How noise rejection works

A loud room is the main threat to this project, so the gates are layered, and the
live monitor reports which one rejected a frame rather than leaving you guessing.

1. **Search band.** Peaks are only looked for between `min_hz` and `max_hz`.
2. **Peak-to-median.** A whistle is a narrow spike; clapping and scraping are
   broadband.
3. **Peak-to-second-peak.** This is the one that does the real work. A whistle is
   a *lone* spike, while voiced speech is a harmonic comb whose peaks come in
   families of comparable height. Measured on synthetic signals: whistles score
   52–70 dB, a whistle buried in babble still 10–21 dB, babble alone under 4 dB.
4. **Sub-harmonic check.** If there is energy at f/2 or f/3, the peak is a
   harmonic of something lower — a voice, not a whistle.

Layer 2 alone is *not* sufficient, which is worth knowing if you retune it: a
220 Hz voice puts a narrow, tonal peak at 440 Hz, right inside the whistle band,
and it passes a spikiness test comfortably. `test_pitch.py` has that case as a
regression test.

## Why FFT peak-picking, not YIN

A whistle is very nearly a pure sine with almost no harmonic content. The usual
pitch trackers exist to work out which harmonic is the fundamental, a problem a
whistle does not have, and they spend latency solving it. A windowed FFT peak
refined by parabolic interpolation is both more accurate here and far cheaper.

Bin spacing at 44.1 kHz over 2048 samples is 21.5 Hz, which is too coarse on its
own — A5 to A#5 is only 52 Hz. Interpolating a parabola through the peak bin and
its two neighbours in the log-magnitude domain recovers the true peak to about
1 Hz, roughly 2 cents. Windows are 2048 samples long for accuracy but produced
every 512 samples for latency: 46 ms of evidence, refreshed 86 times a second.

## Layout

```
config.py          every tunable, plus config.local.json overrides
lelib.py           course LEGO wrapper (copied unmodified)
mqttlib.py         course MQTT wrapper (copied unmodified)
main.py            the match program
calibrate.py       measure the room and your whistle
monitor.py         the live view (runs in its own process)
whistle/
  notes.py         note names, frequencies, cents
  pitch.py         detection and the noise gates
  stream.py        PyAudio -> overlapping analysis windows
  commands.py      the Drive vocabulary
  throttle.py      pitch zones with hysteresis
  gestures.py      sweep and chirp recognisers
  interpreter.py   the whole control scheme, as a pure function
  driver.py        Drive -> tank commands, with BLE rate limiting
  sensor.py        light-sensor proximity watch
  match.py         match rules as a state machine, plus the runner
  songs.py         the death song and the song of success
```

Threading, in `main.py`: the main thread runs audio → pitch → motors and nothing
slow is allowed on it. The monitor is a **separate process** fed a drop-on-full
queue, so a matplotlib redraw can never sit between a whistle and the wheels. The
light sensor is polled on its own thread because BLE reads are far too slow to do
inline at 86 frames a second.

## Tests

```bash
pytest                       # 268 tests, about 1.5 s
pytest --cov=whistle --cov=config --cov-report=term-missing
```

Nothing in the suite touches a microphone, a robot or a broker. Audio is
synthesized, hardware is faked, and time is passed in as an argument rather than
read from a clock — so gestures that take a second to perform are tested
instantly, and the whole suite runs anywhere.

That is why the DSP and the state machines are pure functions with injected
dependencies. It is the main thing shaping the architecture.

- `test_pitch.py` — accuracy on off-bin frequencies (interpolation is only
  tested by frequencies that fall between bins), and every noise gate
- `test_throttle.py` — zones and the hysteresis that stops boundary chatter
- `test_gestures.py` — sweeps and chirps, including every non-interference case:
  a held note must not steer, and nothing you do to drive may claim the goal
- `test_interpreter.py` — the control scheme over synthetic pitch streams
- `test_end_to_end.py` — **audio samples in, motor commands out**, through the
  real chain, including babble-only and whistle-over-babble
- `test_match.py` — the rules, including the echo trap: a public broker hands
  your own publish straight back to you
- `test_driver.py` — tank mapping and BLE rate limiting against a fake robot
- `test_sensor.py`, `test_songs.py`, `test_stream.py`, `test_config.py`,
  `test_monitor.py`, `test_notes.py`

## Status

Verified: the full suite passes, and the live path has been validated at real
time through a loopback audio device — held A5, D7 and C4 produced forward,
forward-fast and reverse, rising and falling glides produced right and left
turns, pitch tracked within 2 cents, no dropped audio blocks.

Not yet verified on hardware: the robot, the colour sensor and the broker. Those
need the cars and an opponent.
