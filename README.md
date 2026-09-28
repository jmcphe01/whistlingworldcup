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
| warble left-right-left: fall, rise, fall in one unbroken whistle | claim the goal |

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

The double motor's forward and backward are flipped (`invert_drive` in
`ThrottleConfig`, on by default), because the car is mounted so that forward needs
negative track speeds. Only the linear motion is flipped. Pivots are left alone,
since inverting them would swap left and right turns. Set it to `False` to undo.

Driving is hold-to-go: the car moves only while you are whistling a steady note,
and stops the moment you stop. Steering is live: you turn for exactly as long as
you slide, so a longer slide turns further. Zone edges are set as note names in
[`config.py`](config.py) and compared in cents.

The goal command is a warble, a series of ups and downs. "Left" is a falling
slide and "right" a rising one, matching how each steers, so left-right-left is
one unbroken whistle that falls, rises, then falls again. Each leg has to travel
about 3.5 semitones, so vibrato is invisible to it, and it counts a leg as soon as
it has travelled far enough rather than when it ends, since the last leg has no
reversal after it to wait for.

It uses the same slides as steering, so the car will pivot left, right, left as
you whistle it; that is harmless, because scoring stops the car. What keeps
ordinary steering from scoring by accident is that the warble must be one
unbroken whistle (a pause longer than 0.25 s abandons it), every leg must be
quick, and the whole thing must fit in 2.5 s. If it ever fires while you steer,
lengthen `goal_pattern` to five legs in `config.py`.

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
python main.py --monitor
```

Choose the **role** (ball or goalie) and the **topic** in the control strip at the
bottom of the monitor. The topic is applied when you press Return. Changing either
starts a fresh match, and **new match** does the same without changing anything.
A finished match stays finished until then: a second `start` arriving after the
whistle must not quietly begin another match, and the robot stays connected
between matches so there is no Bluetooth reconnect.

Useful flags: `--role goalie` to start as the goalie, `--no-robot` and `--no-mqtt`
for a desk test with nothing connected (a loopback stands in for the broker and
the match starts at once), `--device "Scarlett"` to pick an input by name,
`--broker` and `--topic` to change them trackside, `--list-devices`.

Both roles subscribe to the one topic and wait for `start`. This program sends
exactly two messages, **goal** (the ball scored) and **tagged** (the goalie reached
the ball), and both are text boxes in the monitor, applied with Return like the
topic, so the wording can be agreed with your opponent on the day. Changing one
does not end a running match. Blank messages, and messages that collide with each
other or with `start`, are refused: the two teams could not tell them apart.
`start` is only ever received. Its wording is `start_message` in `MqttConfig`.

### Playing the other team

While it runs, type commands in the terminal. This is how you test against the
real robot without a real opponent.

| command | effect |
|---|---|
| `start` | publish the start message (over MQTT, through the broker and back) |
| `tagged` | publish the tagged message. A goalie sings on this |
| `scored` | publish the goal message. A goalie mourns on this |
| `send <text>` | publish anything to the topic |
| `goal` | pretend the ball whistled the goal command (local) |
| `tag` | pretend the goalie reached the light sensor (local) |
| `role ball\|goalie`, `topic <name>`, `msg goal\|tagged <text>`, `reset` | same as the controls in the window |
| `song win\|lose` | play a song, to check the speakers |
| `status` | role, phase, topic, light sensor reading against its trigger |
| `help`, `quit` | |

A worked test of each ending, as the ball: `start`, then either `goal` (publishes
"goal" and plays the winning song) or `tag` (publishes "tagged" and
plays the death song); `reset` between them. As the goalie: `role goalie`,
`start`, then `tagged` (victory song) or `scored` (death song).

What each ending does, for both roles:

| event | ball | goalie |
|---|---|---|
| ball scores | publishes scored, winning song | hears scored, death song |
| ball tagged | publishes tagged, death song, stops | hears tagged, winning song |

### The single motor

A single motor that a partner turns by MQTT message: **pivot[angle]**, for example
`pivot[90]` or `pivot[-45]`. The motor turns by that many degrees, and a negative
angle means the other direction. It connects with the same card as everything else,
and the message arrives on the same topic as the match messages.

It is independent of the match. It answers in either role, in any phase, before
`start` and after the match is over, and starting, ending or resetting a match never
touches it. The session recognises the message first, so it never reaches the match
rules.

Only the bracketed form is claimed, with case and spaces ignored (`Pivot [ -45 ]`
works). A bare `pivot` is not a single motor message and goes to the match rules. Brackets
with no readable number, like `pivot[abc]`, are reported rather than ignored. Angles
are rounded to a whole degree, `pivot[0]` does nothing, and an angle past
`single_motor_max_degrees` (3600, ten turns) is refused so a mistyped `pivot[9000000]` cannot
spin the motor for minutes.

The turn is started without waiting for it to finish. Waiting would stall the loop
that listens for whistles, and the car would keep driving on its last command
meanwhile. That also means a new pivot arriving mid-turn is sent straight away, and
what the hub does with two overlapping turns is untested. The motor is stopped when
the program exits, in case it is mid-turn.

In `HardwareConfig`, `single_motor_speed` sets the speed (percent) and `single_motor_invert` flips
which way a positive angle turns, if it comes out the wrong way for how the motor is
mounted. The word is `pivot_message` in `MqttConfig`, and the game messages cannot be
reworded to collide with it. If the motor fails to connect the program carries on with
a warning, and `--no-single-motor` skips it. To test it alone, type `pivot 90` or `pivot -45`
in the console: it publishes the exact `pivot[90]` form your partner would send, so
the whole path is exercised.

### The songs

They play on the robot's own beeper, so they are audible wherever the robot is
(`--speaker` plays them on the laptop instead, and `--no-robot` does so
automatically). The winning song is the hook of the Rick Astley chorus, transcribed
by ear in C major (only its first two lines, with an approximate rhythm). The losing
song is four descending half steps with the last held: womp womp womp womp.

**Loudness.** The beeper has no volume control (`beep()` takes only pattern,
frequency and count), so pitch is the one lever. A small speaker is weak at low
pitches, so the songs play an octave up by default (`beep_octave_shift` in
`HardwareConfig`; 0 plays them as written). If the top note would pass the 2700 Hz
limit the whole song is lowered rather than clamped, which would flatten the
melody. `--both` also plays them on the laptop speaker at the same time. The
losing song holds every note (0.6 s each, the last 1.8 s).

`beep()` takes a frequency (0-2700 Hz) but no duration, so a note's length comes from
timing: start the beep without waiting, sleep, then `stop_beep()`. A held note is a
run of beeps restarted every `beep_sustain_seconds` (in `HardwareConfig`, 0.3 by
default). If held notes stutter, raise it; if you hear clicks mid-note, lower it.
`song win` and `song lose` in the console play them on demand for tuning.

Publishes go out at QoS 1 and the client resubscribes after a reconnect, because
venue Wi-Fi drops connections and a message sent while the link is down would
otherwise be lost, or the car would stay connected but deaf.

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
  sensor.py        light-sensor proximity watch, and the armable monitor thread
  match.py         match rules as a state machine, plus the runner
  session.py       role, topic, rematch, messages and sensor trips, on one thread
  comms.py         MQTT adapter around mqttlib, plus a loopback for --no-mqtt
  console.py       the typed test commands
  songs.py         the death song and the song of success
```

Threading, in `main.py`: the main thread runs audio → pitch → motors and nothing
slow is allowed on it. The monitor is a **separate process** fed a drop-on-full
queue, so a matplotlib redraw can never sit between a whistle and the wheels. The
light sensor is polled on its own thread because BLE reads are far too slow to do
inline at 86 frames a second.

## Tests

```bash
pytest                       # about 430 tests, a few seconds
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

Verified by the unit suite: the whole control scheme, the warble (including
through the real audio chain), the match rules for both roles, role/topic/rematch
handling, the typed commands, the MQTT adapter against a fake client, and the
light-sensor monitor against a scripted sensor. The monitor's controls were
checked with synthetic mouse and key events.

Not yet verified on hardware: the robot, the colour sensor and the broker. Two
things to check on the day with the `status` command:

- **The sensor's reflection scale.** `trigger_delta` in `SensorConfig` is 12,
  which assumes reflection reads 0-100. The course wrapper normalises it by 255
  elsewhere, which suggests 0-255. Watch the reading in `status` while you move a
  hand toward the sensor and scale `trigger_delta` to match.
- **False tags from the goal itself.** A forward-facing reflection sensor also
  rises when the car nears the goal wall. Drive up to the goal and check the
  reading stays under the trigger.
