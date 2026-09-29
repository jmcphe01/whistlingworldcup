# WhistleSoccer

Two scripts, running on two different laptops, connected by MQTT:

- **`whistle_controller.py`** -- run on the whistler's laptop. Grabs the
  mic via PyAudio, classifies the whistle's pitch, shows the live signal
  and decision on screen, and publishes the decision to the MQTT topic
  `ME193/WhistleSoccer/command`.
- **`robot_listener.py`** -- run on the laptop that's Bluetooth-connected
  to the actual robot. Subscribes to that same topic and drives the
  Double Motor as each command arrives.

Splitting it this way means the person whistling doesn't need to be
anywhere near the robot or its BLE range -- only their laptop's mic and an
internet connection (for the MQTT broker) matter.

## Setup

```bash
pip install -r requirements.txt
```

`pyaudio` needs the PortAudio C library to build; if `pip install pyaudio`
fails with a missing `portaudio.h`, install PortAudio first (e.g. via
`vcpkg install portaudio:x64-windows` on Windows, or your OS package
manager elsewhere) before retrying.

`mqttlib.py` points at the public `test.mosquitto.org` broker (the same
one used for the class-wide `ME193/Rogers` game-start topic) -- no
credentials needed. It connects over TLS on port 8883, which uses
`test.mosquitto.org`'s own self-signed CA rather than a publicly-trusted
one; that CA cert is checked into this folder as `mosquitto_org_ca.crt`
(downloaded from `https://test.mosquitto.org/ssl/mosquitto.org.crt`) so
`mqttlib.py` can verify the server without extra setup.

Then, on the whistler's laptop:

```bash
python whistle_controller.py
```

And on the robot's laptop:

```bash
python robot_listener.py
```

## Describe the policy -- how does it make decisions?

`whistle_controller.py` reads the mic in small chunks (1024 samples,
~23ms at 44.1kHz) and feeds each one to `PitchWhistleDetector`
(`whistle_detector.py`). For each chunk it:

1. Bandpass-filters the chunk to the whistle range (1200-4500 Hz by
   default).
2. Checks whether that band's energy dominates the chunk's total energy
   (the "dominance ratio") for at least 0.25 continuous seconds -- this
   is *whether* a whistle is happening at all (see the noise-masking
   question below).
3. If so, estimates the whistle's actual pitch (the frequency where the
   filtered chunk's spectrum peaks) and classifies it into one of six
   zones, low pitch to high pitch:

   | Zone | Pitch range (of 1200-4500 Hz) | Robot action |
   |---|---|---|
   | STOP | 1st sixth | stop |
   | pivot[90] | 2nd sixth | turn left 90 degrees |
   | pivot[45] | 3rd sixth | turn left 45 degrees |
   | pivot[-45] | 4th sixth | turn right 45 degrees |
   | pivot[-90] | 5th sixth | turn right 90 degrees |
   | FORWARD | 6th sixth | drive forward ("speed up") |

   A pivot command's bracketed number is the turn angle in degrees:
   positive = left, negative = right, so there are two step sizes (90 and
   45 degrees) in each direction instead of one open-ended "keep turning"
   command per side. Commands are spelled out in full (`pivot[90]`, not a
   bare `LEFT`) because the MQTT broker/topic namespace is shared with
   other teams -- a one-word "LEFT" is too likely to collide with someone
   else's message on the same channel.

Whenever the classified zone changes, `whistle_controller.py` publishes
the new zone name as a plain-text MQTT message on
`ME193/WhistleSoccer/command`. `robot_listener.py` applies whatever it
receives to the Double Motor: `STOP`/`FORWARD` are held until a different
command arrives (so they don't re-send the same BLE command on every
message), while each `pivot[N]` is a one-shot, self-terminating turn
(`movement_turn_for_degrees()` stops automatically after N degrees) fired
once per whistle that lands in that zone.

## What does the code do if no whistle is detected?

It's treated the same as a low-pitch whistle: **STOP**. Concretely,
`PitchWhistleDetector.process()` only estimates a pitch once the
dominance ratio has held above threshold for the full 0.25s duration gate;
until then (including the entire time there's no whistle at all) it
reports the `"STOP"` zone and `pitch_hz = None`. This was a deliberate
choice over "hold the last command": if the whistler stops whistling
(on purpose, or because they got distracted, cut off, etc.), the robot
should fail safe and stop rather than keep charging toward the goal on a
stale command.

## How did you try to mask out unwanted noise?

Three layers, in order (implemented in `whistle_detector.py`,
`PitchWhistleDetector`/`WhistleDetector`):

1. **Bandpass filter.** A 4th-order Butterworth bandpass (`scipy.signal
   .butter`, `sosfilt` with persistent filter state across chunks) only
   lets the whistle's frequency range through before anything else is
   computed. Low-frequency noise (voices, footsteps, motors) and
   high-frequency hiss outside that range are attenuated before they can
   affect the pitch estimate at all.
2. **Dominance ratio.** `RMS(filtered chunk) / RMS(raw chunk)` measures
   what fraction of the chunk's total loudness lives inside that
   passband. A clean whistle scores near 1.0; a loud but broadband sound
   (a shout, a hammer strike, a dropped part) still spreads most of its
   energy *outside* the passband and scores low even though it's loud --
   so raw loudness alone was rejected as a trigger, on purpose.
3. **Duration gate.** The dominance ratio has to stay above threshold
   *continuously* for 0.25 seconds before it counts. This filters out
   brief transients that might momentarily spike the ratio (a single
   sharp broadband click can briefly look tonal by chance) -- a held
   whistle is the only thing that satisfies a quarter-second of
   sustained dominance.

`whistle_controller.py`'s live plot shows all of this directly: a
spectrogram with the passband shaded, a pitch trace with the four zone
boundaries marked, and the title showing the live dominance ratio and how
long it's been held, so you can watch workshop/gym noise fail to trigger
a decision in real time while an actual whistle does.

## Still open (not yet implemented)

The World Cup game-day protocol -- waiting for an MQTT "start" message on
`ME193/Rogers`, detecting proximity to the goalie's light sensor, playing
success/failure songs, and publishing the goal-scored message -- isn't
built yet. `whistle_controller.py`/`robot_listener.py` currently only
cover continuous whistle-driven control.
