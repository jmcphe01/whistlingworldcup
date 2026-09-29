"""Robot-side listener for WhistleSoccer.

Run this on the laptop that's actually Bluetooth-connected to the robot's
Double Motor. It subscribes to the same MQTT topic whistle_controller.py
publishes to (run by your teammate, whistling into their own laptop's
mic) and drives the robot as each command arrives:

    STOP        -> stop
    FORWARD     -> drive forward ("speed up")
    pivot[N]    -> turn in place by N degrees:
                   positive N = turn left, negative N = turn right
                   (whistle_controller.py only ever sends N in
                   {90, 45, -45, -90}, but any integer works here)

Unlike STOP/FORWARD, which are held until a different command arrives,
each pivot[N] is a one-shot, self-terminating turn (movement_turn_for_
degrees() stops automatically once it's turned N degrees) -- it fires
once per whistle that lands in that pitch zone, not continuously for as
long as the whistle is held.

If no message has arrived yet (e.g. right at startup, before your
teammate's script has published anything), the robot just stays stopped
-- that's the same "no whistle -> stop" default whistle_controller.py
uses, carried through to this side too.
"""

import re
import time

import legoeducation as le

from mqttlib import MQTTClient

MQTT_TOPIC = "ME193/Rogers"  # must match whistle_controller.py

# Set to your Double Motor's connection card if you have more than one
# advertising nearby; None connects to the first one found.
CARD_COLOR = None
CARD_SERIAL = None

DRIVE_SPEED = 60  # percent, for FORWARD
TURN_SPEED = 50   # percent, for pivot[N] turns

PIVOT_RE = re.compile(r"^pivot\[(-?\d+)\]$")

current_command = "STOP"


def apply_command(command):
    global current_command
    if command == current_command:
        return  # already doing this -- don't re-issue the same BLE command every message

    print(f"Command: {command}")
    if command == "STOP":
        motor.movement_stop()
    elif command == "FORWARD":
        motor.movement_move_tank(DRIVE_SPEED, DRIVE_SPEED, blocking=False)
    else:
        match = PIVOT_RE.match(command)
        if not match:
            print(f"Unknown command {command!r}, ignoring.")
            return
        angle = int(match.group(1))
        direction = le.MOVEMENT_TURN_DIRECTION_LEFT if angle >= 0 else le.MOVEMENT_TURN_DIRECTION_RIGHT
        motor.movement_turn_for_degrees(abs(angle), direction=direction, blocking=False)

    current_command = command


def on_message(topic, payload):
    apply_command(payload.strip())


motor = le.DoubleMotor()
print("Scanning for Double Motor...")
motor.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
if not motor.connected:
    raise ConnectionError("Could not connect. Make sure the Double Motor is on and in range.")
print("Connected!")

# movement_turn_for_degrees() (used by pivot[N]) doesn't take its own speed
# argument -- it uses whatever movement speed was last set. FORWARD/STOP
# pass their own explicit speed via movement_move_tank(), so this only
# affects pivot turns.
motor.movement_set_speed(TURN_SPEED)

print(f"Connecting to MQTT broker, listening on '{MQTT_TOPIC}'...")
with MQTTClient() as client:
    client.subscribe(MQTT_TOPIC, on_message)
    print("Listening for whistle commands -- press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Stopping...")
        motor.movement_stop()
        motor.disconnect()
        print("Disconnected.")
