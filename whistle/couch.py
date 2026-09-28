"""The couch: a single motor that a partner turns by MQTT message.

The partner sends `pivot[<angle>]`, for example `pivot[90]` or `pivot[-45]`, and the
motor turns by that many degrees. Positive is one way and negative the other.

It is deliberately independent of the match. It answers in either role, in any
phase, before `start` and after the match is over, and starting, ending or resetting
a match never touches it. The commands arrive on the same topic as the match
messages, so the session recognises them first and they never reach the match rules
(which would otherwise log them as unknown messages).

Only the bracketed form is claimed. A bare `pivot`, or anything else, is not a couch
message and goes to the match rules like any other payload; a bracketed message with
no readable number is reported rather than ignored, since it was plainly meant for
the couch.
"""

from __future__ import annotations

import math
import re

from legoeducation import (
    MOTOR_MOVE_DIRECTION_CLOCKWISE,
    MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE,
)

from config import HardwareConfig, MqttConfig
from whistle.match import normalise


def parse_command(payload: str, mqtt: MqttConfig) -> float | None:
    """The angle in `pivot[<angle>]`, or None if this is not a pivot message.

    The word is matched like every other message (case and surrounding whitespace
    ignored), and space is allowed before the bracket. Raises ValueError, with a
    message fit to show, when the brackets are there but hold no usable number.
    """
    word = re.escape(normalise(mqtt.pivot_message))
    found = re.fullmatch(rf"{word}\s*\[(.*)\]", normalise(payload))
    if found is None:
        return None

    text = found.group(1).strip()
    try:
        angle = float(text)
    except ValueError:
        angle = math.nan
    if not math.isfinite(angle):
        raise ValueError(f"could not read an angle from {payload.strip()!r}; "
                         f"expected something like {mqtt.pivot_message}[90]")
    return angle


class CouchMotor:
    """Turns a single motor by an angle."""

    def __init__(self, motor, hardware: HardwareConfig | None = None, log=print):
        self.motor = motor
        self.hardware = hardware or HardwareConfig()
        self._log = log
        self._state = "idle"

    @property
    def state(self) -> str:
        """What it last did, for the monitor. The motor finishes a turn on its own,
        so this is the last command rather than a live position."""
        return self._state

    def pivot(self, angle: float) -> str:
        """Turn by `angle` degrees. Returns a description for the log.

        Positive is clockwise and negative counter-clockwise, unless `couch_invert`
        is set. The angle is rounded to a whole degree. Zero does nothing. An angle
        beyond `couch_max_degrees` is refused, so a typo like pivot[9000000] cannot
        spin the motor for minutes.

        The turn is started without waiting for it to finish. Waiting would stall
        the loop that is listening for whistles, and the car would keep driving on
        its last command while it did.

        A Bluetooth failure is reported and swallowed: a lost couch command must not
        take down the loop that is driving the robot.
        """
        degrees = int(round(angle))
        if degrees == 0:
            return "no turn (0 degrees)"
        limit = self.hardware.couch_max_degrees
        if abs(degrees) > limit:
            raise ValueError(f"{degrees} degrees is past the {limit} degree limit")

        clockwise = (degrees > 0) != self.hardware.couch_invert
        direction = (MOTOR_MOVE_DIRECTION_CLOCKWISE if clockwise
                     else MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
        try:
            self.motor.motor_run_for_degrees(
                abs(degrees), direction=direction, speed=self.hardware.couch_speed,
                blocking=False)
        except Exception as error:
            self._log(f"  [couch] pivot failed: {error}")
            return f"pivot {degrees:+d} degrees failed ({error})"

        self._state = f"pivoted {degrees:+d} deg"
        return f"pivoting {degrees:+d} degrees"

    def shutdown(self) -> None:
        """Stop the motor, in case it is mid-turn when the program exits."""
        try:
            self.motor.stop()
        except Exception as error:
            self._log(f"  [couch] stop failed: {error}")
        self._state = "idle"
