"""The vocabulary shared by the interpreter and the driver."""

from __future__ import annotations

from enum import Enum


class Drive(Enum):
    """What the car should be doing right now.

    Throttle comes from a sustained pitch and stops the instant you stop
    whistling. The two turns come from pitch sweeps and are latched for a short
    while, because the whistle that requests a turn is over by the time the
    sweep has been recognised.
    """

    STOP = "stop"
    FORWARD = "forward"
    FORWARD_FAST = "forward fast"
    BACKWARD = "backward"
    TURN_LEFT = "turn left"
    TURN_RIGHT = "turn right"

    @property
    def is_turn(self) -> bool:
        return self in (Drive.TURN_LEFT, Drive.TURN_RIGHT)
