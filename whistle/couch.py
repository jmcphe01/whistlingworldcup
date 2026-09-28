"""The couch: a single motor that a partner spins by MQTT message.

It is deliberately independent of the match. It answers in either role, in any
phase, before `start` and after the match is over, and starting, ending or
resetting a match never touches it. The commands arrive on the same topic as the
match messages, so the session recognises them first and they never reach the
match rules (which would otherwise log them as unknown messages).

The motor is only ever told to run, or to stop. It keeps spinning until it is told
otherwise, so `shutdown` matters: without it, quitting the program would leave the
couch turning.
"""

from __future__ import annotations

from config import HardwareConfig, MqttConfig
from whistle.match import normalise

LEFT = "left"
RIGHT = "right"
STOP = "stop"


def parse_command(payload: str, mqtt: MqttConfig) -> str | None:
    """LEFT, RIGHT or STOP if the payload is a couch command, otherwise None.

    Case and surrounding whitespace are ignored, the same as every other message.
    """
    text = normalise(payload)
    for action, message in ((LEFT, mqtt.couch_left_message),
                            (RIGHT, mqtt.couch_right_message),
                            (STOP, mqtt.couch_stop_message)):
        if text == normalise(message):
            return action
    return None


class CouchMotor:
    """Spins a single motor left or right, or stops it."""

    def __init__(self, motor, hardware: HardwareConfig | None = None, log=print):
        self.motor = motor
        self.hardware = hardware or HardwareConfig()
        self._log = log
        self._state = "stopped"

    @property
    def state(self) -> str:
        """"stopped", "spinning left" or "spinning right", for the monitor."""
        return self._state

    def execute(self, action: str) -> str:
        """Carry out LEFT, RIGHT or STOP. Returns a description for the log.

        A Bluetooth failure is reported and swallowed: a lost couch command must
        not take down the loop that is driving the robot. The state is left as it
        was, since the motor was not told anything.
        """
        try:
            if action == LEFT:
                self.motor.run(self.hardware.couch_left_speed)
                self._state = "spinning left"
            elif action == RIGHT:
                self.motor.run(self.hardware.couch_right_speed)
                self._state = "spinning right"
            elif action == STOP:
                self.motor.stop()
                self._state = "stopped"
            else:
                raise ValueError(f"unknown couch action {action!r}")
        except ValueError:
            raise
        except Exception as error:
            self._log(f"  [couch] {action} failed: {error}")
            return f"{action} failed ({error})"
        return self._state

    def shutdown(self) -> None:
        """Stop the motor. Called on exit so the couch is not left spinning."""
        self.execute(STOP)
