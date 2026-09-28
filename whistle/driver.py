"""Drive commands -> LEGO tank-drive calls.

Two reasons this is its own module. First, it is the only place that talks to the
motors, so every test above it can use a fake. Second, it is where BLE latency is
managed: a Bluetooth link is slow, and writing to it every analysis frame (86
times a second) would add latency rather than remove it.

So commands go out on *change*, immediately, plus a slow keepalive resend. The
layers above -- hysteresis in the throttle zones, persistence in the tracker, the
latch on turns -- are what make changes rare enough for that to work.

Steering uses `movement_move_tank(left, right)` rather than
`turn_left`/`turn_right`, because those run for a fixed number of degrees and
block until they finish, which fights responsiveness.
"""

from __future__ import annotations

from typing import Protocol

from config import ThrottleConfig
from whistle.commands import Drive


class TankRobot(Protocol):
    """The slice of `lelib.doubleMotor` this project actually uses."""

    def movement_move_tank(self, left: int, right: int) -> None: ...
    def motor_stop(self) -> None: ...


def tank_for(drive: Drive, config: ThrottleConfig) -> tuple[int, int]:
    """Left and right track speeds as percentages, for one command.

    `invert_drive` reverses forward and backward only. Pivots counter-rotate the
    tracks, so inverting them too would swap left and right turns, which is not
    what a car whose front is the wrong end needs.
    """
    forward, fast = config.speed_forward, config.speed_fast
    back, turn = config.speed_backward, config.speed_turn
    sign = -1 if config.invert_drive else 1
    return {
        Drive.STOP: (0, 0),
        Drive.FORWARD: (sign * forward, sign * forward),
        Drive.FORWARD_FAST: (sign * fast, sign * fast),
        Drive.BACKWARD: (-sign * back, -sign * back),
        Drive.TURN_RIGHT: (turn, -turn),
        Drive.TURN_LEFT: (-turn, turn),
    }[drive]


class RobotDriver:
    def __init__(self, robot: TankRobot, config: ThrottleConfig | None = None,
                 resend_interval: float = 0.5):
        self.robot = robot
        self.config = config or ThrottleConfig()
        # A dropped BLE packet would otherwise leave the car running; repeating
        # the current command occasionally is cheap insurance.
        self.resend_interval = resend_interval
        self._last_tank: tuple[int, int] | None = None
        self._last_sent = float("-inf")
        self.commands_sent = 0

    @property
    def last_tank(self) -> tuple[int, int] | None:
        return self._last_tank

    def apply(self, now: float, drive: Drive) -> bool:
        """Send `drive` to the robot if needed. True if a command went out."""
        tank = tank_for(drive, self.config)
        changed = tank != self._last_tank
        stale = now - self._last_sent >= self.resend_interval
        if not (changed or stale):
            return False

        self.robot.movement_move_tank(tank[0], tank[1])
        self._last_tank = tank
        self._last_sent = now
        self.commands_sent += 1
        return True

    def stop(self) -> None:
        """Halt now, unconditionally. Used on shutdown and when the match ends."""
        self.robot.movement_move_tank(0, 0)
        self.robot.motor_stop()
        self._last_tank = (0, 0)
        self.commands_sent += 1
