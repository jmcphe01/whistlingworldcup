"""Tank mapping and BLE rate limiting, against a fake robot."""

from __future__ import annotations

import pytest

from config import ThrottleConfig
from whistle.commands import Drive
from whistle.driver import RobotDriver, tank_for


class FakeRobot:
    """Records calls instead of talking to Bluetooth."""

    def __init__(self):
        self.tank_calls: list[tuple[int, int]] = []
        self.stops = 0

    def movement_move_tank(self, left, right):
        self.tank_calls.append((left, right))

    def motor_stop(self):
        self.stops += 1


@pytest.fixture
def robot():
    return FakeRobot()


@pytest.fixture
def driver(robot):
    return RobotDriver(robot, ThrottleConfig(), resend_interval=0.5)


def test_stop_is_both_tracks_still():
    assert tank_for(Drive.STOP, ThrottleConfig()) == (0, 0)


def test_forward_drives_both_tracks_together():
    config = ThrottleConfig()
    left, right = tank_for(Drive.FORWARD, config)
    assert left == right == config.speed_forward


def test_fast_is_faster_than_forward():
    config = ThrottleConfig()
    assert tank_for(Drive.FORWARD_FAST, config)[0] > tank_for(Drive.FORWARD, config)[0]


def test_backward_reverses_both_tracks():
    left, right = tank_for(Drive.BACKWARD, ThrottleConfig())
    assert left == right < 0


@pytest.mark.parametrize(
    "drive, sign", [(Drive.TURN_RIGHT, 1), (Drive.TURN_LEFT, -1)]
)
def test_turns_counter_rotate_the_tracks(drive, sign):
    left, right = tank_for(drive, ThrottleConfig())
    assert left == -right
    assert left * sign > 0, "right turn drives the left track forward"


def test_every_command_has_a_tank_mapping():
    for drive in Drive:
        assert len(tank_for(drive, ThrottleConfig())) == 2


# --- rate limiting ----------------------------------------------------------

def test_a_change_is_sent_immediately(driver, robot):
    assert driver.apply(0.0, Drive.FORWARD) is True
    assert robot.tank_calls == [(45, 45)]


def test_an_unchanged_command_is_not_resent_every_frame(driver, robot):
    """86 frames a second down a BLE link would add latency, not remove it."""
    driver.apply(0.0, Drive.FORWARD)
    for frame in range(1, 30):
        driver.apply(frame * 0.0116, Drive.FORWARD)   # ~0.34 s of frames
    assert len(robot.tank_calls) == 1


def test_a_change_mid_stream_goes_out_at_once(driver, robot):
    driver.apply(0.0, Drive.FORWARD)
    driver.apply(0.02, Drive.FORWARD)
    assert driver.apply(0.03, Drive.TURN_RIGHT) is True
    turn = ThrottleConfig().speed_turn
    assert robot.tank_calls[-1] == (turn, -turn)


def test_the_current_command_is_resent_as_a_keepalive(driver, robot):
    """Insurance against a dropped BLE packet leaving the car running."""
    driver.apply(0.0, Drive.FORWARD)
    assert driver.apply(0.4, Drive.FORWARD) is False
    assert driver.apply(0.5, Drive.FORWARD) is True
    assert robot.tank_calls == [(45, 45), (45, 45)]


def test_stop_always_goes_through(driver, robot):
    driver.apply(0.0, Drive.FORWARD)
    driver.stop()
    assert robot.tank_calls[-1] == (0, 0)
    assert robot.stops == 1


def test_stop_also_halts_the_motors_not_just_the_movement(driver, robot):
    """Zero speed and an explicit motor stop: belt and braces at match end."""
    driver.stop()
    assert robot.tank_calls[-1] == (0, 0) and robot.stops == 1


def test_last_tank_reports_what_the_robot_was_told(driver):
    driver.apply(0.0, Drive.BACKWARD)
    assert driver.last_tank == (-40, -40)
