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


# The car is mounted so that "forward" needs negative track speeds, so the shipped
# default inverts. Tests of the mapping and the rate limiting use the raw one, and
# the inversion has its own tests below.
RAW = ThrottleConfig(invert_drive=False)


@pytest.fixture
def robot():
    return FakeRobot()


@pytest.fixture
def driver(robot):
    return RobotDriver(robot, RAW, resend_interval=0.5)


def test_stop_is_both_tracks_still():
    assert tank_for(Drive.STOP, ThrottleConfig()) == (0, 0)


def test_forward_drives_both_tracks_together():
    config = RAW
    left, right = tank_for(Drive.FORWARD, config)
    assert left == right == config.speed_forward


def test_fast_is_faster_than_forward():
    for config in (RAW, ThrottleConfig()):
        assert abs(tank_for(Drive.FORWARD_FAST, config)[0]) > abs(tank_for(Drive.FORWARD, config)[0])


def test_backward_reverses_both_tracks():
    left, right = tank_for(Drive.BACKWARD, RAW)
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


# --- flipping the direction -----------------------------------------------------

INVERTED = ThrottleConfig(invert_drive=True)


def test_the_shipped_default_is_flipped():
    assert ThrottleConfig().invert_drive is True


def test_flipped_forward_drives_the_tracks_the_other_way():
    left, right = tank_for(Drive.FORWARD, INVERTED)
    assert left == right == -INVERTED.speed_forward


def test_flipped_backward_is_positive():
    left, right = tank_for(Drive.BACKWARD, INVERTED)
    assert left == right == INVERTED.speed_backward > 0


@pytest.mark.parametrize("drive", [Drive.FORWARD, Drive.FORWARD_FAST, Drive.BACKWARD])
def test_flipping_exactly_mirrors_the_linear_motion(drive):
    raw, flipped = tank_for(drive, RAW), tank_for(drive, INVERTED)
    assert flipped == (-raw[0], -raw[1])


@pytest.mark.parametrize("drive", [Drive.TURN_LEFT, Drive.TURN_RIGHT, Drive.STOP])
def test_flipping_leaves_pivots_and_stop_alone(drive):
    """Inverting a pivot would swap left and right turns."""
    assert tank_for(drive, INVERTED) == tank_for(drive, RAW)


def test_forward_and_backward_still_oppose_each_other_when_flipped():
    forward = tank_for(Drive.FORWARD, INVERTED)[0]
    backward = tank_for(Drive.BACKWARD, INVERTED)[0]
    assert forward * backward < 0
