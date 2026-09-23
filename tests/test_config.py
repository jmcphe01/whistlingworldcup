"""Config defaults, local overrides and CLI folding."""

from __future__ import annotations

import json

import pytest

import main
from config import Config, throttle_bounds
from whistle.notes import note_to_hz


def test_defaults_load_without_a_local_file(tmp_path):
    config = Config.load(tmp_path / "absent.json")
    assert config.audio.sample_rate == 44100
    assert config.mqtt.topic == "ME193/Rogers"


def test_the_hardware_matches_the_connection_cards():
    hardware = Config().hardware
    assert (hardware.card_color, hardware.card_serial) == ("red", 1129)


def test_local_overrides_are_applied(tmp_path):
    path = tmp_path / "config.local.json"
    path.write_text(json.dumps({
        "audio": {"input_device": "Scarlett"},
        "gates": {"noise_floor_db": -48.5},
    }))
    config = Config.load(path)
    assert config.audio.input_device == "Scarlett"
    assert config.gates.noise_floor_db == -48.5


def test_overriding_one_field_leaves_the_rest_alone(tmp_path):
    path = tmp_path / "config.local.json"
    path.write_text(json.dumps({"gates": {"noise_floor_db": -40.0}}))
    config = Config.load(path)
    assert config.gates.noise_floor_db == -40.0
    assert config.gates.peak_to_second_db == Config().gates.peak_to_second_db


def test_an_unknown_section_is_an_error_not_a_silent_no_op(tmp_path):
    """A typo in the file must be loud: a silently ignored calibration would be
    discovered during the match."""
    path = tmp_path / "config.local.json"
    path.write_text(json.dumps({"gatez": {"noise_floor_db": -40.0}}))
    with pytest.raises(ValueError, match="gatez"):
        Config.load(path)


def test_an_unknown_field_is_an_error(tmp_path):
    path = tmp_path / "config.local.json"
    path.write_text(json.dumps({"gates": {"noize_floor_db": -40.0}}))
    with pytest.raises(TypeError):
        Config.load(path)


# --- the zones ---------------------------------------------------------------

def test_the_throttle_zones_are_where_the_notes_say():
    low, high = throttle_bounds(Config().throttle)
    assert low == pytest.approx(note_to_hz("F#5"))
    assert high == pytest.approx(note_to_hz("A6"))


def test_the_target_notes_sit_inside_their_zones():
    """D7 is the "go fast" note and C4 the lowest reverse note; both must be
    comfortably inside, not on an edge."""
    low, high = throttle_bounds(Config().throttle)
    assert note_to_hz("D7") > high * 1.2
    assert note_to_hz("C4") < low * 0.8


def test_the_search_band_covers_the_whole_whistle_range():
    gates = Config().gates
    assert gates.min_hz < note_to_hz("C4")
    assert gates.max_hz > note_to_hz("D7")


def test_the_goal_chirps_sit_above_the_top_zone_edge():
    """So claiming the goal is a distinct register, not a drive command."""
    config = Config()
    assert config.gestures.goal_chirp_min_hz >= throttle_bounds(config.throttle)[1]


def test_a_chirp_is_shorter_than_a_sweep():
    """Otherwise the two gestures would overlap."""
    gestures = Config().gestures
    assert gestures.goal_chirp_max_duration < gestures.sweep_min_duration


# --- CLI overrides -----------------------------------------------------------

def parse(*argv):
    return main.build_arg_parser().parse_args(list(argv))


def test_the_broker_and_topic_can_be_overridden_on_the_day():
    config = main.apply_overrides(
        Config(), parse("--role", "ball", "--broker", "10.0.0.5", "--topic", "ME193/Test"))
    assert config.mqtt.broker == "10.0.0.5"
    assert config.mqtt.topic == "ME193/Test"


def test_a_numeric_device_argument_becomes_an_index():
    config = main.apply_overrides(Config(), parse("--role", "ball", "--device", "3"))
    assert config.audio.input_device == 3


def test_a_named_device_argument_stays_a_string():
    config = main.apply_overrides(Config(), parse("--role", "ball", "--device", "Scarlett"))
    assert config.audio.input_device == "Scarlett"


def test_no_overrides_changes_nothing():
    assert main.apply_overrides(Config(), parse("--role", "goalie")) == Config()


def test_both_roles_are_accepted():
    assert parse("--role", "ball").role == "ball"
    assert parse("--role", "goalie").role == "goalie"


def test_an_invalid_role_is_rejected():
    with pytest.raises(SystemExit):
        parse("--role", "referee")
