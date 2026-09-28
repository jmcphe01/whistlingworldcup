"""The typed test commands. Parsing is pure; the thread is exercised with a fake
`input`, so nothing here needs a terminal."""

from __future__ import annotations

import queue

import pytest

from whistle.console import ConsoleThread, parse_console_line


@pytest.mark.parametrize("line, expected", [
    ("start", ("opponent", "start")),
    ("TAGGED", ("opponent", "tagged")),
    ("  scored  ", ("opponent", "scored")),
    ("goal", ("sim_goal", None)),
    ("tag", ("sim_tag", None)),
    ("reset", ("reset", None)),
    ("new", ("reset", None)),
    ("role goalie", ("role", "goalie")),
    ("topic ME193/Test", ("topic", "ME193/Test")),
    ("song win", ("song", "win")),
    ("song LOSE", ("song", "lose")),
    ("status", ("status", None)),
    ("help", ("help", None)),
    ("quit", ("quit", None)),
    ("exit", ("quit", None)),
])
def test_commands_parse(line, expected):
    assert parse_console_line(line) == expected


def test_send_keeps_the_text_exactly_as_typed():
    """The opponent's wording is case sensitive to whoever reads it."""
    assert parse_console_line("send Ball  SCORED") == ("publish", "Ball  SCORED")


@pytest.mark.parametrize("line", ["", "   ", "\n"])
def test_blank_lines_are_ignored(line):
    assert parse_console_line(line) is None


@pytest.mark.parametrize("line", ["send", "role", "topic", "song", "song sing", "dance"])
def test_bad_commands_explain_themselves(line):
    with pytest.raises(ValueError):
        parse_console_line(line)


def run_console(lines):
    sink, messages = queue.Queue(), []
    feed = iter(lines)

    def read(_prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    thread = ConsoleThread(sink, read=read, log=messages.append)
    thread.start()
    thread.join(timeout=2.0)
    items = []
    while not sink.empty():
        items.append(sink.get_nowait())
    return items, messages


def test_the_thread_posts_parsed_commands():
    items, _ = run_console(["start", "", "tag"])
    assert items == [("opponent", "start"), ("sim_tag", None)]


def test_a_typo_is_reported_and_the_console_carries_on():
    items, messages = run_console(["nonsense", "goal"])
    assert items == [("sim_goal", None)]
    assert any("unknown command" in m for m in messages)


def test_the_thread_stops_at_quit():
    items, _ = run_console(["quit", "start"])
    assert items == [("quit", None)]


def test_the_thread_ends_quietly_with_no_terminal():
    items, _ = run_console([])
    assert items == []
