"""Typed test commands, for playing the other team against the real robot.

The console reads lines on a background thread and hands them to the main loop
through a queue, exactly as the monitor's buttons do, so a typed command and a
click take the same path. Parsing is a pure function; the thread is only the
plumbing around `input()`.

The commands split into two kinds, and the difference matters when testing:

  opponent messages   go out over MQTT and come back through the broker, so they
                      exercise the whole real path
  simulated events    happen locally, for checking the robot's reaction without
                      the physical trigger
"""

from __future__ import annotations

import queue
import threading

HELP = """
Test commands (type one and press Return). You are the other team.

  Over MQTT, through the broker and back:
    start            publish the start message
    tagged           publish the tagged message   (a goalie sings on this)
    scored           publish the goal message     (a goalie mourns on this)
    couchleft, couchright, couchstop   publish a couch command (as your partner would)
    send <text>      publish anything to the topic

  Simulated locally, no whistle or sensor needed:
    goal             pretend the ball whistled the goal command
    tag              pretend the goalie reached the light sensor

  Control:
    role ball|goalie switch role (starts a fresh match)
    topic <name>     change the topic (starts a fresh match)
    msg goal|tagged <text>   change what those two messages say
    reset            new match, back to waiting for start
    song win|lose    play a song, to check the speakers
    status           role, phase, topic, light sensor
    help, quit
"""

_OPPONENT = {"start", "tagged", "scored", "couchleft", "couchright", "couchstop"}
_PLAIN = {"status", "help", "quit"}


def parse_console_line(line: str) -> tuple[str, object] | None:
    """One typed line -> (kind, value), or None for a blank line.

    Raises ValueError with a message fit to show the user.
    """
    text = line.strip()
    if not text:
        return None

    word, _, rest = text.partition(" ")
    word = word.lower()
    rest = rest.strip()

    if word in _OPPONENT:
        return ("opponent", word)
    if word == "send":
        if not rest:
            raise ValueError("send what? e.g.  send hello")
        return ("publish", rest)          # case and spacing kept as typed
    if word == "goal":
        return ("sim_goal", None)
    if word == "tag":
        return ("sim_tag", None)
    if word in ("role", "topic"):
        if not rest:
            raise ValueError(f"{word} what? type help for examples")
        return (word, rest)
    if word in ("reset", "new"):
        return ("reset", None)
    if word == "msg":
        which, _, text = rest.partition(" ")
        if which.lower() not in ("goal", "tagged") or not text.strip():
            raise ValueError("msg goal <text>   or   msg tagged <text>")
        return ("message", (which.lower(), text.strip()))
    if word == "song":
        if rest.lower() not in ("win", "lose"):
            raise ValueError("song win  or  song lose")
        return ("song", rest.lower())
    if word == "exit":
        return ("quit", None)
    if word in _PLAIN:
        return (word, None)
    raise ValueError(f"unknown command {word!r} -- type help")


class ConsoleThread(threading.Thread):
    """Reads typed lines and posts the parsed commands to `sink`."""

    def __init__(self, sink: queue.Queue, read=input, log=print):
        super().__init__(daemon=True, name="console")
        self.sink = sink
        self._read = read
        self._log = log

    def run(self) -> None:
        while True:
            try:
                line = self._read("")
            except (EOFError, OSError):
                return          # no terminal attached; nothing to read
            try:
                command = parse_console_line(line)
            except ValueError as error:
                self._log(f"  {error}")
                continue
            if command is None:
                continue
            self.sink.put(command)
            if command[0] == "quit":
                return
