#!/usr/bin/env python3
"""The match program: whistle-driven LEGO car, MQTT signalling, songs.

    python main.py --monitor                 # pick role and topic in the window
    python main.py --role goalie --monitor
    python main.py --no-robot --no-mqtt      # bench test, nothing connected
    python main.py --list-devices

While it runs, type `help` in the terminal for test commands that let you play
the other team: publish start, tagged or scored over MQTT, or simulate a goal or a
tag locally.

How it is put together, and why:

  main thread      audio -> pitch -> interpreter -> motors. Nothing slow is
                   allowed on it, because this is the path between a whistle and
                   the wheels.
  monitor process  matplotlib, fed a drop-on-full queue. A separate process, so a
                   redraw can never sit in that path.
  sensor thread    polls the colour sensor over BLE, which is far too slow to do
                   inline at 86 frames a second.
  console thread   reads typed test commands.
  mqtt thread      paho's own network loop.

Requests from the monitor, the console and the broker all arrive on the main
thread as (kind, value) and are handled by whistle.session.Session, so match
state has exactly one writer.

The control scheme:

  hold a note steady   C4-A5 reverse, A5-A6 forward, above A6 forward fast
  stop whistling       stop
  slide the pitch up   pivot right, for as long as you keep sliding
  slide the pitch down pivot left
  warble left-right-left   claim the goal (a falling, rising, falling slide)
"""

from __future__ import annotations

import argparse
import multiprocessing
import queue
import sys
import time
from dataclasses import replace

from config import Config, throttle_bounds
from monitor import Snapshot, run_monitor
from whistle.comms import Comms, LoopbackComms, make_resilient_client
from whistle.console import HELP, ConsoleThread
from whistle.couch import CouchMotor
from whistle.driver import RobotDriver
from whistle.gestures import describe_pattern
from whistle.interpreter import Interpreter
from whistle.match import Match, MatchRunner, Role
from whistle.pitch import PitchDetector, measure_noise_floor
from whistle.sensor import SensorMonitor
from whistle.session import Session
from whistle.songs import BeepPlayer, PlayerPair, SongPlayer
from whistle.stream import (
    AudioStream,
    SilentInputError,
    check_audio_present,
    frames_for_seconds,
    list_input_devices,
    resolve_input_device,
)

MONITOR_HZ = 25.0
COORDINATED_KINDS = ("device", "sensitivity")    # only the newest of each matters


class NullRobot:
    """Stands in for the car so the audio half can be run at a desk."""

    def movement_move_tank(self, left, right):
        pass

    def motor_stop(self):
        pass


def print_devices() -> None:
    import pyaudio

    audio = pyaudio.PyAudio()
    try:
        for device in list_input_devices(audio):
            print(f"  {device}")
    finally:
        audio.terminate()


def enumerate_devices() -> list:
    """Every input device, read once in the parent so the monitor process does
    not need PortAudio of its own."""
    import pyaudio

    audio = pyaudio.PyAudio()
    try:
        return list_input_devices(audio)
    finally:
        audio.terminate()


def default_device_index() -> int | None:
    import pyaudio

    audio = pyaudio.PyAudio()
    try:
        return int(audio.get_default_input_device_info()["index"])
    except Exception:
        return None
    finally:
        audio.terminate()


def connect_robot(config: Config):
    """Connect the drive base using the class wrapper."""
    import lelib

    hardware = config.hardware
    print(f"Connecting to the double motor (card {hardware.card_color} "
          f"{hardware.card_serial})...")
    robot = lelib.doubleMotor()
    robot.connect(card_serial=hardware.card_serial, card_color=hardware.card_color)
    print("  connected")
    return robot


def connect_colour_sensor(config: Config):
    import lelib

    hardware = config.hardware
    print(f"Connecting to the colour sensor (card {hardware.card_color} "
          f"{hardware.card_serial})...")
    sensor = lelib.colorSensor()
    sensor.connect(card_serial=hardware.card_serial, card_color=hardware.card_color)
    print("  connected")
    return sensor


def connect_couch_motor(config: Config):
    import lelib

    hardware = config.hardware
    print(f"Connecting to the couch motor (card {hardware.card_color} "
          f"{hardware.card_serial})...")
    motor = lelib.singleMotor()
    motor.connect(card_serial=hardware.card_serial, card_color=hardware.card_color)
    print("  connected")
    return motor


def build_comms(args, config: Config, inbox: queue.Queue):
    """The broker connection, or a loopback standing in for it."""
    topic = config.mqtt.topic
    if args.no_mqtt:
        print("MQTT off: a loopback stands in for the broker, so what you publish "
              "comes straight back.")
        comms = LoopbackComms(topic, inbox)
        comms.start()
        return comms

    print(f"Connecting to {config.mqtt.broker}:{config.mqtt.port}, topic {topic}...")
    client = make_resilient_client(config.mqtt.broker, config.mqtt.port)
    client.connect()
    confirmed = getattr(client, "_connected", None)
    if confirmed is not None and not confirmed.is_set():
        print("  WARNING: the broker has not confirmed the connection. It will keep "
              "retrying, but nothing will be heard until it does.")
    else:
        print("  connected")

    comms = Comms(client, topic, inbox)
    comms.start()
    time.sleep(1.0)     # let the subscription reach the broker before we rely on it
    return comms


def calibrate_room(stream: AudioStream, detector: PitchDetector, config: Config) -> float:
    seconds = config.gates.calibration_seconds
    print(f"Measuring the room for {seconds:.1f}s -- please don't whistle yet...")
    frames = stream.read_frames(frames_for_seconds(seconds, config.audio))
    check_audio_present(frames, stream.device)
    floor = measure_noise_floor(frames, detector)
    detector.set_noise_floor(floor)
    print(f"  noise floor {floor:.1f} dBFS, gate opens at "
          f"{floor + config.gates.noise_margin_db:.1f} dBFS")
    return floor


def start_monitor(detector: PitchDetector, config: Config, devices, current_index,
                  role: Role, topic: str):
    """Launch the monitor process.

    Two queues: snapshots out to the monitor, commands back from it. The monitor
    never touches the audio device or the match itself -- clicking only posts a
    request, and this process, which owns them, decides what to do with it.
    """
    snapshots: multiprocessing.Queue = multiprocessing.Queue(maxsize=4)
    commands: multiprocessing.Queue = multiprocessing.Queue(maxsize=16)
    process = multiprocessing.Process(
        target=run_monitor,
        args=(snapshots, detector.band_frequencies, throttle_bounds(config.throttle),
              devices, current_index, commands, detector.sensitivity, role.value, topic,
              config.mqtt.goal_message, config.mqtt.tagged_message),
        daemon=True,
        name="monitor",
    )
    process.start()
    return process, snapshots, commands


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=[role.value for role in Role],
                        help="starting role; defaults to ball, and can be changed "
                             "in the monitor or with `role` in the console")
    parser.add_argument("--monitor", action="store_true",
                        help="open the live monitor in a separate process")
    parser.add_argument("--list-devices", action="store_true",
                        help="show the input devices and exit")
    parser.add_argument("--device", help="input device name or index")
    parser.add_argument("--broker", help="override the MQTT broker host")
    parser.add_argument("--topic", help="starting topic; can be changed in the monitor")
    parser.add_argument("--no-robot", action="store_true",
                        help="run the audio half with no car connected")
    parser.add_argument("--no-mqtt", action="store_true",
                        help="no broker: a loopback stands in, and the match starts "
                             "at once")
    parser.add_argument("--no-sensor", action="store_true",
                        help="skip the light sensor (the ball needs it to be tagged)")
    parser.add_argument("--speaker", action="store_true",
                        help="play the songs on the laptop speaker, not the robot's beeper")
    parser.add_argument("--no-couch", action="store_true",
                        help="do not connect the couch motor")
    parser.add_argument("--both", action="store_true",
                        help="play the songs on the robot and the laptop together")
    parser.add_argument("--no-console", action="store_true",
                        help="do not read typed test commands")
    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Fold the command-line overrides into the loaded config."""
    if args.device is not None:
        spec: int | str = int(args.device) if args.device.strip().isdigit() else args.device
        config = replace(config, audio=replace(config.audio, input_device=spec))
    mqtt = config.mqtt
    if args.broker is not None:
        mqtt = replace(mqtt, broker=args.broker)
    if args.topic is not None:
        mqtt = replace(mqtt, topic=args.topic)
    return replace(config, mqtt=mqtt)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.list_devices:
        print_devices()
        return 0

    role = Role(args.role or "ball")
    config = apply_overrides(Config.load(), args)
    detector = PitchDetector(config.audio.sample_rate, config.audio.frame_size,
                             config.gates)
    interpreter = Interpreter(config)
    match = Match(role, config.mqtt)

    print(f"Whistling World Cup -- starting as the {role.value} "
          "(change it in the monitor, or type `role goalie`)")

    robot = NullRobot() if args.no_robot else connect_robot(config)
    driver = RobotDriver(robot, config.throttle)

    # The songs play on the robot's own beeper, so they are audible wherever the
    # robot is. The laptop speaker is for a desk test with no robot, or on request.
    if args.no_robot or args.speaker:
        print("Songs will play on the laptop speaker.")
        player = SongPlayer(config.audio.sample_rate)
    else:
        player = BeepPlayer(robot, config.hardware.beep_sustain_seconds,
                            config.hardware.beep_octave_shift)
        if args.both:
            print("Songs will play on the robot and the laptop speaker together.")
            player = PlayerPair(player, SongPlayer(config.audio.sample_rate))

    # The sensor is connected for both roles, because the role can change while
    # the program runs. Only the ball ever arms it.
    sensor = None
    if args.no_robot or args.no_sensor or not config.hardware.use_color_sensor:
        print("Light sensor skipped: the ball cannot be tagged this run.")
    else:
        try:
            colour = connect_colour_sensor(config)
            sensor = SensorMonitor(lambda: float(colour.reflection()), config.sensor)
        except Exception as error:
            if role is Role.BALL:
                raise       # the rules require it, so do not start without it
            print(f"  WARNING: could not connect the light sensor ({error}). "
                  "Fine as the goalie, but switching to ball will not work.")

    # The couch is separate from the match, so failing to connect it is a warning
    # and never a reason not to play.
    couch = None
    if args.no_robot or args.no_couch or not config.hardware.couch_enabled:
        print("Couch motor skipped.")
    else:
        try:
            couch = CouchMotor(connect_couch_motor(config), config.hardware)
        except Exception as error:
            print(f"  WARNING: could not connect the couch motor ({error}). "
                  "The match still works; couch commands will be ignored.")

    inbox: queue.Queue = queue.Queue()
    comms = build_comms(args, config, inbox)
    runner = MatchRunner(match, comms, player, driver)
    session = Session(match, runner, driver, interpreter, comms, player, sensor, couch)

    console_queue: queue.Queue = queue.Queue()
    monitor_process, snapshots, commands = (None, None, None)
    exit_code = 0
    device_spec = config.audio.input_device
    previous_spec = device_spec
    opened_once = False
    clock_start = time.monotonic()      # one clock for the whole run, across mic switches

    try:
        if args.monitor:
            devices = enumerate_devices()
            current = resolve_input_device(devices, device_spec)
            if current is None:
                current = default_device_index()
            monitor_process, snapshots, commands = start_monitor(
                detector, config, devices, current, role, config.mqtt.topic)

        # The stream is reopened whenever the dropdown picks another microphone,
        # so the whole audio path lives inside this loop. The match, the robot and
        # the broker sit outside it and survive a switch untouched.
        while True:
            audio_config = replace(config.audio, input_device=device_spec)
            try:
                with AudioStream(audio_config) as stream:
                    print(f"Listening on {stream.device}")
                    calibrate_room(stream, detector, config)

                    if not opened_once:
                        opened_once = True
                        warning = session.arm_sensor()
                        if warning:
                            session.say(warning)
                        if args.no_mqtt:
                            print("  [mqtt off] starting immediately")
                            session.on_message(config.mqtt.start_message)
                        else:
                            print(f'Waiting for "{config.mqtt.start_message}" on '
                                  f"{match.config.topic}...")
                        if not args.no_console:
                            print(HELP)
                            ConsoleThread(console_queue).start()
                        print("(Ctrl-C to quit)")

                    interpreter.reset()   # the old device's pitch history is stale
                    switch_to = run_loop(stream, detector, interpreter, session, inbox,
                                         snapshots, commands, console_queue,
                                         clock_start)
            except SilentInputError as error:
                print(f"\nNo audio is reaching the program.\n\n{error}")
                if not opened_once:
                    exit_code = 1
                    break
                # A live switch to a dud device should not end the match.
                print("\nFalling back to the previous input device.")
                device_spec = previous_spec
                continue

            if switch_to is None:
                break
            previous_spec, device_spec = device_spec, switch_to
            print(f"Switching input device to index {switch_to}...")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Shutting down...")
        driver.stop()
        if sensor is not None:
            sensor.disarm()
        if couch is not None:
            couch.shutdown()      # or the couch keeps spinning after we quit
        if snapshots is not None:
            try:
                snapshots.put_nowait(None)
            except queue.Full:
                pass
        if monitor_process is not None:
            monitor_process.join(timeout=2.0)
            if monitor_process.is_alive():
                monitor_process.terminate()
        comms.close()
        player.close()
    return exit_code


def run_loop(stream, detector, interpreter, session: Session, inbox, snapshots, commands,
             console_queue, clock_start: float) -> int | None:
    """The control loop. One pass per analysis window, ~86 times a second.

    Returns the device index the monitor asked to switch to, or None when the
    program should stop. A finished match does *not* end the loop: the robot
    stays up so the next match can start with `reset` or the new-match button,
    without reconnecting Bluetooth.
    """
    next_snapshot = 0.0
    snapshot_interval = 1.0 / MONITOR_HZ
    device_name = stream.device.name if stream.device else ""

    for frame in stream.frames():
        now = time.monotonic() - clock_start

        switch_to = None
        for kind, value in collect_commands(commands, console_queue):
            if kind == "sensitivity":
                # Gates only: no stream reopen, so this takes effect on the next frame.
                detector.set_sensitivity(value)
            elif kind == "device":
                switch_to = value
            else:
                session.handle(kind, value)
        if session.quit:
            return None
        if switch_to is not None:
            return switch_to

        while True:
            try:
                payload = inbox.get_nowait()
            except queue.Empty:
                break
            session.on_message(payload)

        reading, spectrum_db = detector.analyse_with_spectrum(frame)
        intent = interpreter.update(now, reading)
        session.step(now, intent)

        if snapshots is not None and now >= next_snapshot:
            next_snapshot = now + snapshot_interval
            push_snapshot(snapshots, now, reading, spectrum_db, intent, interpreter,
                          session, detector, device_name)
    return None


def collect_commands(*sources) -> list[tuple[str, object]]:
    """Every pending (kind, value) from the given queues, oldest first. Never blocks.

    Only the newest device and the newest sensitivity are kept. Dragging the
    slider posts several values in a second and clicking around the dropdown
    posts several devices; replaying each would mean a stream reopen and a
    recalibration per tick. Everything else is a discrete request and is kept in
    order.
    """
    pending: list[tuple[str, object]] = []
    for source in sources:
        if source is None:
            continue
        while True:
            try:
                item = source.get_nowait()
            except (queue.Empty, ValueError, OSError):
                break
            try:
                kind, value = item
            except (TypeError, ValueError):
                continue
            pending.append((kind, value))

    kept: list[tuple[str, object]] = []
    seen: set[str] = set()
    for kind, value in reversed(pending):
        if kind in COORDINATED_KINDS:
            if kind in seen:
                continue
            seen.add(kind)
        kept.append((kind, value))
    kept.reverse()
    return kept


def push_snapshot(snapshots, now, reading, spectrum_db, intent, interpreter,
                  session: Session, detector, device_name="") -> None:
    """Hand the monitor a frame, or skip it. Never block the control loop."""
    match = session.match
    goal = interpreter.goal
    snapshot = Snapshot(
        t=now,
        spectrum_db=spectrum_db,
        frequency=intent.frequency,
        note=intent.note,
        cents_off=reading.cents_off,
        level_db=reading.level_db,
        noise_floor_db=reading.noise_floor_db,
        peak_to_median_db=reading.peak_to_median_db,
        peak_to_second_db=reading.peak_to_second_db,
        subharmonic_db=min(reading.subharmonic_db, 999.0),
        reject_reason=reading.reject_reason,
        drive=intent.drive.value,
        steering=intent.steering,
        phase=match.phase.value,
        role=match.role.value,
        slide_rate=intent.slide_rate or interpreter.motion.rate_cents,
        motion=intent.motion.value,
        goal_progress=goal.legs_matched,
        goal_pattern=describe_pattern(goal.pattern),
        topic=match.config.topic,
        goal_message=match.config.goal_message,
        tagged_message=match.config.tagged_message,
        notice=session.notice,
        light=session.light_readout(),
        couch=session.couch_state(),
        gate_thresholds=detector.gate_thresholds,
        sensitivity=detector.sensitivity,
        device=device_name,
    )
    try:
        snapshots.put_nowait(snapshot)
    except queue.Full:
        pass   # the monitor is behind; it only ever wants the freshest frame


if __name__ == "__main__":
    sys.exit(main())
