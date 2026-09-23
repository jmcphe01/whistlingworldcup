#!/usr/bin/env python3
"""The match program: whistle-driven LEGO car, MQTT signalling, songs.

    python main.py --role ball    --monitor
    python main.py --role goalie  --monitor
    python main.py --role ball    --no-robot     # bench test, audio only
    python main.py --list-devices

How it is put together, and why:

  main thread      audio -> pitch -> interpreter -> motors. Nothing slow is
                   allowed on it, because this is the path between a whistle and
                   the wheels.
  monitor process  matplotlib, fed a drop-on-full queue. A separate process, so a
                   redraw can never sit in that path.
  sensor thread    polls the colour sensor over BLE, which is far too slow to do
                   inline at 86 frames a second.
  mqtt thread      paho's own network loop.

The control scheme:

  hold a note          C4-F#5 reverse, F#5-A6 forward, above A6 forward fast
  stop whistling       stop
  long rising sweep    pivot right
  long falling sweep   pivot left
  three high chirps    claim the goal
"""

from __future__ import annotations

import argparse
import multiprocessing
import queue
import sys
import threading
import time
from dataclasses import replace

from config import Config, throttle_bounds
from monitor import Snapshot, run_monitor
from whistle.commands import Drive
from whistle.driver import RobotDriver
from whistle.interpreter import Interpreter
from whistle.match import Match, MatchRunner, Phase, Role
from whistle.pitch import PitchDetector, measure_noise_floor
from whistle.sensor import ProximityWatch
from whistle.songs import SongPlayer
from whistle.stream import (
    AudioStream,
    SilentInputError,
    check_audio_present,
    frames_for_seconds,
    list_input_devices,
    resolve_input_device,
)

MONITOR_HZ = 25.0


class NullRobot:
    """Stands in for the car so the audio half can be run at a desk."""

    def movement_move_tank(self, left, right):
        pass

    def motor_stop(self):
        pass


class NullPublisher:
    def publish(self, topic, message):
        print(f"  [mqtt off] would publish {message!r} to {topic}")


def print_devices() -> None:
    import pyaudio

    audio = pyaudio.PyAudio()
    try:
        for device in list_input_devices(audio):
            print(f"  {device}")
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


class SensorThread(threading.Thread):
    """Polls the forward light sensor. BLE reads are far too slow to do inline.

    Sets an event instead of acting, so the decision stays on the main thread
    with the rest of the match logic.
    """

    def __init__(self, sensor, watch: ProximityWatch, interval: float):
        super().__init__(daemon=True, name="proximity")
        self.sensor = sensor
        self.watch = watch
        self.interval = interval
        self.tripped = threading.Event()
        self._stop = threading.Event()
        self.last_reflection: float | None = None

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                reflection = float(self.sensor.reflection())
            except Exception as error:      # a BLE hiccup must not end the match
                print(f"  [sensor] read failed: {error}")
                time.sleep(self.interval)
                continue
            self.last_reflection = reflection
            if self.watch.update(reflection):
                self.tripped.set()
                return
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()


def baseline_sensor(sensor, watch: ProximityWatch, config: Config) -> float:
    seconds = config.sensor.baseline_seconds
    print(f"Baselining the light sensor for {seconds:.1f}s -- keep the area in "
          "front of it clear...")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            watch.add_baseline_sample(float(sensor.reflection()))
        except Exception as error:
            print(f"  [sensor] read failed: {error}")
        time.sleep(config.sensor.poll_interval)

    baseline = watch.finish_baseline()
    print(f"  baseline {baseline:.1f}, triggers at {watch.threshold:.1f}")
    return baseline


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


def start_monitor(detector: PitchDetector, config: Config, devices, current_index):
    """Launch the monitor process.

    Two queues: snapshots out to the monitor, commands back from it. The monitor
    never touches the audio device itself -- picking one in the dropdown only
    posts a request, and this process, which owns the stream, decides what to do
    with it.
    """
    snapshots: multiprocessing.Queue = multiprocessing.Queue(maxsize=4)
    commands: multiprocessing.Queue = multiprocessing.Queue(maxsize=8)
    process = multiprocessing.Process(
        target=run_monitor,
        args=(snapshots, detector.band_frequencies, throttle_bounds(config.throttle),
              devices, current_index, commands, detector.sensitivity),
        daemon=True,
        name="monitor",
    )
    process.start()
    return process, snapshots, commands


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=[role.value for role in Role],
                        help="the role you were assigned on the day")
    parser.add_argument("--monitor", action="store_true",
                        help="open the live pitch monitor in a separate process")
    parser.add_argument("--list-devices", action="store_true",
                        help="show the input devices and exit")
    parser.add_argument("--device", help="input device name or index")
    parser.add_argument("--broker", help="override the MQTT broker host")
    parser.add_argument("--topic", help="override the MQTT topic")
    parser.add_argument("--no-robot", action="store_true",
                        help="run the audio half with no car connected")
    parser.add_argument("--no-mqtt", action="store_true",
                        help="run with no broker; start locally instead of waiting")
    parser.add_argument("--no-sensor", action="store_true",
                        help="skip the light sensor (it is required for the ball)")
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
    if args.role is None:
        build_arg_parser().error("--role is required (ball or goalie)")

    role = Role(args.role)
    config = apply_overrides(Config.load(), args)
    detector = PitchDetector(config.audio.sample_rate, config.audio.frame_size,
                             config.gates)
    interpreter = Interpreter(config)
    match = Match(role, config.mqtt)
    player = SongPlayer(config.audio.sample_rate)

    print(f"Whistling World Cup -- role: {role.value}")

    robot = NullRobot() if args.no_robot else connect_robot(config)
    driver = RobotDriver(robot, config.throttle)

    sensor_thread = None
    wants_sensor = role is Role.BALL and config.hardware.use_color_sensor and not args.no_sensor
    if wants_sensor:
        if args.no_robot:
            print("  [no-robot] skipping the light sensor too")
        else:
            watch = ProximityWatch(config.sensor)
            sensor = connect_colour_sensor(config)
            baseline_sensor(sensor, watch, config)
            sensor_thread = SensorThread(sensor, watch, config.sensor.poll_interval)
    elif role is Role.BALL:
        print("  WARNING: running as the ball with no light sensor. The rules "
              "require it to be open and facing forward.")

    client = None
    publisher: object = NullPublisher()
    if not args.no_mqtt:
        from mqttlib import MQTTClient

        print(f"Connecting to {config.mqtt.broker}:{config.mqtt.port}, "
              f"topic {config.mqtt.topic}...")
        client = MQTTClient(config.mqtt.broker, config.mqtt.port)
        client.connect()
        publisher = client
        print("  connected")

    runner = MatchRunner(match, publisher, player, driver)
    inbox: queue.Queue = queue.Queue()

    if client is not None:
        # The paho callback runs on its own thread; hand the payload to the main
        # loop rather than mutating match state from under it.
        client.subscribe(config.mqtt.topic, lambda topic, payload: inbox.put(payload))
        time.sleep(1.0)   # let the subscription reach the broker before we rely on it

    monitor_process, snapshots, commands = (None, None, None)
    exit_code = 0
    device_spec = config.audio.input_device
    previous_spec = device_spec
    opened_once = False

    try:
        if args.monitor:
            devices = enumerate_devices()
            current = resolve_input_device(devices, device_spec)
            if current is None:
                current = default_device_index()
            monitor_process, snapshots, commands = start_monitor(
                detector, config, devices, current)

        # The stream is reopened whenever the dropdown picks another microphone,
        # so the whole audio path lives inside this loop. Match state, the robot
        # and the broker sit outside it and survive a switch untouched.
        while True:
            audio_config = replace(config.audio, input_device=device_spec)
            try:
                with AudioStream(audio_config) as stream:
                    print(f"Listening on {stream.device}")
                    calibrate_room(stream, detector, config)

                    if not opened_once:
                        opened_once = True
                        if args.no_mqtt:
                            print("  [mqtt off] starting immediately")
                            runner.handle(match.on_message(config.mqtt.start_message))
                        else:
                            print(f'Waiting for "{config.mqtt.start_message}" on '
                                  f"{config.mqtt.topic}...  (Ctrl-C to quit)")
                        if sensor_thread is not None:
                            sensor_thread.start()

                    interpreter.reset()   # the old device's pitch history is stale
                    switch_to = run_loop(stream, detector, interpreter, driver, runner,
                                         inbox, sensor_thread, snapshots, commands,
                                         config, role)
            except SilentInputError as error:
                print(f"\nNo audio is reaching the program.\n\n{error}")
                if not opened_once:
                    exit_code = 1
                    break
                # A live switch to a dud device should not end the match.
                print(f"\nFalling back to the previous input device.")
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
        if sensor_thread is not None:
            sensor_thread.stop()
        if snapshots is not None:
            try:
                snapshots.put_nowait(None)
            except queue.Full:
                pass
        if monitor_process is not None:
            monitor_process.join(timeout=2.0)
            if monitor_process.is_alive():
                monitor_process.terminate()
        if client is not None:
            client.disconnect()
        player.close()
    return exit_code


def run_loop(stream, detector, interpreter, driver, runner, inbox, sensor_thread,
             snapshots, commands, config: Config, role: Role) -> int | None:
    """The control loop. One pass per analysis window, ~86 times a second.

    Returns the device index the monitor asked to switch to, or None when the
    match is finished and the program should stop.
    """
    match = runner.match
    started = time.monotonic()
    next_snapshot = 0.0
    snapshot_interval = 1.0 / MONITOR_HZ
    announced = match.phase

    device_name = stream.device.name if stream.device else ""

    for frame in stream.frames():
        now = time.monotonic() - started

        requested_device, requested_sensitivity = drain_commands(commands)
        if requested_sensitivity is not None:
            # Gates only: no stream reopen, so this takes effect on the next frame.
            detector.set_sensitivity(requested_sensitivity)
        if requested_device is not None:
            return requested_device

        while True:
            try:
                runner.handle(match.on_message(inbox.get_nowait()))
            except queue.Empty:
                break

        reading, spectrum_db = detector.analyse_with_spectrum(frame)
        intent = interpreter.update(now, reading)

        if match.running:
            if sensor_thread is not None and sensor_thread.tripped.is_set():
                runner.handle(match.on_proximity())
            elif intent.goal_whistle:
                runner.handle(match.on_goal_whistle())
            else:
                driver.apply(now, intent.drive)
        else:
            driver.apply(now, Drive.STOP)

        if match.phase is not announced:
            print(f"  phase: {match.phase.value}")
            announced = match.phase

        if snapshots is not None and now >= next_snapshot:
            next_snapshot = now + snapshot_interval
            push_snapshot(snapshots, now, reading, spectrum_db, intent, interpreter,
                          match, role, config, device_name, detector)

        if match.phase is Phase.OVER:
            print(f"  match over: {runner.log[-1] if runner.log else 'done'}")
            return None
    return None


def drain_commands(commands) -> tuple[int | None, float | None]:
    """Latest (device, sensitivity) the monitor asked for. Never blocks.

    Only the newest of each is kept. Dragging the slider posts several values in
    a second and clicking around the dropdown posts several devices; replaying
    every one would mean a stream reopen and a recalibration per tick.
    """
    if commands is None:
        return None, None
    device = sensitivity = None
    while True:
        try:
            kind, value = commands.get_nowait()
        except (queue.Empty, ValueError, OSError):
            break
        if kind == "device":
            device = value
        elif kind == "sensitivity":
            sensitivity = value
    return device, sensitivity


def push_snapshot(snapshots, now, reading, spectrum_db, intent, interpreter,
                  match, role, config, device_name="", detector=None) -> None:
    """Hand the monitor a frame, or skip it. Never block the control loop."""
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
        role=role.value,
        sweep_cents=intent.sweep_cents or interpreter.sweeps.travel_cents,
        chirps=interpreter.chirps.chirps_so_far,
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
