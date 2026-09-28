"""MQTT plumbing around the course's `mqttlib` wrapper.

Two implementations share one small interface -- `topic`, `publish`,
`set_topic`, `close` -- so the rest of the program never asks which it has:

  Comms          talks to the real broker through `mqttlib.MQTTClient`
  LoopbackComms  a broker with exactly one client, for `--no-mqtt`: whatever you
                 publish comes straight back, as it would from a real broker

Both feed received payloads into a queue rather than calling into match logic.
The paho callback runs on its own thread, and the match state belongs to the
main loop.
"""

from __future__ import annotations

import queue


def validate_topic(text: str) -> str:
    """Clean and check a topic typed into the UI.

    Wildcards are refused because this program publishes to the topic as well as
    listening on it: publishing to a wildcard is invalid, and subscribing to one
    would also deliver traffic from other teams' topics.
    """
    topic = (text or "").strip()
    if not topic:
        raise ValueError("the topic is empty")
    if "#" in topic or "+" in topic:
        raise ValueError("wildcards (# and +) are not allowed in the topic")
    if topic.startswith("/") or topic.endswith("/"):
        raise ValueError("a leading or trailing '/' makes a different topic; "
                         "check for a typo")
    return topic


def make_resilient_client(broker: str, port: int):
    """An `MQTTClient` that resubscribes after paho reconnects.

    paho reconnects a dropped link by itself, but the course wrapper only
    subscribes once, so after a reconnect the car would stay connected and deaf:
    it would never hear "start" again. Venue Wi-Fi drops connections, so this
    matters. It leans on the wrapper's private attributes, which is acceptable
    for a file copied into this repo and not expected to change.
    """
    from mqttlib import MQTTClient

    class ResilientClient(MQTTClient):
        def _on_connect(self, client, userdata, flags, reason_code, properties):
            super()._on_connect(client, userdata, flags, reason_code, properties)
            for topic in list(self._callbacks):
                self._client.subscribe(topic)

    return ResilientClient(broker, port)


class Comms:
    """The real broker."""

    # QoS 1 for what we publish: the outcome messages are the ones that must
    # arrive, and with QoS 0 a message sent while the link is down is dropped.
    PUBLISH_QOS = 1

    def __init__(self, client, topic: str, inbox: queue.Queue):
        self.client = client
        self.topic = validate_topic(topic)
        self.inbox = inbox

    def start(self) -> None:
        self.client.subscribe(self.topic, self._handle)

    def _handle(self, topic: str, payload: str) -> None:
        # Old topics stay subscribed at the broker after a change, because the
        # wrapper has no unsubscribe. Their traffic is dropped here.
        if topic == self.topic:
            self.inbox.put(payload)

    def publish(self, topic: str, message: str) -> None:
        self.client.publish(topic, message, qos=self.PUBLISH_QOS)

    def set_topic(self, topic: str) -> str:
        topic = validate_topic(topic)
        self.topic = topic
        self.client.subscribe(topic, self._handle)
        return topic

    def close(self) -> None:
        self.client.disconnect()


class LoopbackComms:
    """A broker with one client: your own publishes come back to you."""

    def __init__(self, topic: str, inbox: queue.Queue, log=print):
        self.topic = validate_topic(topic)
        self.inbox = inbox
        self._log = log

    def start(self) -> None:
        pass

    def publish(self, topic: str, message: str) -> None:
        self._log(f"  [mqtt off] {topic}: {message!r}")
        if topic == self.topic:
            self.inbox.put(message)

    def set_topic(self, topic: str) -> str:
        self.topic = validate_topic(topic)
        return self.topic

    def close(self) -> None:
        pass
