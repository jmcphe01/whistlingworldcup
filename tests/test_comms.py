"""The MQTT adapter, against a fake client with the same shape as `mqttlib`."""

from __future__ import annotations

import queue

import pytest

from whistle.comms import Comms, validate_topic


class FakeClient:
    def __init__(self):
        self.subscriptions: list[str] = []
        self.published: list[tuple] = []
        self.disconnected = False

    def subscribe(self, topic, callback, qos=0):
        self.subscriptions.append(topic)

    def publish(self, topic, message, qos=0, retain=False):
        self.published.append((topic, message, qos))

    def disconnect(self):
        self.disconnected = True


@pytest.fixture
def comms():
    inbox = queue.Queue()
    c = Comms(FakeClient(), "ME193/Rogers", inbox)
    c.start()
    return c


def test_it_subscribes_to_the_topic_on_start(comms):
    assert comms.client.subscriptions == ["ME193/Rogers"]


def test_messages_on_the_topic_reach_the_inbox(comms):
    comms._handle("ME193/Rogers", "start")
    assert comms.inbox.get_nowait() == "start"


def test_messages_on_an_old_topic_are_dropped_after_a_change(comms):
    """The wrapper has no unsubscribe, so the old topic stays subscribed at the
    broker. Its traffic must not start matches on the new one."""
    comms.set_topic("ME193/Test")
    comms._handle("ME193/Rogers", "start")
    assert comms.inbox.empty()
    comms._handle("ME193/Test", "start")
    assert comms.inbox.get_nowait() == "start"


def test_changing_topic_subscribes_to_the_new_one(comms):
    comms.set_topic("ME193/Test")
    assert comms.client.subscriptions[-1] == "ME193/Test"


def test_outcomes_are_published_at_qos_1(comms):
    """With QoS 0 a message sent while the link is down is simply lost, and the
    outcome messages are the ones that must arrive."""
    comms.publish("ME193/Rogers", "ball scored")
    assert comms.client.published == [("ME193/Rogers", "ball scored", 1)]


def test_a_rejected_topic_leaves_the_old_one_in_place(comms):
    with pytest.raises(ValueError):
        comms.set_topic("ME193/#")
    assert comms.topic == "ME193/Rogers"
    assert comms.client.subscriptions == ["ME193/Rogers"]


def test_closing_disconnects(comms):
    comms.close()
    assert comms.client.disconnected


def test_the_starting_topic_is_validated():
    with pytest.raises(ValueError):
        Comms(FakeClient(), "", queue.Queue())


@pytest.mark.parametrize("bad", ["", "  ", "a/#", "a/+", "/a", "a/"])
def test_bad_topics_are_refused(bad):
    with pytest.raises(ValueError):
        validate_topic(bad)


def test_nested_topics_are_fine():
    assert validate_topic("ME193/Rogers/team1") == "ME193/Rogers/team1"
