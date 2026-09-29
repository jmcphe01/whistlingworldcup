import time

from mqttlib import MQTTClient

TOPIC = "ME193"


def on_message(topic, payload):
    print(f"Got it back: [{topic}] {payload}")


# Uses the public test.mosquitto.org broker configured in mqttlib.py --
# no credentials needed.
with MQTTClient() as client:
    client.subscribe(TOPIC, on_message)
    time.sleep(1)  # give the subscription time to reach the broker

    client.publish(TOPIC, "hello world")
    print(f"Published 'hello world' to '{TOPIC}'")

 # give the message time to come back before disconnecting
