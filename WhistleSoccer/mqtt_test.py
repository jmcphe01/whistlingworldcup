"""Send test messages to ME193/Rogers on test.mosquitto.org, bypassing mqttlib.

If these show up in the web debugger, the broker/topic are fine and the
problem is whatever broker mqttlib.MQTTClient connects to.
"""

import time

import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "ME193/Rogers"

try:  # paho-mqtt 2.x
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:  # paho-mqtt 1.x
    client = mqtt.Client()

client.connect(BROKER, PORT, keepalive=60)
client.loop_start()

for i in range(5):
    info = client.publish(TOPIC, f"HELLO {i}")
    info.wait_for_publish()
    print(f"sent 'HELLO {i}' to {TOPIC} on {BROKER}")
    time.sleep(1)

client.loop_stop()
client.disconnect()
