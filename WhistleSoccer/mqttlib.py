'''
Shared MQTT wrapper for all ME193 examples.

Usage:
    from mqttlib import MQTTClient
'''

import os
import ssl
import threading
import uuid

import paho.mqtt.client as mqtt

BROKER_HOST = "test.mosquitto.org"
BROKER_PORT = 8883  # TLS port; test.mosquitto.org is public/anonymous, no username or password needed

# test.mosquitto.org's TLS ports (8883/8081) use their own self-signed CA,
# not one a normal CA bundle (including certifi) trusts -- downloaded from
# https://test.mosquitto.org/ssl/mosquitto.org.crt and checked into this
# folder so tls_set() below can verify the server's certificate against it.
_CA_CERT_PATH = os.path.join(os.path.dirname(__file__), "mosquitto_org_ca.crt")

CONNECT_TIMEOUT_S = 10  # how long connect() waits for the broker's CONNACK


class MQTTClient:
    """Thin wrapper around paho-mqtt for topic subscribe/publish with
    per-topic callbacks. Use as a context manager so the connection is
    always closed cleanly:

        with MQTTClient() as client:
            client.subscribe(TOPIC, on_message)
            client.publish(TOPIC, "hello world")

    test.mosquitto.org is a public broker -- no username/password needed,
    everyone in the class (and beyond) can publish/subscribe to any topic
    on it. If you ever point this at a broker that does need auth, set
    MQTT_USERNAME/MQTT_PASSWORD as environment variables (or pass
    username=/password= explicitly to MQTTClient()) and it'll pick them up.

    `connect()` blocks until the broker's CONNACK actually arrives (or
    raises after CONNECT_TIMEOUT_S) instead of returning as soon as the
    TCP/TLS handshake starts -- paho-mqtt's own `Client.connect()` returns
    before that handshake finishes, so publishing right after it returns
    can silently drop messages (QoS 0 has no queueing) if the CONNACK
    hasn't arrived yet. `publish()` also logs a warning if the broker
    round-trip fails, instead of failing silently.
    """

    def __init__(
        self,
        host=BROKER_HOST,
        port=BROKER_PORT,
        client_id=None,
        username=None,
        password=None,
        use_tls=True,
    ):
        self.host = host
        self.port = port
        self._callbacks = {}
        self._connected_event = threading.Event()

        client_id = client_id or f"me193-{uuid.uuid4().hex[:8]}"
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_message = self._on_message
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        if use_tls:
            self._client.tls_set(ca_certs=_CA_CERT_PATH, tls_version=ssl.PROTOCOL_TLS_CLIENT)

        username = username or os.environ.get("MQTT_USERNAME")
        password = password or os.environ.get("MQTT_PASSWORD")
        if username:
            self._client.username_pw_set(username, password)

    def connect(self):
        self._client.connect(self.host, self.port)
        self._client.loop_start()
        if not self._connected_event.wait(timeout=CONNECT_TIMEOUT_S):
            raise ConnectionError(
                f"No CONNACK from {self.host}:{self.port} within {CONNECT_TIMEOUT_S}s "
                "-- check your network/firewall allows outbound MQTT (port "
                f"{self.port})."
            )
        return self

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()

    def subscribe(self, topic, callback):
        """Subscribe to topic, calling callback(topic, payload) for each message."""
        self._callbacks[topic] = callback
        self._client.subscribe(topic)

    def publish(self, topic, payload, qos=1, retain=False):
        """Publish payload to topic. qos=1 (at-least-once) by default so
        the broker retries delivery instead of a fire-and-forget QoS 0
        send that can drop silently on a flaky connection."""
        info = self._client.publish(topic, payload, qos=qos, retain=retain)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[mqtt] publish to {topic!r} failed immediately: {mqtt.error_string(info.rc)}")
        return info

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print(f"[mqtt] connected to {self.host}:{self.port}")
            self._connected_event.set()
        else:
            print(f"[mqtt] connection to {self.host}:{self.port} refused: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        print(f"[mqtt] disconnected from {self.host}:{self.port} ({reason_code})")
        self._connected_event.clear()

    def _on_message(self, client, userdata, message):
        callback = self._callbacks.get(message.topic)
        if callback:
            callback(message.topic, message.payload.decode())

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
