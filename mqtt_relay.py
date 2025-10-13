import json
import os
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from urllib.parse import urlparse


subscribers_lock = threading.Lock()
subscribers = []  # list[Queue]


def _register_client_queue():
    q = Queue(maxsize=64)
    with subscribers_lock:
        subscribers.append(q)
    return q


def _unregister_client_queue(q):
    with subscribers_lock:
        if q in subscribers:
            subscribers.remove(q)


def broadcast_event(event_dict):
    with subscribers_lock:
        for q in list(subscribers):
            try:
                q.put_nowait(event_dict)
            except Exception:
                try:
                    _ = q.get_nowait()
                    q.put_nowait(event_dict)
                except Exception:
                    pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_index()
            return
        if self.path == "/events":
            self._serve_sse()
            return
        self.send_response(404)
        self.end_headers()

    def _serve_index(self):
        index_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
        try:
            with open(index_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<h1>MQTT Relay</h1><p>Missing web/index.html</p>")

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        client_q = _register_client_queue()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
        except Exception:
            _unregister_client_queue(client_q)
            return
        try:
            while True:
                data = client_q.get()
                payload = ("data: " + json.dumps(data) + "\n\n").encode("utf-8")
                self.wfile.write(payload)
                self.wfile.flush()
        except Exception:
            pass
        finally:
            _unregister_client_queue(client_q)


def start_http_server(host="0.0.0.0", port=8080):
    server = ThreadingHTTPServer((host, port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def start_mqtt_subscriber():
    # Create MQTT client (explicitly select Callback API v1 to avoid warnings on paho-mqtt v2)
    try:
        from paho.mqtt.client import Client, CallbackAPIVersion
        client = Client(client_id=os.getenv("MQTT_CLIENT_ID") or "relay-sse", callback_api_version=CallbackAPIVersion.VERSION2)
    except Exception:
        try:
            import paho.mqtt.client as mqtt
        except Exception as e:
            print(f"[relay] paho-mqtt not available: {e}")
            return None
        client = mqtt.Client(client_id=os.getenv("MQTT_CLIENT_ID") or "relay-sse")

    url = os.getenv("MQTT_URL")
    if not url:
        print("[relay] MQTT_URL not set; cannot subscribe")
        return None

    parsed = urlparse(url)
    scheme = (parsed.scheme or "mqtt").lower()
    host = parsed.hostname or "localhost"
    port = parsed.port or (8883 if scheme in ("mqtts", "ssl", "tls") else 1883)
    user = os.getenv("MQTT_USER")
    password = os.getenv("MQTT_PASS")
    cafile = os.getenv("MQTT_CAFILE")
    sub_topic = os.getenv("MQTT_SUB_TOPIC", "drone/+/events")

    if user:
        client.username_pw_set(user, password or None)

    if scheme in ("mqtts", "ssl", "tls"):
        ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
        client.tls_set_context(ctx)

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("[relay] connected; subscribing to", sub_topic)
            client.subscribe(sub_topic, qos=1)
        else:
            print("[relay] connect failed rc=", rc)

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            broadcast_event(data)
        except Exception:
            pass

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(host, port, keepalive=30)
    client.loop_start()
    return client


if __name__ == "__main__":
    http_host = os.getenv("HTTP_HOST", "0.0.0.0")
    http_port = int(os.getenv("HTTP_PORT", "8080"))
    start_http_server(http_host, http_port)
    print(f"Relay HTTP server on http://{http_host}:{http_port}")
    start_mqtt_subscriber()
    # Block main thread
    threading.Event().wait()
