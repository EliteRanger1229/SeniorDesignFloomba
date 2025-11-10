import cv2
# Renamed from jil.py to PD_main.py
import json
import os
import threading
import time
import platform
import ssl
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from urllib.parse import urlparse

# ---------------------------
# Simple SSE broadcast server
# ---------------------------
subscribers_lock = threading.Lock()
subscribers = []  # list[Queue]

# MQTT publisher (optional; enabled via env)
_mqtt_publisher = None
_mqtt_initialized = False


def _register_client_queue():
    q = Queue(maxsize=16)
    with subscribers_lock:
        subscribers.append(q)
    return q


def _unregister_client_queue(q):
    with subscribers_lock:
        if q in subscribers:
            subscribers.remove(q)


def broadcast_event(event_dict):
    # Fan-out to all subscriber queues without blocking the detector loop
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
    # Also enqueue for MQTT if configured (lazy init)
    try:
        global _mqtt_publisher, _mqtt_initialized
        if not _mqtt_initialized:
            _mqtt_initialized = True
            mqtt_url = os.getenv("MQTT_URL")
            if mqtt_url:
                _mqtt_publisher = MqttPublisher(
                    url=mqtt_url,
                    user=os.getenv("MQTT_USER"),
                    password=os.getenv("MQTT_PASS"),
                    topic=os.getenv("MQTT_TOPIC"),
                    qos=os.getenv("MQTT_QOS", "1"),
                    client_id=os.getenv("MQTT_CLIENT_ID"),
                    cafile=os.getenv("MQTT_CAFILE"),
                    drone_id=os.getenv("DRONE_ID") or platform.node(),
                )
                _mqtt_publisher.start()
        if _mqtt_publisher is not None:
            _mqtt_publisher.enqueue(event_dict)
    except Exception:
        pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep server quiet in console
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_index()
            return
        if self.path == "/events":
            self._serve_sse()
            return
        if self.path == "/health":
            self._serve_health()
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
            self.wfile.write(b"<h1>Person Detection</h1><p>Missing web/index.html</p>")

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        client_q = _register_client_queue()
        # Initial ping to open the stream
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

    def _serve_health(self):
        try:
            payload = {
                "ok": True,
                "subscribers": len(subscribers),
                "server": "pd",
            }
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass


def start_http_server(host="0.0.0.0", port=8000):
    # Prefer dual-stack when binding to IPv6 any (::) so localhost over ::1 and 127.0.0.1 both work
    server = None
    if host in ("::", "[::]"):
        try:
            class DualStackServer(ThreadingHTTPServer):
                address_family = socket.AF_INET6

                def server_bind(self):
                    if hasattr(socket, "has_dualstack_ipv6") and socket.has_dualstack_ipv6():
                        try:
                            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                        except Exception:
                            pass
                    super().server_bind()

            server = DualStackServer(("::", port), Handler)
            print("[http] dual-stack IPv6 server bound on [::]:%d (IPv4 mapped enabled)" % port)
        except Exception as e:
            print(f"[http] IPv6 bind failed ({e}); falling back to IPv4 0.0.0.0")
    if server is None:
        server = ThreadingHTTPServer((host, port), Handler)
        print(f"[http] server bound on {host}:{port}")
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ---------------------------
# MQTT Publish (optional)
# ---------------------------
class MqttPublisher:
    def __init__(self, url, user=None, password=None, topic=None, qos=1, client_id=None, cafile=None, drone_id=None):
        self.url = url
        self.user = user
        self.password = password
        self.topic = topic
        self.qos = int(qos or 1)
        self.client_id = client_id
        self.cafile = cafile
        self.drone_id = drone_id or platform.node() or "drone"
        self._q = Queue(maxsize=256)
        self._connected_evt = threading.Event()
        self._stop_evt = threading.Event()
        self._thread = None
        self._client = None

    def start(self):
        # Create MQTT client (explicit Callback API v1 to avoid warnings on paho-mqtt v2)
        try:
            from paho.mqtt.client import Client, CallbackAPIVersion
            self._client = Client(client_id=self.client_id or f"jil-{self.drone_id}", callback_api_version=CallbackAPIVersion.VERSION2)
        except Exception:
            try:
                import paho.mqtt.client as mqtt
            except Exception as e:
                print(f"[mqtt] paho-mqtt not available: {e}. MQTT disabled.")
                return False
            self._client = mqtt.Client(client_id=self.client_id or f"jil-{self.drone_id}")

        parsed = urlparse(self.url)
        scheme = (parsed.scheme or "mqtt").lower()
        host = parsed.hostname or "localhost"
        port = parsed.port or (8883 if scheme in ("mqtts", "ssl", "tls") else 1883)

        if self.user:
            self._client.username_pw_set(self.user, self.password or None)

        if scheme in ("mqtts", "ssl", "tls"):
            ctx = ssl.create_default_context(cafile=self.cafile) if self.cafile else ssl.create_default_context()
            self._client.tls_set_context(ctx)

        def on_connect(client, userdata, flags, rc, properties=None):
            if rc == 0:
                self._connected_evt.set()
                print("[mqtt] connected")
            else:
                print(f"[mqtt] connect failed rc={rc}")

        def on_disconnect(client, userdata, rc, properties=None):
            self._connected_evt.clear()
            print(f"[mqtt] disconnected rc={rc}")

        self._client.on_connect = on_connect
        self._client.on_disconnect = on_disconnect

        self._client.connect_async(host, port, keepalive=30)
        self._client.loop_start()

        self._thread = threading.Thread(target=self._worker, name="mqtt-publisher", daemon=True)
        self._thread.start()
        return True

    def stop(self):
        try:
            self._stop_evt.set()
            if self._client:
                self._client.loop_stop()
                self._client.disconnect()
        except Exception:
            pass

    def enqueue(self, event):
        # augment with drone_id
        payload = dict(event)
        payload.setdefault("drone_id", self.drone_id)
        try:
            self._q.put_nowait(payload)
        except Exception:
            # drop oldest to make room
            try:
                _ = self._q.get_nowait()
                self._q.put_nowait(payload)
            except Exception:
                pass

    def _worker(self):
        backoff = 0.5
        while not self._stop_evt.is_set():
            try:
                item = self._q.get(timeout=0.25)
            except Exception:
                continue
            try:
                topic = self.topic or f"drone/{self.drone_id}/events"
                data = json.dumps(item, separators=(",", ":"))
                if not self._connected_evt.wait(timeout=5):
                    # not connected; requeue later
                    time.sleep(min(backoff, 5))
                    backoff = min(backoff * 2, 10)
                    self.enqueue(item)
                    continue
                backoff = 0.5
                if self._client:
                    info = self._client.publish(topic, data, qos=self.qos)
                    # Wait briefly for network I/O but do not block the detector
                    info.wait_for_publish(2)
            except Exception as e:
                print(f"[mqtt] publish error: {e}")
                # attempt later
                self.enqueue(item)


# ---------------------------
# Person detection (OpenCV DNN)
# ---------------------------
# Resolve model files relative to this script so it works regardless of CWD
_BASE_DIR = os.path.dirname(__file__)
_PROTOTXT = os.path.join(_BASE_DIR, "MobileNetSSD_deploy.prototxt")
_CAFFEMODEL = os.path.join(_BASE_DIR, "MobileNetSSD_deploy.caffemodel")

net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)


def _setup_depthai_yolo():
    # Enable when DEPTHAI_YOLO_BLOB is provided or CAMERA_SOURCE requests depthai
    blob_path = os.getenv("DEPTHAI_YOLO_BLOB")
    cam_src = os.getenv("CAMERA_SOURCE", "").strip().lower()
    if not blob_path and cam_src not in {"depthai", "oak", "oakd", "oak-d", "oak-d-lite"}:
        return None
    try:
        import depthai as dai
    except Exception as e:
        print(f"[yolo] depthai not available: {e}. Using OpenCV fallback.")
        return None

    if not blob_path or not os.path.isfile(blob_path):
        print("[yolo] DEPTHAI_YOLO_BLOB missing or not a file. Using OpenCV fallback.")
        return None

    class DepthAIYolo:
        def __init__(self):
            print(f"[yolo] DepthAI version: {getattr(dai, '__version__', '?')}")
            pipeline = dai.Pipeline()

            # Camera
            if hasattr(dai, "node") and hasattr(dai.node, "ColorCamera"):
                cam = pipeline.create(dai.node.ColorCamera)
            elif hasattr(pipeline, "createColorCamera"):
                cam = pipeline.createColorCamera()
            else:
                raise RuntimeError("No ColorCamera node available in this DepthAI version")

            try:
                cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            except Exception:
                pass

            # Sensor/video settings
            try:
                if hasattr(dai, "ColorCameraProperties") and hasattr(dai.ColorCameraProperties, "SensorResolution"):
                    res = os.getenv("OAK_RGB_RES", "720").strip().lower()
                    res_map = {
                        "720": dai.ColorCameraProperties.SensorResolution.THE_720_P,
                        "1080": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
                        "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
                    }
                    cam.setResolution(res_map.get(res, dai.ColorCameraProperties.SensorResolution.THE_720_P))
            except Exception:
                pass
            try:
                fps = float(os.getenv("OAK_FPS", os.getenv("FRAME_FPS", "30")))
                if fps > 0 and hasattr(cam, "setFps"):
                    cam.setFps(fps)
            except Exception:
                pass

            # YOLO input size (preview)
            try:
                inp = os.getenv("YOLO_INPUT", "640x640").lower()
                if "x" in inp:
                    w_s, h_s = inp.split("x", 1)
                    pw, ph = int(w_s), int(h_s)
                else:
                    pw = ph = int(inp)
            except Exception:
                pw, ph = 640, 640
            try:
                cam.setPreviewSize(pw, ph)
            except Exception:
                pass

            # Detection network (generic DetectionNetwork in 3.x)
            if hasattr(dai, "node") and hasattr(dai.node, "DetectionNetwork"):
                nn = pipeline.create(dai.node.DetectionNetwork)
            else:
                raise RuntimeError("DetectionNetwork not available in this DepthAI version")

            try:
                nn.setBlobPath(blob_path)
            except Exception as e:
                raise RuntimeError(f"Failed to set blob: {e}")

            # Configure YOLO parsing (must match your blob)
            def _float_list_from_env(key):
                raw = os.getenv(key)
                if not raw:
                    return None
                try:
                    return [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
                except Exception:
                    return None

            try:
                nn.setConfidenceThreshold(float(os.getenv("YOLO_CONF", "0.5")))
            except Exception:
                pass
            try:
                nn.setIouThreshold(float(os.getenv("YOLO_IOU", "0.4")))
            except Exception:
                pass
            try:
                nn.setNumClasses(int(os.getenv("YOLO_NUM_CLASSES", "80")))
            except Exception:
                pass
            try:
                nn.setCoordinateSize(int(os.getenv("YOLO_COORD_SIZE", "4")))
            except Exception:
                pass
            anchors = _float_list_from_env("YOLO_ANCHORS")
            if anchors:
                try:
                    nn.setAnchors(anchors)
                except Exception:
                    pass
            # Anchor masks as JSON-like e.g. {"side52": [0,1,2], "side26": [3,4,5]}
            try:
                masks_env = os.getenv("YOLO_ANCHOR_MASKS")
                if masks_env:
                    masks = json.loads(masks_env)
                    nn.setAnchorMasks({str(k): list(map(int, v)) for k, v in masks.items()})
            except Exception:
                pass

            # Link camera to NN
            cam.preview.link(nn.input)

            # Create output queues (no XLinkOut in 3.x)
            self._pipeline = pipeline
            self._device = None
            try:
                # Using explicit Device is compatible and lets us close cleanly
                self._device = dai.Device(pipeline)
            except Exception:
                # Fallback to implicit device via pipeline.start()
                try:
                    pipeline.start()
                except Exception:
                    pass

            self._frame_q = None
            self._det_q = None
            try:
                self._frame_q = cam.preview.createOutputQueue(maxSize=4, blocking=False)
                self._det_q = nn.out.createOutputQueue(maxSize=4, blocking=False)
            except Exception as e:
                # If queues aren't available, close device and surface error
                try:
                    if self._device:
                        self._device.close()
                except Exception:
                    pass
                raise

            self._last_dets = []

        def read(self):
            # Block for a frame; fetch latest detections if available
            try:
                fmsg = self._frame_q.get()
                frame = fmsg.getCvFrame()
            except Exception:
                return False, None, []

            try:
                dmsg = self._det_q.tryGet()
                if dmsg is not None:
                    dets = []
                    for d in getattr(dmsg, 'detections', []):
                        dets.append({
                            "label": int(getattr(d, 'label', -1)),
                            "confidence": float(getattr(d, 'confidence', 0.0)),
                            "xmin": float(getattr(d, 'xmin', 0.0)),
                            "ymin": float(getattr(d, 'ymin', 0.0)),
                            "xmax": float(getattr(d, 'xmax', 0.0)),
                            "ymax": float(getattr(d, 'ymax', 0.0)),
                        })
                    self._last_dets = dets
            except Exception:
                pass
            return True, frame, list(self._last_dets)

        def release(self):
            try:
                if self._frame_q:
                    self._frame_q.close()
                if self._det_q:
                    self._det_q.close()
            except Exception:
                pass
            try:
                if self._device:
                    self._device.close()
            except Exception:
                pass

    try:
        return DepthAIYolo()
    except Exception as e:
        print(f"[yolo] init failed: {e}. Using OpenCV fallback.")
        return None


def _open_capture_from_env():
    # CAMERA_SOURCE can be an integer index (e.g., "0"), a device path ("/dev/video0"),
    # a file/stream URL, or a GStreamer pipeline string (for libcamera on Raspberry Pi).
    src = os.getenv("CAMERA_SOURCE", "0").strip().lower()
    cap = None

    # DepthAI (OAK-D / OAK-D Lite) source
    if src in {"depthai", "oak", "oakd", "oak-d", "oak-d-lite"}:
        try:
            import depthai as dai

            class DepthAICapture:
                def __init__(self):
                    print(f"[camera] DepthAI version: {getattr(dai, '__version__', '?')}")
                    pipeline = dai.Pipeline()

                    # Prefer newer generic Camera node; fallback to ColorCamera
                    if hasattr(dai, "node") and hasattr(dai.node, "Camera"):
                        cam = pipeline.create(dai.node.Camera)
                        print("[camera] DepthAI: using dai.node.Camera")
                        # Try to select RGB socket when available
                        try:
                            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
                        except Exception:
                            pass
                    elif hasattr(dai, "node") and hasattr(dai.node, "ColorCamera"):
                        cam = pipeline.create(dai.node.ColorCamera)
                        print("[camera] DepthAI: using dai.node.ColorCamera")
                    elif hasattr(pipeline, "createColorCamera"):
                        cam = pipeline.createColorCamera()
                        print("[camera] DepthAI: using legacy createColorCamera()")
                    else:
                        raise RuntimeError("No Camera/ColorCamera node available in this DepthAI version")

                    # Map env to sensor/video settings
                    res = os.getenv("OAK_RGB_RES", "1080").strip().lower()
                    try:
                        if hasattr(dai, "ColorCameraProperties") and hasattr(dai.ColorCameraProperties, "SensorResolution") and hasattr(cam, "setResolution"):
                            res_map = {
                                "720": dai.ColorCameraProperties.SensorResolution.THE_720_P,
                                "1080": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
                                "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
                            }
                            cam.setResolution(res_map.get(res, dai.ColorCameraProperties.SensorResolution.THE_1080_P))
                    except Exception:
                        pass
                    try:
                        fps = int(float(os.getenv("OAK_FPS", os.getenv("FRAME_FPS", "30"))))
                        if fps > 0 and hasattr(cam, "setFps"):
                            cam.setFps(fps)
                    except Exception:
                        pass

                    # Output size to host
                    try:
                        vw = int(os.getenv("FRAME_WIDTH", "640"))
                        vh = int(os.getenv("FRAME_HEIGHT", "480"))
                    except Exception:
                        vw, vh = 640, 480
                    try:
                        cam.setVideoSize(vw, vh)
                    except Exception:
                        pass

                    # Create host output queue
                    # Prefer XLinkOut (DepthAI 2.x); otherwise use DepthAI 3.x per-output queues
                    xout = None
                    if hasattr(dai, "node") and hasattr(dai.node, "XLinkOut"):
                        xout = pipeline.create(dai.node.XLinkOut)
                        print("[camera] DepthAI: using dai.node.XLinkOut")
                    elif hasattr(dai, "XLinkOut"):
                        try:
                            xout = pipeline.create(dai.XLinkOut)
                            print("[camera] DepthAI: using dai.XLinkOut (top-level)")
                        except Exception:
                            xout = None
                    if xout is None and hasattr(pipeline, "createXLinkOut"):
                        xout = pipeline.createXLinkOut()
                        print("[camera] DepthAI: using legacy createXLinkOut()")

                    if xout is not None:
                        # DepthAI 2.x path: link camera output into XLinkOut and open host queue by name
                        xout.setStreamName("video")
                        try:
                            cam.video.link(xout.input)
                        except Exception:
                            # Some variants expose 'preview' or 'isp'
                            linked = False
                            for outlet in ("video", "preview", "isp"):
                                try:
                                    getattr(cam, outlet).link(xout.input)
                                    print(f"[camera] DepthAI: linked via cam.{outlet}")
                                    linked = True
                                    break
                                except Exception:
                                    continue
                            if not linked:
                                raise RuntimeError("Failed to link camera output to XLinkOut")

                        self._device = dai.Device(pipeline)
                        self._q = self._device.getOutputQueue(name="video", maxSize=4, blocking=False)
                    else:
                        # DepthAI 3.x path: create output queue directly from node output
                        print("[camera] DepthAI: XLinkOut not available; using Node.Output.createOutputQueue()")
                        # Ensure pipeline is started (uses implicit default device)
                        try:
                            if hasattr(pipeline, "start"):
                                pipeline.start()
                        except Exception:
                            pass
                        # Pick first available outlet in preferred order
                        out_port = None
                        for outlet in ("video", "preview", "isp"):
                            try:
                                out_port = getattr(cam, outlet)
                                if out_port is not None:
                                    print(f"[camera] DepthAI: streaming from cam.{outlet}")
                                    break
                            except Exception:
                                continue
                        if out_port is None:
                            raise RuntimeError("No usable camera output (video/preview/isp) found")
                        self._q = out_port.createOutputQueue(maxSize=4, blocking=False)
                        # Hold a reference to the default device so we can close it on release
                        try:
                            self._device = pipeline.getDefaultDevice() if hasattr(pipeline, "getDefaultDevice") else None
                        except Exception:
                            self._device = None

                def read(self):
                    msg = self._q.tryGet()
                    if msg is None:
                        return False, None
                    frame = msg.getCvFrame()
                    return True, frame

                def release(self):
                    try:
                        self._device.close()
                    except Exception:
                        pass

            return DepthAICapture()
        except Exception as e:
            print(f"[camera] DepthAI init failed: {e}. Falling back to OpenCV VideoCapture(0).")
            src = "0"
    try:
        if src.isdigit():
            cap = cv2.VideoCapture(int(src))
        else:
            # For GStreamer pipelines, pass CAP_GSTREAMER
            if "!" in src:
                cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
            else:
                cap = cv2.VideoCapture(src)
    except Exception:
        cap = cv2.VideoCapture(0)

    # Optional capture properties
    try:
        w = int(os.getenv("FRAME_WIDTH", "0"))
        h = int(os.getenv("FRAME_HEIGHT", "0"))
        fps = float(os.getenv("FRAME_FPS", "0"))
        if w > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps > 0:
            cap.set(cv2.CAP_PROP_FPS, fps)
    except Exception:
        pass

    # If the capture failed to open, auto-fallback to device 0
    try:
        if not cap or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            print("[camera] Primary source failed to open; falling back to VideoCapture(0)")
            cap = cv2.VideoCapture(0)
    except Exception:
        pass

    return cap


cap = _open_capture_from_env()

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def main():
    # Start SSE server for the front-end
    http_host = os.getenv("HTTP_HOST", "0.0.0.0")
    http_port = int(os.getenv("HTTP_PORT", "8000"))
    start_http_server(http_host, http_port)
    try:
        host_hint = "127.0.0.1" if http_host in ("0.0.0.0", "::", "[::]", "0") else http_host
        print(f"SSE server running — try http://{host_hint}:{http_port} (or http://localhost:{http_port}). Health: /health")
    except Exception:
        print(f"SSE server running on {http_host}:{http_port}")

    last_alert_ts = 0.0
    alert_cooldown = 1.0  # seconds between alerts to the frontend

    show_window = os.getenv("SHOW_WINDOW", "0") == "1"

    # Prefer DepthAI YOLO if configured
    yolo = _setup_depthai_yolo()

    while True:
        person_found = False
        top_conf = 0.0

        if yolo is not None:
            ok, frame, dets = yolo.read()
            if not ok:
                time.sleep(0.02)
                continue
            h, w = frame.shape[:2]
            person_label = int(os.getenv("PERSON_LABEL_ID", "0"))
            thresh = float(os.getenv("PERSON_CONF_THRESH", os.getenv("YOLO_CONF", "0.5")))
            for d in dets:
                lbl = d.get("label", -1)
                conf = float(d.get("confidence", 0.0))
                if lbl == person_label and conf >= thresh:
                    person_found = True
                    top_conf = max(top_conf, conf)
                # draw all detections lightly; highlight person
                x1 = int(d.get("xmin", 0.0) * w)
                y1 = int(d.get("ymin", 0.0) * h)
                x2 = int(d.get("xmax", 0.0) * w)
                y2 = int(d.get("ymax", 0.0) * h)
                color = (0, 255, 0) if lbl == person_label else (50, 150, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{lbl}:{conf:.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
        else:
            ret, frame = cap.read()
            if not ret:
                # If camera temporarily fails, wait and retry
                time.sleep(0.1)
                continue
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                cls = int(detections[0, 0, i, 1])
                if conf > 0.4 and CLASSES[cls] == "person":
                    person_found = True
                    top_conf = max(top_conf, conf)
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"person {conf:.2f}",
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

        # Throttle alert spam; only send when a person is present
        now = time.time()
        if person_found and (now - last_alert_ts) >= alert_cooldown:
            last_alert_ts = now
            broadcast_event(
                {
                    "type": "person_detected",
                    "confidence": round(top_conf, 3),
                    "ts": int(now),
                }
            )

        if show_window:
            cv2.imshow("person-detect", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    try:
        if yolo is not None:
            yolo.release()
    except Exception:
        pass
    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
