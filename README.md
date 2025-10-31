# Drone Person Detection: MQTT Telemetry + Web Relay

This repo runs a lightweight person detector (OpenCV DNN + MobileNet SSD) and streams events two ways:

- Local dashboard via Server-Sent Events (SSE) at `http://<host>:8000`.
- Outbound MQTT telemetry for reliable, low‑bandwidth reporting from a drone.

A tiny public relay (`mqtt_relay.py`) subscribes to MQTT and serves the same SSE UI so you can view detections remotely without exposing the drone.

## Topology

Drone (PD_main.py) → MQTT broker ← Relay (mqtt_relay.py) → Browser (`/events`)

## Files

- `PD_main.py` — detector + local SSE server. Publishes MQTT if configured.
- `mqtt_relay.py` — subscribes to MQTT and rebroadcasts as SSE + serves UI.
- `web/index.html` — dashboard (now supports filtering by `drone_id`).
- `MobileNetSSD_deploy.prototxt`, `MobileNetSSD_deploy.caffemodel` — DNN model.

## MQTT Topics & Payload

- Publish (default): `drone/<DRONE_ID>/events`
- Relay subscribe (default): `drone/+/events`
- Payload example:

```json
{
  "type": "person_detected",
  "confidence": 0.87,
  "ts": 1710000000,
  "drone_id": "my-drone-01"
}
```

## Running on the Drone (Raspberry Pi)

For a Pi-focused, step-by-step guide (including libcamera and systemd), see `RaspberryPi.md`.

1) Dependencies

- Python 3.9+
- OpenCV (already used here), DepthAI for OAK‑D Lite, and paho‑mqtt for telemetry:

```bash
pip install paho-mqtt
# OAK-D Lite support (DepthAI)
pip install depthai
# If needed for dev machines: pip install opencv-python
```

2) Environment

```bash
export DRONE_ID=my-drone-01
export MQTT_URL='mqtts://broker.example.com:8883'  # or mqtt://host:1883
export MQTT_USER='your-user'                       # optional
export MQTT_PASS='your-pass'                       # optional
export MQTT_TOPIC='drone/my-drone-01/events'       # optional (defaults based on DRONE_ID)
export MQTT_QOS=1                                  # optional
export MQTT_CAFILE='/path/to/ca.crt'               # optional for TLS
```

3) Run

```bash
export CAMERA_SOURCE=depthai   # for OAK-D Lite (optional)
python3 PD_main.py
```

- Local dashboard: `http://<drone-ip>:8000` → connects to `/events`.
- MQTT: publishes events if `MQTT_URL` is set. Resilient to drops.

## Running the Public Relay (Python)

Run the relay as a simple Python process:

```bash
pip install paho-mqtt
export MQTT_URL='mqtts://broker.example.com:8883'
export MQTT_USER='your-user'
export MQTT_PASS='your-pass'
export MQTT_CAFILE='/path/to/ca.crt'
export MQTT_SUB_TOPIC='drone/+/events'   # optional
export HTTP_PORT=8080                    # optional
python3 mqtt_relay.py
```

Open `http://<relay-host>:8080`.

## Broker Notes

- Use TLS (`mqtts://…:8883`) where possible.
- Create per‑drone credentials and ACLs: publish‑only to `drone/<id>/events`.
- Managed brokers: EMQX Cloud, HiveMQ Cloud; or self‑host EMQX/Mosquitto.

## UI Filtering

The dashboard shows `drone_id` in the log and lets you filter the live counters by a selected drone. Changing the filter resets the per‑view counters but not the log.

## Alternatives

- If you only need ad‑hoc remote access to the local dashboard, use a tunnel (Cloudflare Tunnel, ngrok, or Tailscale Funnel). MQTT remains the most reliable approach on mobile/roaming links.

## Troubleshooting

- No MQTT from the drone: ensure `paho-mqtt` is installed and `MQTT_URL` is set.
- TLS errors: set `MQTT_CAFILE` to a valid CA bundle.
- No detections: verify camera works and MobileNet SSD files are present.
