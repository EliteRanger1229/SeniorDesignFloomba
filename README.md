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
- `Dockerfile.relay` — container for the relay.
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

1) Dependencies

- Python 3.9+
- OpenCV (already used here) and paho-mqtt for telemetry:

```bash
pip install paho-mqtt
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
python3 PD_main.py
```

- Local dashboard: `http://<drone-ip>:8000` → connects to `/events`.
- MQTT: publishes events if `MQTT_URL` is set. Resilient to drops.

## Running the Public Relay (VM or Container)

Option A — Python directly:

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

Option B — Docker:

```bash
docker build -f Dockerfile.relay -t mqtt-relay .
docker run --rm -p 8080:8080 \
  -e MQTT_URL='mqtts://broker.example.com:8883' \
  -e MQTT_USER='your-user' -e MQTT_PASS='your-pass' \
  -e MQTT_CAFILE='/etc/ssl/certs/ca-certificates.crt' \
  mqtt-relay
```

Supply a CA file inside the image or mount it as needed.

Option C — Docker Compose (recommended)

1) Copy and edit the sample env

```bash
cp .env.example .env
# Edit .env and set MQTT_URL, MQTT_USER, MQTT_PASS, RELAY_HOST_PORT
# If using a custom CA, set MQTT_CAFILE_HOST and uncomment the volume + MQTT_CAFILE in docker-compose.yml
```

2) Bring up the relay

```bash
docker compose up -d --build
```

3) Open the UI

```
http://localhost:${RELAY_HOST_PORT}
```

Notes:
- If port 8080 is busy, set `RELAY_HOST_PORT` in `.env` (e.g., 8081).
- For a private CA, set `MQTT_CAFILE_HOST` in `.env`, uncomment the volume in `docker-compose.yml`, and set `MQTT_CAFILE=/ca.crt` in the service environment.

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
