Raspberry Pi Setup and Run Guide

This guide helps you run the person detection server on a Raspberry Pi (local web UI over HTTP + optional MQTT telemetry). It supports USB webcams out of the box and Pi Camera modules via a GStreamer/libcamera pipeline.

Requirements

- Raspberry Pi OS Bookworm (64‑bit recommended)
- A camera: USB webcam or Raspberry Pi Camera Module (via libcamera)
- Network connectivity on the same LAN as your viewer

Install Dependencies

1) Update the system

    sudo apt update && sudo apt -y upgrade

2) Install Python + OpenCV (use apt on Pi) and helpers

    sudo apt install -y python3 python3-pip python3-opencv
    # Optional: for libcamera via GStreamer pipelines
    sudo apt install -y gstreamer1.0-tools gstreamer1.0-libcamera gstreamer1.0-plugins-good

3) Install DepthAI (OAK‑D Lite) and MQTT client library for Python

    pip3 install --user depthai paho-mqtt

Get the Code

    cd ~
    git clone https://example.com/your-fork-or-copy.git  # or copy the folder over SCP
    cd SeniorDesignFloomba

Configure Camera

- USB webcam: no changes needed. Default `CAMERA_SOURCE=0` opens `/dev/video0`.
- OAK‑D Lite (DepthAI):

    export CAMERA_SOURCE=depthai
    # Optional tuning
    export OAK_RGB_RES=1080   # 720 | 1080 | 4k
    export OAK_FPS=30         # frame rate hint

- Pi Camera (libcamera): use a GStreamer pipeline. Example at 640x480@30fps:

    export CAMERA_SOURCE='libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink'

You can also set optional capture hints:

    export FRAME_WIDTH=640
    export FRAME_HEIGHT=480
    export FRAME_FPS=30

Configure Network/UI

- The built‑in HTTP server listens on all interfaces by default.
- Customize with:

    export HTTP_HOST=0.0.0.0
    export HTTP_PORT=8000

Run Headless (no GUI window)

By default on the Pi we recommend headless mode (no X/Wayland window). The web UI at `/web/index.html` consumes events via `/events`.

    export SHOW_WINDOW=0

MQTT (Optional)

Set these to publish detections to a broker. If unset, MQTT is disabled.

    export DRONE_ID=$(hostname)                               # or a fixed name
    export MQTT_URL='mqtt://broker.local:1883'                 # or mqtts://host:8883
    export MQTT_USER='your-user'                               # optional
    export MQTT_PASS='your-pass'                               # optional
    export MQTT_TOPIC="drone/${DRONE_ID}/events"              # optional (auto‑defaults)
    export MQTT_QOS=1                                          # optional
    export MQTT_CAFILE='/etc/ssl/certs/ca-certificates.crt'    # for TLS, optional

Run It

    cd ~/SeniorDesignFloomba
    python3 PD_main.py

Open a browser to:

- http://<pi-ip>:8000

The page will connect to `/events` and show detection alerts. Use the filter to view a specific `drone_id` when multiple sources report to the same UI.

Autostart with systemd (Optional)

1) Edit the provided service template to fit your paths/user:

    nano systemd/pd.service

2) Copy it into systemd and enable:

    sudo cp systemd/pd.service /etc/systemd/system/pd.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now pd.service

3) Check status and logs:

    systemctl status pd.service
    journalctl -u pd.service -f

Pi Camera Notes

- If using the CSI camera module, prefer the GStreamer pipeline shown above.
- Adjust resolution/framerate to match your Pi model and lighting conditions for best performance.

Troubleshooting

- No camera frames: try `v4l2-ctl --list-devices` (USB) or test `libcamera-hello` (CSI camera). For CSI, use the GStreamer `libcamerasrc` pipeline.
- OpenCV not found in Python: ensure `python3-opencv` is installed via apt, not pip, on Raspberry Pi OS.
- UI not loading: confirm the Pi’s IP, that `HTTP_PORT` matches, and that no firewall blocks port 8000.
- MQTT not publishing: verify `paho-mqtt` is installed and `MQTT_URL` is set correctly. For TLS, confirm `MQTT_CAFILE`.
