#!/usr/bin/env python3

"""
OAK-D-Lite Spatial Object Detection Example

This script demonstrates how to run a pre-trained object detection
model on an OAK-D-Lite to get both the 2D bounding boxes and the
3D spatial coordinates (X, Y, Z) of detected objects.

Required libraries:
- depthai
- opencv-python
- blobconverter (will be installed by depthai if not present)

Press 'q' to exit the application.
"""

import cv2
import depthai as dai
import blobconverter
import time
import os

# --- Constants ---

# Model name to download and use
# This is a pre-trained MobileNet-SSD model that detects 20 common objects
# (e.g., person, car, bird, bottle, chair) and is optimized for spatial detection.
# Renaming to 'mobilenet-ssd' to fix the 404 blobconverter error.
# The 'SpatialDetectionNetwork' node will use this model and the depth
# feed to calculate spatial coordinates.
MODEL_NAME = "mobilenet-ssd"

# NN input size
NN_WIDTH = 300
NN_HEIGHT = 300

# --- Helper Functions ---

def create_xlinkout(pipeline):
    """Create an XLinkOut node compatible with different DepthAI versions."""
    # Prefer modern API if available
    if hasattr(dai, 'node') and hasattr(dai.node, 'XLinkOut'):
        return pipeline.create(dai.node.XLinkOut)
    # Fallback to older class location
    if hasattr(dai, 'XLinkOut'):
        return pipeline.create(dai.XLinkOut)
    # Last resort: very old helper on Pipeline
    if hasattr(pipeline, 'createXLinkOut'):
        return pipeline.createXLinkOut()
    raise AttributeError("DepthAI XLinkOut node is unavailable in this SDK build.")

VERBOSE = str(os.getenv("OD_VERBOSE", "0")).lower() not in ("", "0", "false", "no")
DIAG_ONLY = str(os.getenv("OD_DIAG", "0")).lower() not in ("", "0", "false", "no")


def open_device_with_retry(pipeline, retries: int = 5, delay: float = 2.5):
    """Open a DepthAI device and start the pipeline with retries.

    Uses Device()+startPipeline(pipeline) to be compatible across SDKs,
    and retries if the device is temporarily busy or re-enumerating.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        if VERBOSE:
            print(f"[od] Opening device (attempt {attempt}/{retries})...")
        try:
            # Try non-exclusive config if available (helps diagnostics on 3.x)
            device = None
            try:
                cfg_cls = getattr(dai.Device, 'Config', None)
                if cfg_cls is not None:
                    cfg = cfg_cls()
                    if hasattr(cfg, 'setNonExclusiveMode'):
                        cfg.setNonExclusiveMode(True)
                    elif hasattr(cfg, 'nonExclusiveMode'):
                        cfg.nonExclusiveMode = True
                    device = dai.Device(cfg)
            except Exception:
                device = None
            if device is None:
                device = dai.Device()
            device.startPipeline(pipeline)
            if VERBOSE:
                try:
                    infos = dai.Device.getAllAvailableDevices()
                    if infos:
                        print(f"[od] Available devices: {[getattr(i, 'getMxId', lambda: None)() or getattr(i, 'mxid', None) for i in infos]}")
                except Exception:
                    pass
            return device
        except Exception as e:
            last_err = e
            msg = str(e)
            if "ALREADY_IN_USE" in msg or "already in use" in msg:
                print(f"[od] Device busy; retrying in {delay}s ({attempt}/{retries})...")
                time.sleep(delay)
                continue
            print(f"[od] Device open failed; retrying in {delay}s ({attempt}/{retries})...\n  {e}")
            time.sleep(delay)
    # Exhausted retries
    raise last_err

def get_board_socket(primary: bool = False, left: bool = False, right: bool = False):
    """Return best-available board socket enum with fallbacks."""
    # Prefer CAM_A/B/C if available, else legacy RGB/LEFT/RIGHT
    if hasattr(dai, 'CameraBoardSocket'):
        cbs = dai.CameraBoardSocket
        if primary and hasattr(cbs, 'CAM_A'):
            return cbs.CAM_A
        if left and hasattr(cbs, 'CAM_B'):
            return cbs.CAM_B
        if right and hasattr(cbs, 'CAM_C'):
            return cbs.CAM_C
        # Fallbacks for older SDKs
        if primary and hasattr(cbs, 'RGB'):
            return cbs.RGB
        if left and hasattr(cbs, 'LEFT'):
            return cbs.LEFT
        if right and hasattr(cbs, 'RIGHT'):
            return cbs.RIGHT
    # Final fallback: return None and let caller handle
    return None

def pick_rgb_output_port(cam_node):
    """Pick the best available output port from a camera node for NN/display."""
    # Prefer preview (already sized/scaled), then video, then isp, then out
    if hasattr(cam_node, 'preview'):
        return cam_node.preview
    if hasattr(cam_node, 'video'):
        return cam_node.video
    if hasattr(cam_node, 'isp'):
        return cam_node.isp
    if hasattr(cam_node, 'out'):
        return cam_node.out
    raise AttributeError("No usable output port found on camera node")

def create_pipeline():
    """
    Configures and returns the DepthAI pipeline for spatial object detection.
    """
    print("Creating DepthAI pipeline...")
    
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # --- Core Nodes ---

    # 1. Color/Unified Camera Node (RGB)
    # Prefer unified Camera, fallback to ColorCamera
    if hasattr(dai, 'node') and hasattr(dai.node, 'Camera'):
        cam_rgb = pipeline.create(dai.node.Camera)
        sock = get_board_socket(primary=True)
        if sock is not None and hasattr(cam_rgb, 'setBoardSocket'):
            cam_rgb.setBoardSocket(sock)
        # Configure preview size when supported
        if hasattr(cam_rgb, 'setPreviewSize'):
            cam_rgb.setPreviewSize(NN_WIDTH, NN_HEIGHT)
        # Aim for non-interleaved when option exists
        if hasattr(cam_rgb, 'setInterleaved'):
            cam_rgb.setInterleaved(False)
    else:
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        sock = get_board_socket(primary=True)
        if sock is not None:
            cam_rgb.setBoardSocket(sock)
        cam_rgb.setPreviewSize(NN_WIDTH, NN_HEIGHT)
        if hasattr(dai, 'ColorCameraProperties') and hasattr(dai.ColorCameraProperties, 'SensorResolution'):
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        if hasattr(cam_rgb, 'setInterleaved'):
            cam_rgb.setInterleaved(False)
    # Choose which camera output to use for NN/display; fallback to legacy ColorCamera if needed
    try:
        rgb_out_port = pick_rgb_output_port(cam_rgb)
    except AttributeError:
        # Fallback to legacy ColorCamera which exposes 'preview'
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        sock = get_board_socket(primary=True)
        if sock is not None:
            cam_rgb.setBoardSocket(sock)
        cam_rgb.setPreviewSize(NN_WIDTH, NN_HEIGHT)
        if hasattr(dai, 'ColorCameraProperties') and hasattr(dai.ColorCameraProperties, 'SensorResolution'):
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        if hasattr(cam_rgb, 'setInterleaved'):
            cam_rgb.setInterleaved(False)
        rgb_out_port = pick_rgb_output_port(cam_rgb)

    # 2. Left Camera Node (mono for depth)
    if hasattr(dai, 'node') and hasattr(dai.node, 'Camera'):
        mono_left = pipeline.create(dai.node.Camera)
        sock_l = get_board_socket(left=True)
        if sock_l is not None and hasattr(mono_left, 'setBoardSocket'):
            mono_left.setBoardSocket(sock_l)
    else:
        mono_left = pipeline.create(dai.node.MonoCamera)
        sock_l = get_board_socket(left=True)
        if sock_l is not None:
            mono_left.setBoardSocket(sock_l)
        if hasattr(dai, 'MonoCameraProperties') and hasattr(dai.MonoCameraProperties, 'SensorResolution'):
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    # Determine left output port; fallback to MonoCamera if unified Camera lacks 'out'
    left_out_port = getattr(mono_left, 'out', None)
    if left_out_port is None:
        legacy_left = pipeline.create(dai.node.MonoCamera)
        sock_l = get_board_socket(left=True)
        if sock_l is not None:
            legacy_left.setBoardSocket(sock_l)
        if hasattr(dai, 'MonoCameraProperties') and hasattr(dai.MonoCameraProperties, 'SensorResolution'):
            legacy_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left = legacy_left
        left_out_port = mono_left.out

    # 3. Right Camera Node (mono for depth)
    if hasattr(dai, 'node') and hasattr(dai.node, 'Camera'):
        mono_right = pipeline.create(dai.node.Camera)
        sock_r = get_board_socket(right=True)
        if sock_r is not None and hasattr(mono_right, 'setBoardSocket'):
            mono_right.setBoardSocket(sock_r)
    else:
        mono_right = pipeline.create(dai.node.MonoCamera)
        sock_r = get_board_socket(right=True)
        if sock_r is not None:
            mono_right.setBoardSocket(sock_r)
        if hasattr(dai, 'MonoCameraProperties') and hasattr(dai.MonoCameraProperties, 'SensorResolution'):
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    # Determine right output port; fallback to MonoCamera if unified Camera lacks 'out'
    right_out_port = getattr(mono_right, 'out', None)
    if right_out_port is None:
        legacy_right = pipeline.create(dai.node.MonoCamera)
        sock_r = get_board_socket(right=True)
        if sock_r is not None:
            legacy_right.setBoardSocket(sock_r)
        if hasattr(dai, 'MonoCameraProperties') and hasattr(dai.MonoCameraProperties, 'SensorResolution'):
            legacy_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right = legacy_right
        right_out_port = mono_right.out

    # 4. Stereo Depth Node
    # This node calculates the depth map from the mono camera feeds.
    stereo = pipeline.create(dai.node.StereoDepth)
    
    # Configure stereo depth (replacing deprecated ProfilePreset)
    stereo.initialConfig.setConfidenceThreshold(245)
    # Use 7x7 median filter for better (but slower) depth accuracy
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    
    # Align depth map to the perspective of the RGB camera
    align_sock = get_board_socket(primary=True)
    if align_sock is not None:
        stereo.setDepthAlign(align_sock)
    stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
    
    # 5. Spatial Detection (Neural Network) Node
    # This node runs the object detection model and calculates spatial coordinates.
    # Renaming to SpatialDetectionNetwork to fix AttributeError
    nn = pipeline.create(dai.node.SpatialDetectionNetwork)
    
    # Download and set the model blob file
    try:
        # Removing zoo_type="depthai" to try the default model zoo,
        # as the "depthai" zoo alias seems to be causing a 404/400 error.
        blob_path = blobconverter.from_zoo(name=MODEL_NAME, shaves=6)
        nn.setBlobPath(blob_path)
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please ensure you have an internet connection and 'blobconverter' is installed.")
        raise

    nn.setConfidenceThreshold(0.5)
    nn.input.setBlocking(False)
    nn.setBoundingBoxScaleFactor(0.5)
    nn.setDepthLowerThreshold(100)  # Min distance in mm
    nn.setDepthUpperThreshold(5000) # Max distance in mm

    # --- Linking Nodes ---

    print("Linking pipeline nodes...")

    # Connect camera outputs to stereo depth node
    left_out_port.link(stereo.left)
    right_out_port.link(stereo.right)

    # Connect RGB camera preview to NN input
    rgb_out_port.link(nn.input)

    # Connect stereo depth output to NN depth input
    stereo.depth.link(nn.inputDepth)

    # --- Output / IO Setup ---
    io = {"use_xlink": False, "cam_preview": None, "nn_out": None}

    try:
        # Prefer XLinkOut if available (DepthAI 2.x)
        xout_rgb = create_xlinkout(pipeline)
        xout_rgb.setStreamName("rgb")
        rgb_out_port.link(xout_rgb.input)

        xout_nn = create_xlinkout(pipeline)
        xout_nn.setStreamName("detections")
        nn.out.link(xout_nn.input)

        io["use_xlink"] = True
        print("Using XLinkOut streams: rgb, detections")
    except AttributeError:
        # DepthAI 3.x: Use per-output queues directly from node outputs
        io["use_xlink"] = False
        io["cam_preview"] = rgb_out_port
        io["nn_out"] = nn.out
        print("XLinkOut not available; using node output queues")

    print("Pipeline created successfully.")
    return pipeline, io

def draw_detections(frame, detections):
    """
    Draws bounding boxes and spatial information on the frame.
    """
    height, width, _ = frame.shape
    
    # The 'mobilenet-ssd' model's label map
    label_map = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    for det in detections:
        # --- Draw Bounding Box ---
        
        # Bounding box coordinates are normalized (0.0 to 1.0)
        # We scale them to the frame's dimensions
        x1 = int(det.xmin * width)
        y1 = int(det.ymin * height)
        x2 = int(det.xmax * width)
        y2 = int(det.ymax * height)
        
        # Get label and confidence
        try:
            label = label_map[det.label]
        except IndexError:
            label = f"Label {det.label}"
        
        confidence = f"{int(det.confidence * 100)}%"
        
        # Draw the rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # --- Draw Label and Confidence ---
        label_text = f"{label}: {confidence}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # --- Draw Spatial Coordinates ---
        # spatialCoordinates has .x, .y, .z fields in millimeters
        sp_x = f"X: {int(det.spatialCoordinates.x)} mm"
        sp_y = f"Y: {int(det.spatialCoordinates.y)} mm"
        sp_z = f"Z: {int(det.spatialCoordinates.z)} mm"
        
        cv2.putText(frame, sp_x, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, sp_y, (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, sp_z, (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


# --- Main Application ---

def main():
    pipeline, io = create_pipeline()

    # Diagnostics-only mode: probe device without starting the app loop
    if DIAG_ONLY:
        print("[od] Diagnostics mode enabled (OD_DIAG=1)")
        try:
            infos = dai.Device.getAllAvailableDevices()
            print(f"[od] Enumerated devices: {[getattr(i,'getMxId',lambda:None)() or getattr(i,'mxid',None) for i in infos]}")
        except Exception as e:
            print(f"[od] Enumerate devices failed: {e}")
        try:
            cfg_cls = getattr(dai.Device, 'Config', None)
            dev = None
            if cfg_cls is not None:
                cfg = cfg_cls()
                if hasattr(cfg, 'setNonExclusiveMode'):
                    cfg.setNonExclusiveMode(True)
                elif hasattr(cfg, 'nonExclusiveMode'):
                    cfg.nonExclusiveMode = True
                dev = dai.Device(cfg)
            else:
                dev = dai.Device()
            with dev:
                try:
                    print(f"[od] Device opened. USB: {getattr(dev.getUsbSpeed(),'name',dev.getUsbSpeed())}")
                except Exception:
                    pass
                try:
                    cams = getattr(dev, 'getConnectedCameras', lambda: [])()
                    print(f"[od] Connected cameras: {[getattr(c,'name',str(c)) for c in cams]}")
                except Exception:
                    pass
                print("[od] Diagnostics complete.")
        except Exception as e:
            print(f"[od] Device open failed in diagnostics: {e}")
        print("Application closed.")
        return

    device = None
    try:
        device = open_device_with_retry(pipeline)
        print("Connected to OAK-D-Lite. Pipeline started.")
        if VERBOSE:
            try:
                info = device.getDeviceInfo()
                print(f"[od] Device info: {info}")
            except Exception:
                pass
            try:
                print(f"[od] USB speed: {getattr(device.getUsbSpeed(), 'name', device.getUsbSpeed())}")
            except Exception:
                pass
            try:
                cams = getattr(device, 'getConnectedCameras', lambda: [])()
                cams_str = [getattr(c, 'name', str(c)) for c in cams]
                print(f"[od] Connected cameras: {cams_str}")
            except Exception:
                pass
            try:
                print(f"[od] SDK version: {getattr(dai, '__version__', 'unknown')}")
            except Exception:
                pass

        # Create output queues after pipeline start
        if io.get("use_xlink", False):
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            q_detections = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        else:
            q_rgb = io["cam_preview"].createOutputQueue(maxSize=4, blocking=False)
            q_detections = io["nn_out"].createOutputQueue(maxSize=4, blocking=False)

        print("Streaming... Press 'q' to quit.")

        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            in_detections = q_detections.get()
            detections = getattr(in_detections, 'detections', [])
            if detections:
                draw_detections(frame, detections)
            cv2.imshow("OAK-D-Lite Spatial Detection", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        try:
            if device is not None:
                device.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Application closed.")

    print("Application closed.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
