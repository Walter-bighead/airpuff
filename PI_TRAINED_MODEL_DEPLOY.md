# Pi Trained Model Deployment (AirPuff AP)

## Why we do not run the RKNN demo directly on Pi
- `MonocularDistanceDetect-YOLOV5-RKNN-CPP-MultiThread` depends on Rockchip RKNN runtime (`librknnrt.so`) and is marked as `rk3588/rk3588s` oriented.
- Raspberry Pi does not provide RKNN NPU runtime, so the original RKNN binary path is not directly runnable on Pi.

## What is implemented in this repo
- AirPuff runtime defaults are switched to AP control-page address:
  - `http://10.42.0.1:5000/api/sense`
- Added Pi AP setup script:
  - `pi_configure_airpuff_ap.sh`
  - Configures Pi static IP `10.42.0.82/24` for SSID `AirPuff_AP`.
  - Rewrites `pi_airpuff_video_stream.service` target to `10.42.0.1:5000`.
- Added trained-model startup script on laptop:
  - `laptop_start_trained_yolo.sh`
  - Supports custom YOLO weights and monocular ground-plane distance mode.
- Added monocular geometry distance mode into `airpuff_server.py`:
  - `AIRPUFF_YOLO_DISTANCE_MODE=ground_plane|hybrid|size_prior`
  - Geometry parameters:
    - `AIRPUFF_YOLO_CAMERA_HEIGHT_M`
    - `AIRPUFF_YOLO_CAMERA_PITCH_DEG`
    - `AIRPUFF_YOLO_CY_PX`
    - `AIRPUFF_YOLO_FY_PX`

## Run order
1. On Pi (local terminal):
```bash
sudo ~/airpuff/pi_configure_airpuff_ap.sh 10.42.0.1 10.42.0.82 AirPuff_AP
```

2. On laptop:
```bash
~/airpuff/laptop_start_trained_yolo.sh /home/walter/airpuff/models/best.pt full
```

3. Open control page:
```text
http://10.42.0.1:5000/
```

## Notes
- If your calibrated pitch/height differs, set:
```bash
export AIRPUFF_YOLO_CAMERA_HEIGHT_M=0.935
export AIRPUFF_YOLO_CAMERA_PITCH_DEG=5.16
```
- If intrinsic values are known, optionally set:
```bash
export AIRPUFF_YOLO_FY_PX=<fy_px>
export AIRPUFF_YOLO_CY_PX=<cy_px>
```
